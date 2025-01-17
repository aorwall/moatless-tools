import argparse
import asyncio
import json
import logging
import os
import sys
import time
import queue
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.logging import RichHandler
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from moatless.benchmark.evaluation_v2 import (
    EvaluationStatus, InstanceStatus, create_evaluation_name,
    TreeSearchSettings, EvaluationRunner
)
from moatless.benchmark.repository import EvaluationFileRepository
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.schema import MessageHistoryType
from moatless.agent.settings import AgentSettings
from moatless.benchmark.evaluation_factory import create_evaluation
from moatless.benchmark.schema import EvaluationDatasetSplit, EvaluationInstance
from typing import Optional

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Set up file logging in evaluation directory
def setup_logging(evaluation_dir: str):
    # Create logs directory within evaluation directory
    logs_dir = os.path.join(evaluation_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f"evaluation_{timestamp}.log")
    error_log_file = os.path.join(logs_dir, f"evaluation_errors_{timestamp}.log")

    # Main log file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Error log file handler - only logs ERROR and above
    error_file_handler = logging.FileHandler(error_log_file)
    error_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n%(exc_info)s'))
    error_file_handler.setLevel(logging.ERROR)
    logger.addHandler(error_file_handler)

logging.getLogger("LiteLLM").setLevel(logging.INFO)

load_dotenv()

class Event:
    def __init__(self, event_type: str, message: str, timestamp: datetime = None):
        self.event_type = event_type
        self.message = message
        self.timestamp = timestamp or datetime.now()

class EventPanel:
    def __init__(self, max_events=100, visible_events=20):
        self.events = deque(maxlen=max_events)
        self.visible_events = visible_events
    
    def add_event(self, event_type: str, message: str):
        self.events.append(Event(event_type, message))
    
    def get_panel(self):
        # Get the most recent events up to visible_events
        visible_events = list(self.events)[-self.visible_events:]
        visible_events.reverse()  # Reverse to show newest first
        
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Time", style="cyan", width=8)
        table.add_column("Type", style="magenta", width=12)
        table.add_column("Message", style="white")
        
        for event in visible_events:
            time_str = event.timestamp.strftime("%H:%M:%S")
            table.add_row(
                time_str,
                event.event_type,
                event.message
            )
        
        return Panel(
            table,
            title=f"Events (newest first, showing {len(visible_events)} of {len(self.events)} events)",
            border_style="blue"
        )

class UILogger(logging.Handler):
    def __init__(self, log_panel):
        super().__init__()
        self.log_panel = log_panel
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_panel.write(msg)
        except Exception:
            self.handleError(record)

class EvaluationMonitor:
    def __init__(self, repository, evaluation):
        self.repository = repository
        self.evaluation = evaluation
        self.console = Console()
        self.start_time = time.time()
        self.instances_data = {}
        self.live = None
        self.event_panel = EventPanel(max_events=1000, visible_events=25)
        self.logger = logger
        self.event_queue = queue.Queue()
        self.needs_update = False
        self.needs_stats_update = False
        self.last_event_update = 0
        
        # Load initial instances
        for instance in self.repository.list_instances(self.evaluation.evaluation_name):
            self.instances_data[instance.instance_id] = instance
        
        self.logger.info(f"Starting evaluation monitor for {evaluation.evaluation_name}")
        self.logger.info(f"Found {len(self.instances_data)} instances to evaluate")

    def handle_event(self, event):
        """Handle evaluation events by putting them in the queue"""
        try:
            self.event_queue.put_nowait(event)
            self.needs_update = True  # Mark that we need an update
        except queue.Full:
            self.logger.warning("Event queue is full, dropping event")

    async def process_event(self, event):
        """Process a single event"""
        event_type = event.event_type
        data = event.data if event.data else {}
        
        instance_id = data.get("instance_id")
        
        if event_type == "evaluation_started":
            self.event_panel.add_event("START", "Evaluation started")
            self.logger.info("Evaluation started")
            return

        if not instance_id and event_type != "evaluation_started":
            self.logger.warning(f"Instance ID not found in event data: {data}")
            return

        # Load/reload instance from repository to get latest state
        instance = self.repository.load_instance(self.evaluation.evaluation_name, instance_id)
        if instance:
            self.instances_data[instance_id] = instance
            self.needs_stats_update = True
            
            if event_type == "instance_started":
                self.event_panel.add_event("START", f"Started instance: {instance_id}")
                self.logger.info(f"Started instance: {instance_id}")
            elif event_type == "instance_completed":
                status = "✓" if instance.resolved else "✗"
                self.event_panel.add_event("COMPLETE", f"Completed {instance_id} ({status})")
                self.logger.info(f"Completed instance: {instance_id} (resolved: {instance.resolved})")
            elif event_type == "instance_error":
                self.event_panel.add_event("ERROR", f"Error in {instance_id}: {instance.error}")
                self.logger.error(f"Error in instance {instance_id}: {instance.error}")
            elif event_type == "instance_rerun":
                self.event_panel.add_event("RE-RUN", f"Rerun {instance_id}")
                self.logger.info(f"Rerun {instance_id}")

    async def process_events(self):
        """Process events from the queue"""
        while True:
            try:
                # Check queue in a non-blocking way
                while not self.event_queue.empty():
                    event = self.event_queue.get_nowait()
                    await self.process_event(event)
                    self.event_queue.task_done()
                
                current_time = time.time()
                
                # Update events once per second
                if current_time - self.last_event_update >= 1.0:
                    self.needs_update = True
                    self.last_event_update = current_time
                
                # Update display if needed
                if self.live and (self.needs_update or self.needs_stats_update):
                    self.live.update(self._create_layout())
                    self.needs_update = False
                    self.needs_stats_update = False
                
                await asyncio.sleep(1.0)
            except queue.Empty:
                pass  # Queue is empty, continue
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
                await asyncio.sleep(1.0)

    def create_progress_table(self):
        """Create a rich table showing evaluation progress"""
        table = Table(title=f"Evaluation Progress: {self.evaluation.evaluation_name}")
        
        table.add_column("Instance ID", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Duration", style="green")
        table.add_column("Resolved", style="yellow")
        table.add_column("Iterations", style="blue")
        table.add_column("Cost", style="cyan")
        table.add_column("Tokens", style="blue")
        
        # Sort instances: running first (by start time desc), then others
        sorted_instances = []
        running_instances = []
        other_instances = []
        
        for instance in self.instances_data.values():
            if instance.status == InstanceStatus.STARTED:
                running_instances.append(instance)
            else:
                other_instances.append(instance)
        
        # Sort running instances by start time (most recent first)
        running_instances.sort(key=lambda x: x.started_at or datetime.min, reverse=True)
        sorted_instances = running_instances + other_instances
        
        for instance in sorted_instances:
            status = instance.status
            duration = f"{int(instance.duration)}s" if instance.duration else "-"
            resolved = "✓" if instance.resolved else "✗" if instance.resolved is False else "-"
            
            # Get iterations and tokens from benchmark result
            iterations = instance.iterations
            tokens = 0
            cost = "-"

            if instance.usage:
                tokens = (
                    instance.usage.prompt_tokens +
                    instance.usage.completion_tokens +
                    instance.usage.cached_tokens
                )
                cost = f"${instance.usage.completion_cost:.2f}"
            
            status_style = {
                'pending': 'white',
                'started': 'yellow',
                'completed': 'green',
                'error': 'red'
            }.get(status, 'white')
            
            table.add_row(
                instance.instance_id,
                Text(status, style=status_style),
                duration,
                resolved,
                str(iterations),
                cost,
                f"{tokens:,}" if tokens > 0 else "-"
            )
        
        return table

    def create_stats_panel(self):
        """Create a panel showing evaluation statistics"""
        total = len(self.instances_data)
        completed = sum(1 for i in self.instances_data.values() if i.status == InstanceStatus.COMPLETED)
        errors = sum(1 for i in self.instances_data.values() if i.status == InstanceStatus.ERROR)
        running = sum(1 for i in self.instances_data.values() if i.status == InstanceStatus.STARTED)
        resolved = sum(1 for i in self.instances_data.values() if i.resolved is True)
        
        # Calculate total cost and tokens
        total_cost = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        for instance in self.instances_data.values():
            if instance.benchmark_result:
                total_cost += instance.benchmark_result.total_cost
                total_prompt_tokens += instance.benchmark_result.prompt_tokens
                total_completion_tokens += instance.benchmark_result.completion_tokens
                total_cached_tokens += instance.benchmark_result.cached_tokens
        
        text = Text()
        text.append(f"Total Instances: {total}\n", style="cyan")
        text.append(f"Completed: {completed}\n", style="green")
        text.append(f"Running: {running}\n", style="yellow")
        text.append(f"Errors: {errors}\n", style="red")
        text.append(f"Success Rate: {(resolved/total*100 if total > 0 else 0):.1f}%\n", style="magenta")
        text.append(f"Total Cost: ${total_cost:.2f}\n", style="cyan")
        text.append(f"Total Tokens: {total_prompt_tokens + total_completion_tokens + total_cached_tokens:,}\n", style="blue")
        text.append(f"Elapsed Time: {self._format_elapsed_time()}\n", style="blue")
        
        return Panel(text, title="Evaluation Statistics", border_style="bright_blue")

    def create_info_panel(self):
        """Create a panel showing evaluation configuration"""
        evaluation = self.evaluation
        eval_dir = os.path.join(self.repository.evaluations_dir, evaluation.evaluation_name)

        text = Text()
        # Evaluation info
        text.append("Evaluation Info:\n", style="bold magenta")
        text.append(f"  Directory: ", style="cyan")
        text.append(f"{eval_dir}\n", style="white")
        
        # Model info
        text.append("\nModel Settings:\n", style="bold magenta")
        text.append(f"  Model: ", style="cyan")
        text.append(f"{evaluation.settings.model.model}\n", style="white")
        text.append(f"  Temperature: ", style="cyan")
        text.append(f"{evaluation.settings.model.temperature}\n", style="white")
        text.append(f"  Response Format: ", style="cyan")
        text.append(f"{evaluation.settings.model.response_format.value}\n", style="white")
        
        # Tree search settings
        text.append("\nTree Search Settings:\n", style="bold magenta")
        text.append(f"  Max Iterations: ", style="cyan")
        text.append(f"{evaluation.settings.max_iterations}\n", style="white")
        text.append(f"  Max Expansions: ", style="cyan")
        text.append(f"{evaluation.settings.max_expansions}\n", style="white")
        text.append(f"  Max Cost: ", style="cyan")
        text.append(f"{evaluation.settings.max_cost}\n", style="white")
        
        # Agent settings
        text.append("\nAgent Settings:\n", style="bold magenta")
        text.append(f"  Message History: ", style="cyan")
        text.append(f"{evaluation.settings.agent_settings.message_history_type.value}\n", style="white")
        
        return Panel(text, title="Evaluation Info", border_style="green")

    def _format_elapsed_time(self):
        """Format elapsed time in a human-readable format"""
        seconds = int(time.time() - self.start_time)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _create_layout(self):
        """Create the layout for the display"""
        layout = Layout()
        
        # Split into main content and right side with adjusted ratio (3:2 instead of 2:1)
        layout.split_row(
            Layout(name="main", ratio=3),
            Layout(name="right", ratio=2)
        )
        
        # Split main content into progress and stats with adjusted size
        layout["main"].split_column(
            Layout(self.create_progress_table(), name="progress"),
            Layout(self.create_stats_panel(), name="stats", size=10)
        )
        
        # Split right side into info and events with adjusted size for events
        layout["right"].split_column(
            Layout(self.create_info_panel(), name="info", size=18),
            Layout(self.event_panel.get_panel(), name="events")
        )
        
        return layout

    async def start_monitoring(self):
        """Start monitoring the evaluation"""
        with Live(
            self._create_layout(),
            console=self.console,
            refresh_per_second=1,
            auto_refresh=True
        ) as self.live:
            # Start event processing task
            event_task = asyncio.create_task(self.process_events())
            
            while True:
                try:
                    evaluation = self.repository.load_evaluation(self.evaluation.evaluation_name)
                    if not evaluation:
                        self.logger.error("Evaluation not found!")
                        break
                    
                    if evaluation.status in [EvaluationStatus.COMPLETED, EvaluationStatus.ERROR]:
                        # Force final update of stats
                        self.needs_stats_update = True
                        self.live.update(self._create_layout())
                        break
                        
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring evaluation: {e}")
                    break
            
            # Cancel event processing task
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass

def validate_evaluation_setup(repository, evaluation, args):
    """Validate evaluation setup and throw exceptions for any issues"""
    
    # Validate model configuration
    if not args.model:
        raise ValueError("Model name must be specified")
    
    # Check if evaluation exists
    if not evaluation:
        raise RuntimeError("Failed to create evaluation")

    return evaluation

def load_dataset_split(dataset_name: str) -> Optional[EvaluationDatasetSplit]:
    """Load a dataset split from the datasets directory."""
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", f"{dataset_name}_dataset.json")
    if not os.path.exists(dataset_path):
        return None
    
    with open(dataset_path) as f:
        data = json.load(f)
        return EvaluationDatasetSplit(**data)

def get_missing_instances(evaluation_dir: str, instance_ids: list[str]) -> list[str]:
    """Get instances that haven't been evaluated yet."""
    if not os.path.exists(evaluation_dir):
        return instance_ids
    
    # Get all evaluated instances from the evaluation directory
    evaluated_instances = set()
    for instance_dir in os.listdir(evaluation_dir):
        if os.path.isdir(os.path.join(evaluation_dir, instance_dir)):
            evaluated_instances.add(instance_dir)
    
    # Return instances that haven't been evaluated
    return [instance_id for instance_id in instance_ids if instance_id not in evaluated_instances]

def add_new_instances(repository: EvaluationFileRepository, evaluation_name: str, instance_ids: list[str]):
    """Create and save new instances for an existing evaluation."""
    for instance_id in instance_ids:
        eval_instance = EvaluationInstance(instance_id=instance_id)
        repository.save_instance(evaluation_name, eval_instance)

def main():
    parser = argparse.ArgumentParser(description="Run a model evaluation with progress monitoring")

    # Model and basic settings
    parser.add_argument("--model", required=True, help="Model name (e.g., gemini/gemini-2.0-flash-exp)")
    parser.add_argument("--api-key", help="API key for the model")
    parser.add_argument("--base-url", help="Base URL for the API")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum iterations per instance")
    parser.add_argument("--max-expansions", type=int, default=1, help="Maximum expansions per state")
    parser.add_argument("--max-cost", type=float, default=1.0, help="Maximum cost in tokens")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    
    # Model format settings
    parser.add_argument("--response-format", 
                       choices=[format.value for format in LLMResponseFormat],
                       help="Response format for the model")
    parser.add_argument("--message-history", 
                       choices=[history.value for history in MessageHistoryType],
                       help="Message history type")
    parser.add_argument("--thoughts-in-action",
                       action="store_true",
                       help="Enable thoughts in action for the agent")
    
    # Dataset split selection
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["easy", "lite", "verified", "lite_and_verified", "lite_and_verified_solvable", "small"],
        help="Dataset split to use for evaluation"
    )
    
    # Instance IDs override
    parser.add_argument(
        "--instance-ids",
        type=str,
        nargs="+",
        help="List of specific instance IDs to evaluate (overrides dataset split selection)"
    )
    
    # Evaluation name
    parser.add_argument("--evaluation-name", help="Custom evaluation name. If not provided, will be auto-generated")
    
    # Add rerun_errors flag
    parser.add_argument("--rerun-errors", action="store_true", help="Rerun instances that previously errored")
    
    args = parser.parse_args()
    
    # Initialize repository
    repository = EvaluationFileRepository(os.getenv("MOATLESS_DIR", "./evals"))
    
    # Create or get evaluation name
    if args.evaluation_name:
        evaluation_name = args.evaluation_name
    else:
        model_settings = CompletionModel(
            model=args.model,
            temperature=0.0,
            max_tokens=3000,
            api_key=args.api_key,
            base_url=args.base_url,
            response_format=LLMResponseFormat(args.response_format) if args.response_format else None,
            thoughts_in_action=args.thoughts_in_action
        )
        
        agent_settings = AgentSettings(
            completion_model=model_settings,
            message_history_type=MessageHistoryType(args.message_history) if args.message_history else MessageHistoryType.MESSAGES,
            system_prompt=None,
            thoughts_in_action=args.thoughts_in_action
        )
        
        tree_search_settings = TreeSearchSettings(
            max_iterations=args.max_iterations,
            max_expansions=args.max_expansions,
            max_cost=args.max_cost,
            model=model_settings,
            agent_settings=agent_settings
        )
        
        evaluation_name = create_evaluation_name(
            model=args.model,
            date=datetime.now().strftime("%Y%m%d"),
            max_expansions=args.max_expansions,
            response_format=LLMResponseFormat(args.response_format) if args.response_format else None,
            message_history=MessageHistoryType(args.message_history) if args.message_history else None,
            thoughts_in_action=args.thoughts_in_action
        )

        # Check for existing evaluation directory and modify name if needed
        base_evaluation_name = evaluation_name
        counter = 1
        while os.path.exists(os.path.join(repository.evaluations_dir, evaluation_name)):
            evaluation_name = f"{base_evaluation_name}_{counter}"
            counter += 1

    # Load dataset and get instance IDs
    if args.instance_ids:
        logger.info(f"Using provided instance IDs: {args.instance_ids}")
        instance_ids = args.instance_ids
    else:
        dataset = load_dataset_split(args.split)
        if dataset is None:
            logger.error(f"Dataset split '{args.split}' not found")
            sys.exit(1)
        instance_ids = dataset.instance_ids
        logger.info(f"Using instance IDs from dataset split: {args.split}")

    # Check if evaluation exists
    existing_evaluation = repository.load_evaluation(evaluation_name)
    if existing_evaluation:
        logger.info(f"Using existing evaluation: {evaluation_name}")
        
        eval_dir = os.path.join(repository.evaluations_dir, evaluation_name)
        setup_logging(eval_dir)
        
        missing_instances = get_missing_instances(eval_dir, instance_ids)
        logger.info(f"Adding {len(missing_instances)} new instances to evaluation")
        
        add_new_instances(repository, evaluation_name, missing_instances)
        evaluation = existing_evaluation
    else:
        logger.info(f"Creating new evaluation with name: {evaluation_name}")
        eval_dir = os.path.join(repository.evaluations_dir, evaluation_name)
        setup_logging(eval_dir)
        
        evaluation = create_evaluation(
            repository=repository,
            evaluation_name=evaluation_name,
            settings=tree_search_settings,
            instance_ids=instance_ids
        )

    try:
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create monitor
        monitor = EvaluationMonitor(repository, evaluation)
        
        # Create runner with event handler
        runner = EvaluationRunner(
            repository=repository,
            evaluation=evaluation,
            dataset_name="princeton-nlp/SWE-bench_Lite",
            num_workers=args.num_workers,
            use_testbed=True
        )
        
        # Add event handler
        runner.add_event_handler(monitor.handle_event)
        
        # Create monitoring task
        monitoring_task = loop.create_task(monitor.start_monitoring())
        
        logger.info("Running evaluation")
        # Run evaluation in executor and wait for both tasks
        loop.run_until_complete(asyncio.gather(
            loop.run_in_executor(ThreadPoolExecutor(), lambda: runner.run_evaluation(rerun_errors=args.rerun_errors)),
            monitoring_task
        ))
    except Exception as e:
        # Use rich to print error in red
        console = Console()
        console.print(f"[red]Error: {str(e)}")
        console.print("[red]Traceback:")
        console.print_exception()
        # Log the error with full stack trace
        logger.error("Fatal error in evaluation", exc_info=True)
        sys.exit(1)
    finally:
        if 'loop' in locals():
            loop.close()

if __name__ == "__main__":
    main()