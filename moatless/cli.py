#!/usr/bin/env python3
"""Modern CLI interface for running and monitoring Moatless validation flows."""

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import json
from rich.pretty import pprint

from moatless.runner import agentic_runner
from moatless.events import BaseEvent, event_bus, SystemEvent
from moatless.validation.code_flow_validation import CodeFlowValidation
from moatless.config.model_config import get_model_config, get_all_configs
from moatless.config.agent_config import get_config as get_agent_config, get_all_configs as get_all_agent_configs

logger = logging.getLogger(__name__)
console = Console()

def list_available_configs():
    """Print available models and agents in a formatted table."""
    # Models table
    models_table = Table(title="Available Models")
    models_table.add_column("Model ID")
    models_table.add_column("Description")
    
    for model_id, config in get_all_configs().items():
        models_table.add_row(
            model_id,
            config.get('description', 'No description available')
        )
    
    # Agents table    
    agents_table = Table(title="Available Agents")
    agents_table.add_column("Agent ID")
    agents_table.add_column("Description")
    
    for agent_id, config in get_all_agent_configs().items():
        agents_table.add_row(
            agent_id,
            config.description if hasattr(config, 'description') else 'No description available'
        )
    
    console.print(models_table)
    console.print("\n")
    console.print(agents_table)

def setup_logging(verbose: bool, log_dir: str):
    """Configure logging with appropriate level and format."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Reset handlers
    logging.getLogger().handlers = []

    # Set formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handlers - full debug output
    file_handler = logging.FileHandler(os.path.join(log_dir, 'validation.log'))
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Error log file
    error_handler = logging.FileHandler(os.path.join(log_dir, 'validation_errors.log'))
    error_handler.setFormatter(file_formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Configure root logger - for general application logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Configure LiteLLM logger - minimize console output
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

class ValidationCLI:
    def __init__(self):
        self.active_runs: dict[str, dict] = {}
        self.console = Console()
        self.validator = CodeFlowValidation()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )
        self.events_file = None
        event_bus.subscribe(self.handle_event_sync)
        # event_bus.subscribe(self.handle_event)  # Keep async handler
        
    def handle_event_sync(self, run_id: str, event: BaseEvent):
        """Synchronous event handler for immediate console output"""
        # Get event data
        event_data = event.model_dump()
        
        # Extract node_id if present
        node_id = event_data.get('node_id')
        node_str = f" [Node{node_id}]" if node_id else ""
        
        # Basic event info
        self.console.print(f"\n[cyan]Event[/cyan] [{run_id}]{node_str} {event.event_type}")
        
        # Special handling for different event types
        if event.event_type == "agent_started":
            self.console.print(f"[bright_black]Starting agent: {event_data.get('agent_id')} on node {event_data.get('node_id')}[/bright_black]")
            
        elif event.event_type == "agent_action_created":
            self.console.print(f"[bright_black]Action created{node_str}: {event_data.get('action_name')}[/bright_black]")
            if event_data.get('action_params'):
                self.console.print("Parameters:", style="bright_black")
                self.console.print(event_data['action_params'], style="bright_black")
                
        elif event.event_type == "agent_action_executed":
            self.console.print(f"[bright_black]Action executed{node_str}: {event_data.get('action_name')}[/bright_black]")
            if event_data.get('observation'):
                self.console.print("Observation:", style="bright_black")
                self.console.print(event_data['observation'], style="bright_black")
                
        elif event.event_type == "loop_started":
            self.console.print(f"[bright_black]Starting loop with node {event_data.get('initial_node_id')}[/bright_black]")
            
        elif event.event_type == "loop_completed":
            self.console.print(
                f"[bright_black]Loop completed: {event_data.get('total_iterations')} iterations, "
                f"${event_data.get('total_cost', 0):.4f} cost[/bright_black]"
            )
        
        # For any other events, show the full data
        else:
            if event_data:
                self.console.print(event_data, style="bright_black")

    async def handle_event(self, run_id: str, event: BaseEvent):
        """Async event handler for file writing and state updates"""
        if run_id not in self.active_runs:
            self.active_runs[run_id] = {
                'status': 'initializing',
                'iterations': 0,
                'cost': 0.0,
                'current_action': None,
                'current_node': None
            }

    def create_status_table(self) -> Table:
        """Create a rich table showing current validation status."""
        table = Table(title="Validation Status")
        table.add_column("Run ID")
        table.add_column("Status")
        table.add_column("Iterations")
        table.add_column("Cost ($)")
        table.add_column("Current Action")

        for run_id, data in self.active_runs.items():
            table.add_row(
                run_id,
                data['status'],
                str(data['iterations']),
                f"{data['cost']:.4f}",
                data['current_action'] or '-'
            )
        
        return table

    async def monitor_runs(self, run_ids: List[str]):
        """Monitor multiple validation runs with live updates."""
        with Live(self.create_status_table(), refresh_per_second=4) as live:
            while any(agentic_runner.get_run(run_id) for run_id in run_ids):
                live.update(self.create_status_table())
                await asyncio.sleep(0.25)
            
            # Final update
            live.update(self.create_status_table())

    async def run_validation(self, 
                           model_id: str,
                           agent_id: str,
                           instance_id: str,
                           max_iterations: int = 15):
        """Run validation using CodeFlowValidation."""
        # Generate run ID first
        run_id = f"validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Get the run directory and setup logging
        run_dir = self.validator.get_run_dir(run_id)
        setup_logging(True, os.path.join(run_dir, 'logs'))
        
        # Open events file
        events_path = os.path.join(run_dir, 'events.jsonl')
        self.console.print(f"[blue]Opening events file:[/blue] {events_path}")
        self.events_file = open(events_path, 'w', encoding='utf-8')
        
        try:
            # Start the code loop with our run_id
            self.validator.start_code_loop(
                run_id=run_id,
                agent_id=agent_id,
                model_id=model_id,
                instance_id=instance_id,
                max_iterations=max_iterations
            )
            
            # Monitor the run
            await self.monitor_runs([run_id])
        except Exception as e:
            self.console.print(f"[red]Error during validation:[/red] {e}")
            raise
        finally:
            # Ensure events file is closed
            if self.events_file:
                self.console.print("[blue]Closing events file[/blue]")
                self.events_file.close()
                self.events_file = None

def handle_error(e: Exception, msg: str = None):
    """Handle errors with proper logging and display."""
    if msg:
        console.print(f"\n[red]Error:[/red] {msg}")
    console.print(f"\n[red]Error:[/red] {str(e)}")
    console.print_exception(show_locals=True)
    sys.exit(1)

async def main():
    """Main entry point for the Moatless CLI."""
    try:
        parser = argparse.ArgumentParser(
            description="Modern CLI for running Moatless validation flows",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Create argument groups
        list_group = parser.add_argument_group('list options')
        list_group.add_argument(
            "--list",
            action="store_true",
            help="List available models and agents"
        )

        run_group = parser.add_argument_group('validation options')
        run_group.add_argument(
            "--model",
            help="Model to test (e.g., 'claude-3-5-sonnet-20241022')"
        )
        run_group.add_argument(
            "--agent",
            help="Agent configuration to use"
        )
        run_group.add_argument(
            "--instance",
            help="Instance ID to test"
        )
        run_group.add_argument(
            "--max-iterations",
            type=int,
            default=15,
            help="Maximum number of iterations"
        )

        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Enable verbose logging"
        )
        
        args = parser.parse_args()

        if args.list:
            list_available_configs()
            return

        # Validate required arguments for validation run
        if not all([args.model, args.agent, args.instance]):
            console.print("[red]Error:[/red] --model, --agent, and --instance are required for validation")
            console.print("\nFor available options, run: cli.py --list")
            sys.exit(1)

        # Validate agent config
        try:
            agent_config = get_agent_config(args.agent)
        except Exception as e:
            handle_error(e, f"Failed to get agent config for '{args.agent}'")

        # Validate model config
        try:
            model_config = get_model_config(args.model)
        except Exception as e:
            handle_error(e, f"Failed to get model config for '{args.model}'")

        # Run validation
        cli = ValidationCLI()
        try:
            await cli.run_validation(
                model_id=args.model,
                agent_id=args.agent,
                instance_id=args.instance,
                max_iterations=args.max_iterations
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Validation interrupted by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            handle_error(e, "Error during validation run")

    except Exception as e:
        handle_error(e, "Unexpected error in CLI")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        handle_error(e, "Fatal error in CLI execution") 