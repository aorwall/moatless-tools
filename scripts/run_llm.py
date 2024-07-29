import os
import json
from argparse import ArgumentParser, Action
from moatless.utils_search.misc import deep_get
from moatless.search.reward import LLM_Value_Function
from tqdm import tqdm
from datetime import datetime


class ParseDict(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = {}
        for i in range(0, len(values), 2):
            if i+1 < len(values):
                d[values[i]] = values[i+1]
        setattr(namespace, self.dest, d)


def load_message(path):
    with open(path, "r") as f:
        return json.load(f)

def set_environment_variables(keys):
    for name, value in keys.items():
        os.environ[name] = value
        
def create_run_id(tag, value_args):
    if tag:
        run_id = tag
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add value_args to the run_id
    if value_args:
        args_str = '_'.join(f"{k}-{v}" for k, v in value_args.items())
        run_id = f"{run_id}_{args_str}"
    
    return run_id

def eval_trajectory(value_function, trajectory, 
                    history=False, val_args={}):
    print(f"trajectory_keys: {trajectory.keys()}")
    init_msg_key = "initial_message" if "initial_message" in trajectory else "input.problem_statement"
    problem_statement = deep_get(trajectory, init_msg_key)
    steps = trajectory["transitions"] if "transitions" in trajectory else trajectory["steps"]
    rewards = []
    state_history = []

    for n, step in enumerate(steps):
        if "actions" not in step:
            continue
        if len(step["actions"]) == 0:
            continue
        actions = step["actions"][-1]
        state_info = step["name"] + "\n\n" + str(step.get("state", ""))
        state_message = str(actions["action"]) if "action" in actions else str(actions["input"])
        state_response = str(actions.get("output", None))

        current_state = {
            "state_info": state_info,
            "state_message": state_message,
            "state_response": state_response,
            "step_count": n,
            "node_id": n
        }
        if val_args.get("state_file_context", None):
            current_state["state_file_context"] = step["file_context"]
        
        merged_args = {**current_state, **val_args}
            
        if history:
            reward = value_function.get_reward(
                problem_statement=problem_statement,
                state_history=state_history,
                **merged_args,
            )
            state_history.append(current_state)
        else:
            reward = value_function.get_reward(
                problem_statement=problem_statement,
                **merged_args,
            )

        rewards.append(reward)

    return rewards

def process_file(task, model, file_path, run_id=None, **kwargs):
    message = load_message(file_path)
    filename = os.path.basename(file_path)
    
    # Create a new subdirectory with the run_id
    if run_id:
        save_dir = os.path.dirname(file_path).replace("trajs", os.path.join(task, run_id))
    else:
        save_dir = os.path.dirname(file_path).replace("trajs", task)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    value_function = LLM_Value_Function(model=model, filename=save_path)
    print(f"save_path: {save_path}")
    
    if task in ["rews", "rews_traj"]:
        out = eval_trajectory(value_function, message,
                              history=(task == "rews_traj"),
                              **kwargs)
    elif task == "tree_eval":
        out = value_function.eval_tree(message)
    
    print(f"Results for {save_path} with model {model}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--message", type=str, help="Message file or directory to process")
    parser.add_argument("--task", type=str, help="Task to run", required=True, choices=["rews", "tree_eval", "rews_traj"])
    parser.add_argument("--instances", nargs='+', help="Specify one or more instances", default=None)
    parser.add_argument("--models", nargs='+', help="Specify one or more models", required=True)
    parser.add_argument("--tag", type=str, help="Optional tag for the run", default="")
    parser.add_argument('--val_args', nargs='*', action=ParseDict, default={},
                        help='Additional arguments for value_function in the format: key1 value1 key2 value2 ...')
    args = parser.parse_args()
    
    keys_dir = "../keys.json"
    with open(keys_dir) as f:
        keys = json.load(f)
    set_environment_variables(keys)

    run_id = create_run_id(args.tag, args.val_args)
    print(f"instances: {args.instances}")
    if os.path.isfile(args.message):
        for model in args.models:
            process_file(args.task, model, args.message, run_id,
                         val_args=args.val_args)
    elif os.path.isdir(args.message):
        for model in args.models:
            pbar = tqdm(os.listdir(args.message), desc="Processing files")
            for filename in pbar:
                if args.instances:
                    if not any(instance in filename for instance in args.instances):
                        continue
                if filename.endswith(".json"):  # Assuming the files are JSON
                    file_path = os.path.join(args.message, filename)
                    print(f"file_path: {file_path}")
                    process_file(args.task, model, file_path, run_id, 
                                 val_args=args.val_args)