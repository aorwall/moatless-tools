import json
import os
import sys
from graphviz import Digraph
import textwrap

def wrap_text(text, width=40):
    return '\n'.join(textwrap.wrap(text, width=width))

def create_graph(data):
    dot = Digraph(comment='AgenticLoop Process', format='png')
    dot.attr(rankdir='TB', size='100,100')  # Greatly increased size
    dot.attr('graph', dpi='600')  # Set DPI to 600 for very high-resolution PNG

    # Add initial message node
    dot.node('initial', wrap_text(data['initial_message']), shape='box')

    prev_node = 'initial'
    for i, transition in enumerate(data['transitions']):
        transition_name = transition['name']
        transition_id = f"transition_{i}"
        
        # Add transition node
        dot.node(transition_id, transition_name, shape='diamond')
        dot.edge(prev_node, transition_id)

        for j, action in enumerate(transition['actions']):
            action_id = f"{transition_id}_action_{j}"
            
            # Extract thoughts and output
            thoughts = action['action'].get('thoughts', '')
            output = action['output']
            if isinstance(output, dict):
                output = json.dumps(output, indent=2)
            elif isinstance(output, str):
                output = output
            else:
                output = str(output)

            # Combine thoughts and output
            action_text = f"Thoughts:\n{thoughts}\n\nOutput:\n{output}"
            
            # Add action node
            dot.node(action_id, wrap_text(action_text), shape='box')
            dot.edge(transition_id, action_id)

        prev_node = action_id

    return dot

def main(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    graph = create_graph(data)
    
    # Save as high-resolution PNG
    base_dir = os.path.dirname(os.path.dirname(file_path))
    save_dir = os.path.join(base_dir, 'flow_chart')
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(file_path).replace('.json', '')
    save_path = os.path.join(save_dir, filename)
    graph.render(save_path, cleanup=True, format='png')
    print(f"Graph has been saved as {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_json_file>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)
                main(file_path)
    elif os.path.isfile(folder_path):
        main(folder_path)
