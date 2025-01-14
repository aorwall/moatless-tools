import os
import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set
from moatless_tools.schema import EvaluationResponseDTO

def load_solvable_instances() -> Set[str]:
    """Load the set of instance IDs from lite_and_verified_solvable dataset."""
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "lite_and_verified_solvable_dataset.json")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
        return set(dataset['instance_ids'])

def load_evaluation_response(file_path: str) -> EvaluationResponseDTO:
    """Load and parse an evaluation response file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        return EvaluationResponseDTO(**data)

def analyze_evaluations(evals_dir: str) -> pd.DataFrame:
    # Load solvable instances
    solvable_instances = load_solvable_instances()
    
    # Dictionary to store instance data
    instance_data = defaultdict(lambda: {
        'resolutionRate': None,
        'resolvedCount': 0,
        'statuses': {},
        'models': set()  # Track which models attempted this instance
    })
    
    # Track model-specific data
    model_data = defaultdict(lambda: {
        'eval_count': 0,
        'resolved_instances': set(),
        'total_instances': set()
    })
    
    # Walk through the evals directory
    for root, dirs, files in os.walk(evals_dir):
        if 'evaluation_response.json' in files:
            eval_name = os.path.basename(root)
            response_path = os.path.join(root, 'evaluation_response.json')
            
            try:
                response = load_evaluation_response(response_path)
                model_name = response.settings.model
                
                # Process each instance in the evaluation
                for instance in response.instances:
                    # Skip instances not in the solvable dataset
                    if instance.instanceId not in solvable_instances:
                        continue
                        
                    instance_data[instance.instanceId]['resolutionRate'] = instance.resolutionRate
                    instance_data[instance.instanceId]['models'].add(model_name)
                    
                    # Store status for this evaluation
                    instance_data[instance.instanceId]['statuses'][eval_name] = instance.status
                    
                    # Update model-specific data
                    model_data[model_name]['eval_count'] += 1
                    model_data[model_name]['total_instances'].add(instance.instanceId)
                    
                    # Count if resolved
                    if instance.status == "resolved":
                        instance_data[instance.instanceId]['resolvedCount'] += 1
                        model_data[model_name]['resolved_instances'].add(instance.instanceId)
                        
            except Exception as e:
                print(f"Error processing {response_path}: {e}")
    
    # Convert to DataFrame format
    rows = []
    for instance_id, data in instance_data.items():
        row = {
            'instanceId': instance_id,
            'resolutionRate': data['resolutionRate'],
            'resolvedCount': data['resolvedCount'],
            'models': ','.join(sorted(data['models']))  # Add models that attempted this instance
        }
        # Add status columns for each evaluation
        row.update(data['statuses'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort columns to ensure instanceId, resolutionRate, resolvedCount, and models come first
    status_cols = [col for col in df.columns if col not in ['instanceId', 'resolutionRate', 'resolvedCount', 'models']]
    df = df[['instanceId', 'resolutionRate', 'resolvedCount', 'models'] + status_cols]
    
    return df, model_data

def main():
    evals_dir = "evals"
    output_file = "evaluation_analysis.csv"
    
    df, model_data = analyze_evaluations(evals_dir)
    df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results written to {output_file}")
    
    # Print some basic statistics
    print("\nBasic Statistics:")
    print(f"Total instances analyzed: {len(df)}")
    
    # Count number of evaluations (excluding the fixed columns)
    num_evals = len([col for col in df.columns if col not in ['instanceId', 'resolutionRate', 'resolvedCount', 'models']])
    print(f"Number of evaluations: {num_evals}")
    
    # Count instances resolved by any evaluation
    resolved_by_any = len(df[df['resolvedCount'] > 0])
    resolved_ratio = resolved_by_any / len(df) if len(df) > 0 else 0
    print(f"Instances resolved by at least one evaluation: {resolved_by_any}/{len(df)} ({resolved_ratio:.1%})")
    
    print(f"Average resolution rate: {df['resolutionRate'].mean():.2%}")
    print(f"Average resolved count: {df['resolvedCount'].mean():.2f}")
    
    # Print per-model statistics
    print("\nPer-Model Statistics:")
    for model, data in sorted(model_data.items()):
        resolved_count = len(data['resolved_instances'])
        total_count = len(data['total_instances'])
        resolved_ratio = resolved_count / total_count if total_count > 0 else 0
        print(f"\n{model}:")
        print(f"  Number of evaluations: {data['eval_count']}")
        print(f"  Instances resolved: {resolved_count}/{total_count} ({resolved_ratio:.1%})")
    
    # Print distribution of resolved counts
    print("\nResolution Distribution:")
    resolved_dist = df['resolvedCount'].value_counts().sort_index()
    for count, instances in resolved_dist.items():
        print(f"Resolved by {count} evaluation{'s' if count != 1 else ''}: {instances} instance{'s' if instances != 1 else ''}")

if __name__ == "__main__":
    main() 