import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_eval_results(csv_path: str) -> pd.DataFrame:
    """Load and preprocess evaluation results from CSV."""
    df = pd.read_csv(csv_path)
    # Extract model version and date from filename
    filename = Path(csv_path).stem
    date, model = filename.split('-', 1)[0], '-'.join(filename.split('-')[1:])
    df['model'] = model
    df['date'] = date
    return df

def compare_indices(csv_paths: list[str], output_dir: str):
    """Compare multiple index evaluation results."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load all dataframes
    dfs = [load_eval_results(csv_path) for csv_path in csv_paths]
    combined_df = pd.concat(dfs)
    
    # Basic statistics
    stats = []
    for model in combined_df['model'].unique():
        model_df = combined_df[combined_df['model'] == model]
        stats.append({
            'model': model,
            'total_instances': len(model_df),
            'avg_all_window': model_df['all_matching_context_window'].mean(),
            'avg_any_window': model_df['any_matching_context_window'].mean(),
            'found_all': (model_df['all_matching_context_window'].notna()).sum(),
            'found_any': (model_df['any_matching_context_window'].notna()).sum(),
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f"{output_dir}/comparison_stats.csv", index=False)
    print("\nOverall Statistics:")
    print(stats_df.to_string(index=False))
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    
    # Box plot for context window sizes
    plt.subplot(1, 2, 1)
    sns.boxplot(data=combined_df.melt(id_vars=['model'], 
                                    value_vars=['all_matching_context_window', 'any_matching_context_window'],
                                    var_name='metric', value_name='tokens'),
               x='model', y='tokens', hue='metric')
    plt.xticks(rotation=45)
    plt.title('Context Window Sizes by Model')
    
    # Success rate comparison
    plt.subplot(1, 2, 2)
    success_data = []
    for model in combined_df['model'].unique():
        model_df = combined_df[combined_df['model'] == model]
        total = len(model_df)
        success_data.append({
            'model': model,
            'metric': 'Found All Changes',
            'rate': (model_df['all_matching_context_window'].notna().sum() / total) * 100
        })
        success_data.append({
            'model': model,
            'metric': 'Found Any Change',
            'rate': (model_df['any_matching_context_window'].notna().sum() / total) * 100
        })
    
    success_df = pd.DataFrame(success_data)
    sns.barplot(data=success_df, x='model', y='rate', hue='metric')
    plt.xticks(rotation=45)
    plt.title('Success Rate by Model')
    plt.ylabel('Success Rate (%)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_plots.png")
    
    # Generate detailed instance-level comparison
    comparison = combined_df.pivot(index='instance_id', 
                                 columns='model',
                                 values=['all_matching_context_window', 'any_matching_context_window'])
    
    # Find instances with significant differences
    significant_diff = []
    for idx in comparison.index:
        row = comparison.loc[idx]
        models = combined_df['model'].unique()
        
        # Compare all_matching_context_window between models
        windows = [row[('all_matching_context_window', model)] for model in models]
        any_windows = [row[('any_matching_context_window', model)] for model in models]
        
        # Check if any model found nothing (both all and any are None)
        model_failures = []
        model_successes = []
        for i, model in enumerate(models):
            if pd.isna(windows[i]) and pd.isna(any_windows[i]):
                model_failures.append(model)
            else:
                model_successes.append(model)
        
        # If some models failed while others succeeded, record it
        if model_failures and model_successes:
            significant_diff.append({
                'instance_id': idx,
                'failed_models': ', '.join(model_failures),
                'successful_models': ', '.join(model_successes),
                **{f"{model}_all": row[('all_matching_context_window', model)] for model in models},
                **{f"{model}_any": row[('any_matching_context_window', model)] for model in models}
            })
    
    if significant_diff:
        diff_df = pd.DataFrame(significant_diff)
        diff_df.to_csv(f"{output_dir}/index_failures.csv", index=False)
        print(f"\nFound {len(significant_diff)} instances where some models failed while others succeeded")
        print("See index_failures.csv for details")
        print("\nSample of failures:")
        print(diff_df[['instance_id', 'failed_models', 'successful_models']].head().to_string())

def main():
    parser = argparse.ArgumentParser(description="Compare index evaluation results")
    parser.add_argument("csv_files", nargs="+", help="CSV files containing evaluation results")
    parser.add_argument("--output-dir", default="index_comparison", help="Directory for output files")
    
    args = parser.parse_args()
    compare_indices(args.csv_files, args.output_dir)

if __name__ == "__main__":
    main() 