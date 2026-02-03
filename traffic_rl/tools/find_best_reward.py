import os
import pandas as pd
import glob

def find_best_agent(root_dir="."):
    print(f"🔍 Scanning {root_dir} for metrics.csv files...")
    
    results = []
    
    # Recursive search for all metrics.csv files
    files = glob.glob(os.path.join(root_dir, "**", "metrics.csv"), recursive=True)
    
    for csv_path in files:
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Check if empty
            if df.empty: continue
                
            # Get the folder name (Experiment ID)
            folder_name = os.path.dirname(csv_path)
            
            # Calculate Average Reward of the LAST 50 Episodes
            # (We care about how it performs at the END, not the beginning)
            last_50 = df.tail(50)
            avg_reward = last_50['reward'].mean()
            max_reward = df['reward'].max() # The single best episode ever
            
            results.append({
                "Folder": folder_name,
                "Final_Avg_Reward": avg_reward,
                "Peak_Reward": max_reward
            })
            
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    # Sort by Final Average Reward (Descending = Highest/Least Negative is best)
    results.sort(key=lambda x: x['Final_Avg_Reward'], reverse=True)
    
    print("\n" + "="*80)
    print(f"{'FOLDER':<60} | {'FINAL AVG (Last 50)':<20}")
    print("="*80)
    
    for res in results[:10]: # Print Top 10
        # Clean up folder name for display
        name = os.path.basename(res['Folder'])
        print(f"{name:<60} | {res['Final_Avg_Reward']:.2f}")

if __name__ == "__main__":
    find_best_agent()