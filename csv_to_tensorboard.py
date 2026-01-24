import os
import csv
import argparse
from torch.utils.tensorboard import SummaryWriter

def sync_csv_to_tensorboard(root_dir=".", force=False):
    """
    Scans for metrics.csv files and converts them to TensorBoard event files.
    """
    print(f"🔄 Scanning '{root_dir}' for CSV logs...")
    converted_count = 0
    skipped_count = 0

    # Walk through all directories
    for subdir, dirs, files in os.walk(root_dir):
        if "metrics.csv" in files:
            csv_path = os.path.join(subdir, "metrics.csv")
            
            # Check if TensorBoard file already exists
            has_tb_file = any("events.out.tfevents" in f for f in files)
            
            # Skip if exists and not forcing update
            if has_tb_file and not force:
                skipped_count += 1
                continue

            # --- CONVERSION LOGIC ---
            # We write the TB logs into the SAME folder as the CSV
            print(f"   Converting: {subdir}")
            writer = SummaryWriter(log_dir=subdir)
            
            try:
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            step = int(row['episode'])
                            
                            # Log standard metrics
                            if 'reward' in row and row['reward']:
                                writer.add_scalar("Train/Reward", float(row['reward']), step)
                            if 'avg_queue' in row and row['avg_queue']:
                                writer.add_scalar("Train/Avg_Queue", float(row['avg_queue']), step)
                            if 'epsilon' in row and row['epsilon']:
                                writer.add_scalar("Train/Epsilon", float(row['epsilon']), step)
                            
                            # Log Loss (handle None or empty strings)
                            if 'loss' in row and row['loss'] and row['loss'] != 'None':
                                writer.add_scalar("Train/Loss", float(row['loss']), step)
                                
                        except ValueError:
                            continue # Skip bad lines
            except Exception as e:
                print(f"   ❌ Error reading {csv_path}: {e}")
            finally:
                writer.close()
                converted_count += 1

    print(f"✅ Sync Complete.")
    print(f"   Converted: {converted_count}")
    print(f"   Skipped (Already exists): {skipped_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Root directory to scan")
    parser.add_argument("--force", action="store_true", help="Overwrite existing TB logs")
    args = parser.parse_args()
    
    sync_csv_to_tensorboard(args.dir, args.force)