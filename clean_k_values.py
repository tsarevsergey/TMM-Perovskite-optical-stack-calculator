import os
import glob
import pandas as pd

materials_dir = 'materials'
# Target the specific files we just created
target_files = glob.glob(os.path.join(materials_dir, '*-2.csv'))

for file_path in target_files:
    print(f"Cleaning k-values in {os.path.basename(file_path)}...")
    try:
        df = pd.read_csv(file_path)
        
        # Apply threshold: k < 0.02 -> 0.0
        # Count how many changed
        original_non_zeros = (df['k'] > 0).sum()
        df.loc[df['k'] < 0.02, 'k'] = 0.0
        new_non_zeros = (df['k'] > 0).sum()
        
        print(f"  Set {original_non_zeros - new_non_zeros} points to zero.")
        
        df.to_csv(file_path, index=False)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
