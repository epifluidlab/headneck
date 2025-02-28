import os
import numpy as np
from sklearn.model_selection import KFold

def generate_cv_splits(file_path, output_dir, n_splits=10):
    """
    Generates 10-fold cross-validation splits from a list of train IDs and saves each split to a file.

    Parameters:
    file_path (str): Path to the file containing train IDs, one per line.
    output_dir (str): Directory where CV split files will be saved.
    n_splits (int): Number of folds for cross-validation (default is 10).
    """
    # Read train IDs from the file
    with open(file_path, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]

    # Convert train IDs to a numpy array
    train_ids = np.array(train_ids)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create KFold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Generate and save splits
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_ids)):
        train_split = train_ids[train_idx]
        val_split = train_ids[val_idx]

        # Save train and validation splits to separate files
        train_file = os.path.join(output_dir, f'train_fold_{fold}.txt')
        val_file = os.path.join(output_dir, f'val_fold_{fold}.txt')

        with open(train_file, 'w') as f:
            f.write('\n'.join(train_split))

        with open(val_file, 'w') as f:
            f.write('\n'.join(val_split))

    print(f"Saved {n_splits}-fold CV splits to {output_dir}")

# Example usage:
# generate_cv_splits('train_ids.txt', 'cv_splits')
