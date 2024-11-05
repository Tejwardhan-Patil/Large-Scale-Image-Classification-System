import os
import shutil
import random
from pathlib import Path
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dir_structure(base_dir, categories):
    """
    Create directory structure for train, val, and test splits based on categories.
    """
    logging.info(f"Creating directory structure at {base_dir}")
    for category in categories:
        train_dir = os.path.join(base_dir, 'train', category)
        val_dir = os.path.join(base_dir, 'val', category)
        test_dir = os.path.join(base_dir, 'test', category)
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        Path(val_dir).mkdir(parents=True, exist_ok=True)
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Directories created for category: {category}")

def get_files_from_category(category_dir, extensions):
    """
    Get all files from the category directory with specified extensions.
    """
    logging.info(f"Fetching files from {category_dir}")
    all_files = []
    for ext in extensions:
        files = list(Path(category_dir).glob(f'*{ext}'))
        all_files.extend(files)
    return all_files

def move_files(file_paths, target_dir):
    """
    Move files to the target directory.
    """
    for file_path in file_paths:
        target_path = os.path.join(target_dir, file_path.name)
        logging.debug(f"Moving {file_path} to {target_path}")
        shutil.move(str(file_path), target_path)

def split_files(files, split_ratios):
    """
    Split files into train, val, and test sets based on the given ratios.
    """
    train_split = int(split_ratios['train'] * len(files))
    val_split = int(split_ratios['val'] * len(files)) + train_split
    return files[:train_split], files[train_split:val_split], files[val_split:]

def verify_split_ratios(ratios):
    """
    Verify that split ratios sum to 1.0.
    """
    total = sum(ratios.values())
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Split ratios must sum to 1.0. Received {total}")
    logging.info("Split ratios are valid.")

def validate_directory(directory):
    """
    Validate if a directory exists.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    logging.info(f"Directory {directory} validated.")

def validate_category_structure(data_dir):
    """
    Validate that there are subdirectories in the data directory.
    """
    categories = [d.name for d in Path(data_dir).iterdir() if d.is_dir()]
    if not categories:
        raise ValueError(f"No categories found in {data_dir}. Make sure data is organized into subdirectories.")
    logging.info(f"Categories found: {categories}")
    return categories

def ensure_non_empty_split(splits):
    """
    Ensure no split results in an empty dataset.
    """
    for split, files in splits.items():
        if not files:
            raise ValueError(f"{split} split is empty. Adjust the split ratios or check the data.")
    logging.info("All splits contain data.")

def split_data(data_dir, output_dir, split_ratios, extensions=['.jpg', '.png'], seed=42):
    """
    Split data into training, validation, and test sets based on given ratios.
    """
    logging.info(f"Splitting data from {data_dir} with ratios: {split_ratios}")
    random.seed(seed)
    
    # Validate input
    validate_directory(data_dir)
    verify_split_ratios(split_ratios)
    categories = validate_category_structure(data_dir)

    create_dir_structure(output_dir, categories)
    
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        files = get_files_from_category(category_dir, extensions)
        
        if not files:
            logging.warning(f"No files found for category {category}. Skipping...")
            continue
        
        random.shuffle(files)
        train_files, val_files, test_files = split_files(files, split_ratios)
        
        ensure_non_empty_split({'train': train_files, 'val': val_files, 'test': test_files})

        # Move files to respective directories
        move_files(train_files, os.path.join(output_dir, 'train', category))
        move_files(val_files, os.path.join(output_dir, 'val', category))
        move_files(test_files, os.path.join(output_dir, 'test', category))
        logging.info(f"Data split and moved for category {category}")

def configure_logging(log_file=None):
    """
    Configure logging output to a file.
    """
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging configured to output to {log_file}")

def generate_summary_report(output_dir):
    """
    Generate a summary report after the data has been split.
    """
    report = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        report[split] = {}
        for category in os.listdir(split_dir):
            category_dir = os.path.join(split_dir, category)
            num_files = len([f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))])
            report[split][category] = num_files
    logging.info(f"Summary report: {report}")
    return report

if __name__ == "__main__":
    # Configuration
    data_directory = "data/processed"
    output_directory = "data/split"
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    log_file = "data_split.log"
    file_extensions = ['.jpg', '.png']

    # Set up logging
    configure_logging(log_file)
    
    try:
        split_data(data_directory, output_directory, split_ratios, file_extensions)
        generate_summary_report(output_directory)
        logging.info("Data split completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during data splitting: {e}")