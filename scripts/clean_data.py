import os
import shutil
from PIL import Image
import logging
import hashlib
import json
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
RAW_DATA_DIR = 'data/raw/'
PROCESSED_DATA_DIR = 'data/processed/'
ANNOTATIONS_DIR = 'data/annotations/'
METADATA_FILE = 'data/processed/metadata.json'

# Metadata structure
metadata = {}

def create_directory_structure():
    """
    Create necessary directories if they do not exist.
    """
    dirs = [PROCESSED_DATA_DIR, ANNOTATIONS_DIR]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")

def is_image_corrupted(image_path):
    """
    Check if the image file is corrupted by attempting to open it.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except (IOError, SyntaxError) as e:
        logging.error(f"Corrupted image detected: {image_path}, error: {str(e)}")
        return True

def generate_image_hash(image_path):
    """
    Generate a hash for the image to track duplicates.
    """
    hasher = hashlib.md5()
    with open(image_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def log_metadata(image_path, dest_path, image_hash):
    """
    Log metadata for processed images.
    """
    metadata_entry = {
        'original_path': image_path,
        'processed_path': dest_path,
        'hash': image_hash,
        'timestamp': time.time()
    }
    metadata[os.path.basename(image_path)] = metadata_entry
    logging.info(f"Metadata logged for image: {os.path.basename(image_path)}")

def save_metadata():
    """
    Save metadata to a JSON file after processing.
    """
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata saved to {METADATA_FILE}")

def clean_raw_data():
    """
    Clean the raw data directory by removing corrupted images and organizing them.
    """
    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image_corrupted(file_path):
                os.remove(file_path)
                logging.info(f"Removed corrupted image: {file_path}")
            else:
                image_hash = generate_image_hash(file_path)
                if is_duplicate(image_hash):
                    logging.warning(f"Duplicate image detected and skipped: {file_path}")
                else:
                    organize_image(file_path, image_hash)

def is_duplicate(image_hash):
    """
    Check if the image is a duplicate based on its hash.
    """
    for entry in metadata.values():
        if entry['hash'] == image_hash:
            return True
    return False

def organize_image(image_path, image_hash):
    """
    Organize images by moving them to the processed data directory and log metadata.
    """
    dest_path = os.path.join(PROCESSED_DATA_DIR, os.path.basename(image_path))
    shutil.move(image_path, dest_path)
    logging.info(f"Moved image: {image_path} to {dest_path}")
    log_metadata(image_path, dest_path, image_hash)

def validate_annotation_file(annotation_path):
    """
    Validate annotation files to check for integrity.
    """
    if not os.path.exists(annotation_path):
        logging.warning(f"Missing annotation file: {annotation_path}")
        return False
    return True

def clean_annotation_files():
    """
    Clean and validate annotation files in the annotations directory.
    """
    for root, _, files in os.walk(ANNOTATIONS_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            if not validate_annotation_file(file_path):
                os.remove(file_path)
                logging.info(f"Removed invalid annotation file: {file_path}")

def remove_empty_directories(directory):
    """
    Remove empty directories in the given directory path.
    """
    for root, dirs, _ in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                logging.info(f"Removed empty directory: {dir_path}")

def generate_summary_report():
    """
    Generate a summary report of the cleaning process.
    """
    report = {
        'total_images_processed': len(metadata),
        'unique_images': len(set([entry['hash'] for entry in metadata.values()])),
        'corrupted_images_removed': sum([entry['hash'] == 'corrupted' for entry in metadata.values()]),
        'total_annotations_validated': len(os.listdir(ANNOTATIONS_DIR))
    }
    report_path = 'data/processed/cleaning_summary.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    logging.info(f"Summary report generated: {report_path}")

def backup_processed_data():
    """
    Create a backup of the processed data directory.
    """
    backup_dir = PROCESSED_DATA_DIR.rstrip('/') + '_backup'
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(PROCESSED_DATA_DIR, backup_dir)
    logging.info(f"Backup created for processed data: {backup_dir}")

def restore_backup():
    """
    Restore the processed data from backup if necessary.
    """
    backup_dir = PROCESSED_DATA_DIR.rstrip('/') + '_backup'
    if os.path.exists(backup_dir):
        shutil.rmtree(PROCESSED_DATA_DIR)
        shutil.copytree(backup_dir, PROCESSED_DATA_DIR)
        logging.info(f"Restored data from backup: {backup_dir}")
    else:
        logging.error(f"Backup not found: {backup_dir}")

def main():
    """
    Main function to run the data cleaning pipeline.
    """
    logging.info("Starting data cleaning pipeline...")
    create_directory_structure()
    
    # backup/restore process
    backup_processed_data()
    
    # Clean and organize raw data
    clean_raw_data()

    # Validate and clean annotations
    clean_annotation_files()

    # Remove empty directories
    remove_empty_directories(RAW_DATA_DIR)
    remove_empty_directories(PROCESSED_DATA_DIR)

    # Generate metadata and summary reports
    save_metadata()
    generate_summary_report()

    logging.info("Data cleaning pipeline completed successfully.")

if __name__ == "__main__":
    main()