import os
from PIL import Image
import numpy as np
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Preprocessing configurations
IMAGE_SIZE = (224, 224)  # Default resize dimensions (width, height)
DATA_DIR = 'data/raw'  # Directory for raw images
PROCESSED_DIR = 'data/processed'  # Directory for processed images
NORMALIZE_TYPE = 'min_max'  # 'min_max' or 'z_score'

# Ensure the output directory exists
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def normalize_image(image_array, method='min_max'):
    """Normalize an image array using the specified method."""
    if method == 'min_max':
        return (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    elif method == 'z_score':
        mean = np.mean(image_array)
        std = np.std(image_array)
        return (image_array - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def preprocess_image(image_path, output_dir=PROCESSED_DIR, image_size=IMAGE_SIZE, normalize_type=NORMALIZE_TYPE):
    """Preprocess a single image: resizing, normalization, and saving."""
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Resize the image
            img_resized = img.resize(image_size)

            # Convert the image to a NumPy array
            img_array = np.array(img_resized).astype('float32')

            # Normalize the image
            img_normalized = normalize_image(img_array, method=normalize_type)

            # Save the processed image
            processed_filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, processed_filename)

            # Convert back to an image for saving
            Image.fromarray((img_normalized * 255).astype('uint8')).save(output_path)

            logging.info(f"Processed and saved: {output_path}")
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")

def process_images_sequential(input_dir=DATA_DIR, output_dir=PROCESSED_DIR):
    """Process images sequentially."""
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            preprocess_image(image_path, output_dir)

def process_images_parallel(input_dir=DATA_DIR, output_dir=PROCESSED_DIR):
    """Process images in parallel using multiple CPU cores."""
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if f.endswith('.jpg') or f.endswith('.png')]

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_image_worker, image_paths), total=len(image_paths)))

def process_image_worker(image_path):
    """Helper function for multiprocessing."""
    preprocess_image(image_path)

def check_image_size(image_path):
    """Check the size of a single image."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logging.error(f"Error checking size of {image_path}: {str(e)}")
        return None

def log_image_sizes(input_dir=DATA_DIR):
    """Log the sizes of all images in the input directory."""
    image_sizes = {}
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            size = check_image_size(image_path)
            if size:
                image_sizes[filename] = size

    for image, size in image_sizes.items():
        logging.info(f"{image}: {size}")

def remove_corrupted_images(input_dir=DATA_DIR):
    """Remove corrupted images that cannot be opened."""
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Verify if the image is corrupted
            except Exception as e:
                logging.warning(f"Removing corrupted image: {filename}")
                os.remove(image_path)

def resize_by_aspect_ratio(image_path, target_size=(224, 224), output_dir=PROCESSED_DIR):
    """Resize the image while maintaining its aspect ratio."""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(target_size, Image.ANTIALIAS)

            img_array = np.array(img).astype('float32')
            img_normalized = normalize_image(img_array, method=NORMALIZE_TYPE)

            processed_filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, processed_filename)

            Image.fromarray((img_normalized * 255).astype('uint8')).save(output_path)

            logging.info(f"Aspect-ratio resized and saved: {output_path}")
    except Exception as e:
        logging.error(f"Error resizing {image_path} with aspect ratio: {str(e)}")

def process_images_aspect_ratio(input_dir=DATA_DIR, output_dir=PROCESSED_DIR, target_size=(224, 224)):
    """Process images while maintaining aspect ratio."""
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            resize_by_aspect_ratio(image_path, target_size=target_size, output_dir=output_dir)

def convert_image_to_grayscale(image_path, output_dir=PROCESSED_DIR):
    """Convert an image to grayscale."""
    try:
        with Image.open(image_path) as img:
            grayscale_img = img.convert("L")  # Convert to grayscale

            img_array = np.array(grayscale_img).astype('float32')
            img_normalized = normalize_image(img_array, method=NORMALIZE_TYPE)

            processed_filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, processed_filename)

            Image.fromarray((img_normalized * 255).astype('uint8')).save(output_path)

            logging.info(f"Grayscale and saved: {output_path}")
    except Exception as e:
        logging.error(f"Error converting to grayscale {image_path}: {str(e)}")

def convert_images_to_grayscale(input_dir=DATA_DIR, output_dir=PROCESSED_DIR):
    """Convert all images in the directory to grayscale."""
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            convert_image_to_grayscale(image_path, output_dir)

if __name__ == '__main__':
    logging.info("Starting image preprocessing...")

    # Remove corrupted images
    remove_corrupted_images()

    # Log image sizes before processing
    log_image_sizes()

    # Process images using aspect ratio resizing
    process_images_aspect_ratio()

    # Process images in parallel
    # process_images_parallel()

    # Convert all images to grayscale
    convert_images_to_grayscale()

    logging.info("Preprocessing complete.")