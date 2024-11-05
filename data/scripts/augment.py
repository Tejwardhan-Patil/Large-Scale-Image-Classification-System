import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import random

# Image augmentation functions
def rotate_image(image, angle):
    """Rotates an image by a given angle."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, matrix, (w, h))

def flip_image(image, flip_code):
    """Flips an image. flip_code: 1 for horizontal, 0 for vertical, -1 for both axes."""
    return cv2.flip(image, flip_code)

def scale_image(image, scale_factor):
    """Scales an image by a given factor."""
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

def adjust_brightness(image, factor):
    """Adjusts brightness of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, factor):
    """Adjusts contrast of the image."""
    return cv2.addWeighted(image, factor, np.zeros(image.shape, image.dtype), 0, 0)

def add_noise(image, noise_type='gaussian'):
    """Adds noise to the image. Supported types: 'gaussian', 'salt_pepper'."""
    if noise_type == 'gaussian':
        row, col, ch = image.shape
        mean = 0
        sigma = 0.01 * 255
        gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch)
        noisy_img = image + gauss
        return np.clip(noisy_img, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5
        amount = 0.004
        noisy_img = image.copy()
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_img[coords[0], coords[1], :] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_img[coords[0], coords[1], :] = 0
        return noisy_img

def change_saturation(image, factor):
    """Adjusts saturation of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def crop_image(image, crop_factor):
    """Crops the image by a certain factor."""
    h, w = image.shape[:2]
    crop_h, crop_w = int(h * crop_factor), int(w * crop_factor)
    return image[crop_h:h-crop_h, crop_w:w-crop_w]

def augment_image(image_path, save_dir):
    """Applies multiple augmentations to the image and saves the results."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Rotation
        for angle in [90, 180, 270]:
            rotated_img = rotate_image(image, angle)
            cv2.imwrite(f'{save_dir}/{base_name}_rotate_{angle}.jpg', rotated_img)

        # Flip
        for flip_code in [0, 1, -1]:
            flipped_img = flip_image(image, flip_code)
            cv2.imwrite(f'{save_dir}/{base_name}_flip_{flip_code}.jpg', flipped_img)

        # Scale
        for scale_factor in [0.8, 1.2]:
            scaled_img = scale_image(image, scale_factor)
            cv2.imwrite(f'{save_dir}/{base_name}_scale_{scale_factor}.jpg', scaled_img)

        # Adjust Brightness
        for brightness_factor in [0.7, 1.3]:
            bright_img = adjust_brightness(image, brightness_factor)
            cv2.imwrite(f'{save_dir}/{base_name}_bright_{brightness_factor}.jpg', bright_img)

        # Adjust Contrast
        for contrast_factor in [0.8, 1.5]:
            contrast_img = adjust_contrast(image, contrast_factor)
            cv2.imwrite(f'{save_dir}/{base_name}_contrast_{contrast_factor}.jpg', contrast_img)

        # Add Noise
        for noise_type in ['gaussian', 'salt_pepper']:
            noisy_img = add_noise(image, noise_type)
            cv2.imwrite(f'{save_dir}/{base_name}_noise_{noise_type}.jpg', noisy_img)

        # Saturation
        for saturation_factor in [0.7, 1.3]:
            saturated_img = change_saturation(image, saturation_factor)
            cv2.imwrite(f'{save_dir}/{base_name}_saturation_{saturation_factor}.jpg', saturated_img)

        # Crop
        for crop_factor in [0.1, 0.2]:
            cropped_img = crop_image(image, crop_factor)
            cv2.imwrite(f'{save_dir}/{base_name}_crop_{crop_factor}.jpg', cropped_img)

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")

def augment_batch(image_paths, save_dir):
    """Applies augmentations to a batch of images."""
    for image_path in image_paths:
        augment_image(image_path, save_dir)

def split_data(image_paths, train_ratio=0.8, val_ratio=0.1):
    """Splits data into training, validation, and test sets."""
    random.shuffle(image_paths)
    train_size = int(len(image_paths) * train_ratio)
    val_size = int(len(image_paths) * val_ratio)
    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size:train_size + val_size]
    test_paths = image_paths[train_size + val_size:]
    return train_paths, val_paths, test_paths

def save_image_sets(train_paths, val_paths, test_paths, base_dir):
    """Saves the image paths for train, val, and test sets."""
    with open(os.path.join(base_dir, 'train.txt'), 'w') as f:
        f.write("\n".join(train_paths))
    with open(os.path.join(base_dir, 'val.txt'), 'w') as f:
        f.write("\n".join(val_paths))
    with open(os.path.join(base_dir, 'test.txt'), 'w') as f:
        f.write("\n".join(test_paths))

def process_images(input_dir, output_dir, split=False):
    """Processes all images in the input directory and applies augmentations."""
    os.makedirs(output_dir, exist_ok=True)
    images = glob(f"{input_dir}/*.jpg")
    
    if split:
        train_paths, val_paths, test_paths = split_data(images)
        save_image_sets(train_paths, val_paths, test_paths, output_dir)
        augment_batch(train_paths, os.path.join(output_dir, 'train'))
        augment_batch(val_paths, os.path.join(output_dir, 'val'))
        augment_batch(test_paths, os.path.join(output_dir, 'test'))
    else:
        augment_batch(images, output_dir)

if __name__ == "__main__":
    input_directory = 'data/processed/'  # Directory with preprocessed images
    output_directory = 'data/augmented/'  # Directory to save augmented images
    split_data_flag = True  # Set to True to split the data into train/val/test
    process_images(input_directory, output_directory, split_data_flag)