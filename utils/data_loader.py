import os
import json
import logging
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging for the data loader
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, multi_label=False):
        self.image_dir = image_dir
        self.annotations = self._load_annotations(annotation_file)
        self.transform = transform
        self.multi_label = multi_label

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            img_path, label = self.annotations[idx]['image'], self.annotations[idx]['label']
            img = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')

            if self.transform:
                img = self.transform(img)

            if self.multi_label:
                label = torch.tensor(label, dtype=torch.float32)

            return img, label
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return None, None

    def _load_annotations(self, annotation_file):
        try:
            with open(annotation_file, 'r') as file:
                annotations = json.load(file)
            logging.info(f"Loaded {len(annotations)} annotations from {annotation_file}")
            return annotations
        except Exception as e:
            logging.error(f"Failed to load annotations: {e}")
            return []

def get_transform(img_size=224, augment=False):
    if augment:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def get_dataloader(image_dir, annotation_file, batch_size=32, shuffle=True, num_workers=4, img_size=224, augment=False, multi_label=False):
    transform = get_transform(img_size=img_size, augment=augment)

    dataset = ImageDataset(image_dir=image_dir, annotation_file=annotation_file, transform=transform, multi_label=multi_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

def split_dataset(image_dir, annotation_file, split_ratio=0.8, seed=42):
    random.seed(seed)
    annotations = ImageDataset(image_dir, annotation_file)._load_annotations(annotation_file)
    
    split_idx = int(len(annotations) * split_ratio)
    random.shuffle(annotations)
    
    train_annotations = annotations[:split_idx]
    val_annotations = annotations[split_idx:]

    train_file = os.path.join(os.path.dirname(annotation_file), 'train_annotations.json')
    val_file = os.path.join(os.path.dirname(annotation_file), 'val_annotations.json')

    with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f:
        json.dump(train_annotations, train_f)
        json.dump(val_annotations, val_f)

    logging.info(f"Dataset split into {len(train_annotations)} training and {len(val_annotations)} validation samples.")

    return train_file, val_file

def verify_data_integrity(image_dir, annotation_file):
    dataset = ImageDataset(image_dir, annotation_file)
    missing_files = []
    for idx in range(len(dataset)):
        img_path = dataset.annotations[idx]['image']
        full_img_path = os.path.join(image_dir, img_path)
        if not os.path.exists(full_img_path):
            logging.warning(f"Missing file: {full_img_path}")
            missing_files.append(full_img_path)
    if missing_files:
        logging.error(f"Total missing files: {len(missing_files)}")
    else:
        logging.info("All files are present and accounted for.")
    return missing_files

def balance_dataset(image_dir, annotation_file, threshold=100):
    dataset = ImageDataset(image_dir, annotation_file)._load_annotations(annotation_file)
    label_counts = {}
    for ann in dataset:
        label = ann['label']
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    underrepresented_labels = [label for label, count in label_counts.items() if count < threshold]
    augmented_data = []
    
    for ann in dataset:
        label = ann['label']
        if label in underrepresented_labels:
            augmented_data.extend([ann] * (threshold // label_counts[label]))
    
    balanced_annotations = dataset + augmented_data
    balanced_file = os.path.join(os.path.dirname(annotation_file), 'balanced_annotations.json')
    
    with open(balanced_file, 'w') as f:
        json.dump(balanced_annotations, f)
    
    logging.info(f"Balanced dataset with {len(balanced_annotations)} samples.")
    
    return balanced_file

def load_multi_label_data(image_dir, annotation_file, class_names, batch_size=32, shuffle=True, num_workers=4, img_size=224, augment=False):
    dataset = ImageDataset(image_dir, annotation_file, transform=get_transform(img_size=img_size, augment=augment), multi_label=True)
    
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    for annotation in dataset.annotations:
        label = annotation['label']
        annotation['label'] = [0] * len(class_names)
        for lbl in label:
            annotation['label'][class_to_idx[lbl]] = 1

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def convert_to_grayscale(image_dir, output_dir, annotation_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = ImageDataset(image_dir, annotation_file)

    for idx in range(len(dataset)):
        img_path = dataset.annotations[idx]['image']
        img = Image.open(os.path.join(image_dir, img_path)).convert('L')  # Convert to grayscale
        img.save(os.path.join(output_dir, img_path))

    logging.info(f"Converted all images to grayscale in {output_dir}.")

def summarize_dataset(image_dir, annotation_file):
    dataset = ImageDataset(image_dir, annotation_file)
    
    total_images = len(dataset)
    label_distribution = {}
    
    for ann in dataset.annotations:
        label = ann['label']
        if label not in label_distribution:
            label_distribution[label] = 0
        label_distribution[label] += 1
    
    logging.info(f"Total images: {total_images}")
    logging.info("Label distribution:")
    for label, count in label_distribution.items():
        logging.info(f"{label}: {count} images")

def resize_dataset(image_dir, output_dir, annotation_file, img_size=128):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = ImageDataset(image_dir, annotation_file)

    for idx in range(len(dataset)):
        img_path = dataset.annotations[idx]['image']
        img = Image.open(os.path.join(image_dir, img_path)).convert('RGB')
        img = img.resize((img_size, img_size))
        img.save(os.path.join(output_dir, img_path))
    
    logging.info(f"Resized all images to {img_size}x{img_size} and saved to {output_dir}.")