import unittest
import os
import numpy as np
from data.scripts.preprocess import preprocess_images
from data.scripts.augment import augment_images
from data.scripts.split import split_data
from utils.data_loader import load_data

class TestDataPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = 'data/raw/'
        cls.processed_data_dir = 'data/processed/'
        cls.annotations_dir = 'data/annotations/'
        cls.sample_image = np.random.rand(256, 256, 3)  # Simulated random image for testing
        cls.sample_labels = [0, 1, 0]  # Simulated label for testing

    def test_preprocess_images(self):
        """Test that image preprocessing (resizing, normalization) works correctly."""
        processed_images = preprocess_images(self.sample_image)
        self.assertEqual(processed_images.shape, (224, 224, 3)) 
        self.assertTrue(np.max(processed_images) <= 1.0)
        self.assertTrue(np.min(processed_images) >= 0.0)
        
    def test_preprocess_invalid_input(self):
        """Test preprocessing with invalid inputs, ensuring robust error handling."""
        with self.assertRaises(ValueError):
            preprocess_images(None)  # Test if None input raises an error
        with self.assertRaises(ValueError):
            preprocess_images(np.random.rand(10, 10, 3))  # Too small input image
        
    def test_preprocess_large_image(self):
        """Test preprocessing with a very large image to check memory handling."""
        large_image = np.random.rand(4096, 4096, 3)  # Simulate a 4096x4096 image
        processed_image = preprocess_images(large_image)
        self.assertEqual(processed_image.shape, (224, 224, 3))  # Check resizing works for large images
    
    def test_preprocess_grayscale_image(self):
        """Test preprocessing with a grayscale image."""
        grayscale_image = np.random.rand(256, 256)  # Grayscale image without a 3rd channel
        processed_image = preprocess_images(grayscale_image)
        self.assertEqual(processed_image.shape, (224, 224, 3))  # Should still be 3-channel after preprocessing
    
    def test_augment_images(self):
        """Test that image augmentation (rotation, flipping) works as expected."""
        augmented_images = augment_images(self.sample_image)
        self.assertEqual(augmented_images.shape, self.sample_image.shape)  # Augmented image should have same dimensions
    
    def test_augment_with_randomness(self):
        """Test augmentation multiple times to ensure randomness (different results)."""
        augmented_image_1 = augment_images(self.sample_image)
        augmented_image_2 = augment_images(self.sample_image)
        self.assertFalse(np.array_equal(augmented_image_1, augmented_image_2))  # Augmentations should be different
    
    def test_augment_invalid_input(self):
        """Test augmenting invalid inputs to ensure robust error handling."""
        with self.assertRaises(ValueError):
            augment_images(None)  # Invalid input test for augmentation
        with self.assertRaises(ValueError):
            augment_images(np.random.rand(10, 10, 3))  # Too small image
    
    def test_split_data(self):
        """Test the data splitting function (training, validation, test sets)."""
        train_set, val_set, test_set = split_data(self.raw_data_dir)
        total_images = len(os.listdir(self.raw_data_dir))
        self.assertEqual(total_images, len(train_set) + len(val_set) + len(test_set))
        self.assertTrue(len(train_set) > 0 and len(val_set) > 0 and len(test_set) > 0)
    
    def test_split_data_empty_dir(self):
        """Test data splitting with an empty directory."""
        empty_dir = 'data/empty_raw/'
        if not os.path.exists(empty_dir):
            os.makedirs(empty_dir)
        train_set, val_set, test_set = split_data(empty_dir)
        self.assertEqual(len(train_set), 0)
        self.assertEqual(len(val_set), 0)
        self.assertEqual(len(test_set), 0)
    
    def test_split_data_invalid_ratio(self):
        """Test data splitting with invalid train/validation/test ratios."""
        with self.assertRaises(ValueError):
            split_data(self.raw_data_dir, train_ratio=1.5, val_ratio=-0.5)  # Invalid ratios should raise errors
    
    def test_data_loader(self):
        """Test data loading utility function."""
        images, labels = load_data(self.processed_data_dir, self.annotations_dir)
        self.assertEqual(len(images), len(labels))
        self.assertTrue(all(isinstance(image, np.ndarray) for image in images))
        self.assertTrue(all(isinstance(label, int) for label in labels))
    
    def test_data_loader_empty_dir(self):
        """Test data loader with empty directories."""
        empty_processed_dir = 'data/empty_processed/'
        empty_annotations_dir = 'data/empty_annotations/'
        if not os.path.exists(empty_processed_dir):
            os.makedirs(empty_processed_dir)
        if not os.path.exists(empty_annotations_dir):
            os.makedirs(empty_annotations_dir)
        images, labels = load_data(empty_processed_dir, empty_annotations_dir)
        self.assertEqual(len(images), 0)
        self.assertEqual(len(labels), 0)
    
    def test_data_loader_corrupted_files(self):
        """Test data loader with corrupted image/annotation files."""
        corrupted_image_dir = 'data/corrupted_processed/'
        corrupted_annotation_dir = 'data/corrupted_annotations/'
        if not os.path.exists(corrupted_image_dir):
            os.makedirs(corrupted_image_dir)
        if not os.path.exists(corrupted_annotation_dir):
            os.makedirs(corrupted_annotation_dir)
        # Creating mock corrupted files
        with open(os.path.join(corrupted_image_dir, 'corrupted_image.jpg'), 'wb') as f:
            f.write(b'corrupted data')
        with open(os.path.join(corrupted_annotation_dir, 'corrupted_annotation.txt'), 'w') as f:
            f.write('corrupted data')
        
        with self.assertRaises(ValueError):
            load_data(corrupted_image_dir, corrupted_annotation_dir)
    
    def test_data_loader_large_files(self):
        """Test data loader with large files to check memory handling."""
        large_image_dir = 'data/large_processed/'
        large_annotation_dir = 'data/large_annotations/'
        if not os.path.exists(large_image_dir):
            os.makedirs(large_image_dir)
        if not os.path.exists(large_annotation_dir):
            os.makedirs(large_annotation_dir)
        
        # Simulate large files
        large_image = np.random.rand(4096, 4096, 3)
        np.save(os.path.join(large_image_dir, 'large_image.npy'), large_image)
        
        with open(os.path.join(large_annotation_dir, 'large_annotation.txt'), 'w') as f:
            f.write('1')
        
        images, labels = load_data(large_image_dir, large_annotation_dir)
        self.assertEqual(len(images), 1)
        self.assertEqual(len(labels), 1)

if __name__ == '__main__':
    unittest.main()