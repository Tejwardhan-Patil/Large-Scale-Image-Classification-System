#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

// Function to normalize a vector of image data (pixel values)
std::vector<float> normalize_data(const std::vector<float>& data, float mean, float std) {
    std::vector<float> normalized(data.size());
    std::transform(data.begin(), data.end(), normalized.begin(), [mean, std](float pixel) {
        return (pixel - mean) / std;
    });
    return normalized;
}

// Function to compute mean of a dataset (image pixel values)
float compute_mean(const std::vector<float>& data) {
    return std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
}

// Function to compute standard deviation of a dataset
float compute_std(const std::vector<float>& data, float mean) {
    float variance = 0.0f;
    for (const float& value : data) {
        variance += std::pow(value - mean, 2);
    }
    return std::sqrt(variance / data.size());
}

// Function to resize image data (simplified nearest neighbor)
std::vector<std::vector<float>> resize_image(const std::vector<std::vector<float>>& image, int new_width, int new_height) {
    std::vector<std::vector<float>> resized_image(new_height, std::vector<float>(new_width, 0));
    int old_width = image[0].size();
    int old_height = image.size();

    float x_ratio = static_cast<float>(old_width) / new_width;
    float y_ratio = static_cast<float>(old_height) / new_height;

    for (int i = 0; i < new_height; ++i) {
        for (int j = 0; j < new_width; ++j) {
            int px = static_cast<int>(j * x_ratio);
            int py = static_cast<int>(i * y_ratio);
            resized_image[i][j] = image[py][px];
        }
    }

    return resized_image;
}

// Function to flatten a 2D image into a 1D vector
std::vector<float> flatten_image(const std::vector<std::vector<float>>& image) {
    std::vector<float> flattened;
    for (const auto& row : image) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

// Function to rotate an image by 90 degrees clockwise
std::vector<std::vector<float>> rotate_image_90(const std::vector<std::vector<float>>& image) {
    int width = image[0].size();
    int height = image.size();
    std::vector<std::vector<float>> rotated_image(width, std::vector<float>(height, 0));

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            rotated_image[j][height - i - 1] = image[i][j];
        }
    }
    return rotated_image;
}

// Function to flip an image horizontally
std::vector<std::vector<float>> flip_image_horizontal(const std::vector<std::vector<float>>& image) {
    std::vector<std::vector<float>> flipped_image = image;
    for (auto& row : flipped_image) {
        std::reverse(row.begin(), row.end());
    }
    return flipped_image;
}

// Function to flip an image vertically
std::vector<std::vector<float>> flip_image_vertical(const std::vector<std::vector<float>>& image) {
    std::vector<std::vector<float>> flipped_image = image;
    std::reverse(flipped_image.begin(), flipped_image.end());
    return flipped_image;
}

// Function to transpose an image (matrix)
std::vector<std::vector<float>> transpose_image(const std::vector<std::vector<float>>& image) {
    int width = image[0].size();
    int height = image.size();
    std::vector<std::vector<float>> transposed_image(width, std::vector<float>(height, 0));

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            transposed_image[j][i] = image[i][j];
        }
    }
    return transposed_image;
}

// Function to apply padding to an image
std::vector<std::vector<float>> pad_image(const std::vector<std::vector<float>>& image, int pad_size, float pad_value = 0) {
    int width = image[0].size();
    int height = image.size();
    std::vector<std::vector<float>> padded_image(height + 2 * pad_size, std::vector<float>(width + 2 * pad_size, pad_value));

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            padded_image[i + pad_size][j + pad_size] = image[i][j];
        }
    }

    return padded_image;
}

// Function to batch process a set of images and normalize them
std::vector<std::vector<float>> batch_normalize(const std::vector<std::vector<float>>& batch_data, float mean, float std) {
    std::vector<std::vector<float>> normalized_batch(batch_data.size());
    for (size_t i = 0; i < batch_data.size(); ++i) {
        normalized_batch[i] = normalize_data(batch_data[i], mean, std);
    }
    return normalized_batch;
}

// Function to combine multiple images (concatenation)
std::vector<float> concatenate_images(const std::vector<std::vector<float>>& images) {
    std::vector<float> concatenated;
    for (const auto& image : images) {
        concatenated.insert(concatenated.end(), image.begin(), image.end());
    }
    return concatenated;
}

// Function to calculate Euclidean distance between two images (represented as flattened vectors)
float euclidean_distance(const std::vector<float>& image1, const std::vector<float>& image2) {
    float distance = 0.0f;
    for (size_t i = 0; i < image1.size(); ++i) {
        distance += std::pow(image1[i] - image2[i], 2);
    }
    return std::sqrt(distance);
}

// Function to crop an image
std::vector<std::vector<float>> crop_image(const std::vector<std::vector<float>>& image, int x, int y, int width, int height) {
    std::vector<std::vector<float>> cropped_image(height, std::vector<float>(width, 0));
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cropped_image[i][j] = image[y + i][x + j];
        }
    }
    return cropped_image;
}

// Function to perform element-wise addition of two images
std::vector<std::vector<float>> add_images(const std::vector<std::vector<float>>& image1, const std::vector<std::vector<float>>& image2) {
    int height = image1.size();
    int width = image1[0].size();
    std::vector<std::vector<float>> result_image(height, std::vector<float>(width, 0));

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            result_image[i][j] = image1[i][j] + image2[i][j];
        }
    }
    return result_image;
}

// Function to subtract one image from another
std::vector<std::vector<float>> subtract_images(const std::vector<std::vector<float>>& image1, const std::vector<std::vector<float>>& image2) {
    int height = image1.size();
    int width = image1[0].size();
    std::vector<std::vector<float>> result_image(height, std::vector<float>(width, 0));

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            result_image[i][j] = image1[i][j] - image2[i][j];
        }
    }
    return result_image;
}

// Function to scale pixel values of an image
std::vector<std::vector<float>> scale_image(const std::vector<std::vector<float>>& image, float scale_factor) {
    int height = image.size();
    int width = image[0].size();
    std::vector<std::vector<float>> scaled_image(height, std::vector<float>(width, 0));

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            scaled_image[i][j] = image[i][j] * scale_factor;
        }
    }
    return scaled_image;
}

// Function to print a 2D image (for debugging purposes)
void print_image(const std::vector<std::vector<float>>& image) {
    for (const auto& row : image) {
        for (const auto& pixel : row) {
            std::cout << pixel << " ";
        }
        std::cout << std::endl;
    }
}

// Function to print vector (for debugging purposes)
void print_vector(const std::vector<float>& vec) {
    for (const auto& value : vec) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}