#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <iostream>

class ImageAugmentor {
public:
    ImageAugmentor(float rotationRange, float scaleRange, float shiftRange, 
                   float brightnessRange, float contrastRange, float noiseRange)
        : rotationRange(rotationRange), scaleRange(scaleRange), shiftRange(shiftRange), 
          brightnessRange(brightnessRange), contrastRange(contrastRange), noiseRange(noiseRange) {}

    cv::Mat augment(const cv::Mat& image) {
        cv::Mat augmentedImage = image.clone();

        // Apply transformations sequentially
        augmentedImage = applyRotation(augmentedImage);
        augmentedImage = applyScaling(augmentedImage);
        augmentedImage = applyShifting(augmentedImage);
        augmentedImage = applyFlipping(augmentedImage);
        augmentedImage = applyBrightness(augmentedImage);
        augmentedImage = applyContrast(augmentedImage);
        augmentedImage = applyNoise(augmentedImage);

        return augmentedImage;
    }

private:
    float rotationRange;
    float scaleRange;
    float shiftRange;
    float brightnessRange;
    float contrastRange;
    float noiseRange;

    float getRandomValue(float min, float max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }

    // Rotation
    cv::Mat applyRotation(const cv::Mat& image) {
        float angle = getRandomValue(-rotationRange, rotationRange);
        float scale = getRandomValue(1.0 - scaleRange, 1.0 + scaleRange);
        cv::Point2f center(image.cols / 2, image.rows / 2);
        cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, scale);
        cv::Mat rotatedImage;
        cv::warpAffine(image, rotatedImage, rotationMatrix, image.size());
        return rotatedImage;
    }

    // Scaling
    cv::Mat applyScaling(const cv::Mat& image) {
        float scaleFactor = getRandomValue(1.0 - scaleRange, 1.0 + scaleRange);
        cv::Mat scaledImage;
        cv::resize(image, scaledImage, cv::Size(), scaleFactor, scaleFactor);
        return scaledImage;
    }

    // Shifting
    cv::Mat applyShifting(const cv::Mat& image) {
        float dx = getRandomValue(-shiftRange, shiftRange) * image.cols;
        float dy = getRandomValue(-shiftRange, shiftRange) * image.rows;
        cv::Mat translationMatrix = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
        cv::Mat shiftedImage;
        cv::warpAffine(image, shiftedImage, translationMatrix, image.size());
        return shiftedImage;
    }

    // Flipping
    cv::Mat applyFlipping(const cv::Mat& image) {
        cv::Mat flippedImage = image.clone();
        if (getRandomValue(0, 1) > 0.5) {
            cv::flip(flippedImage, flippedImage, 1); // Horizontal flip
        }
        return flippedImage;
    }

    // Brightness Adjustment
    cv::Mat applyBrightness(const cv::Mat& image) {
        float brightnessFactor = getRandomValue(1.0 - brightnessRange, 1.0 + brightnessRange);
        cv::Mat brightImage = image * brightnessFactor;
        brightImage.convertTo(brightImage, -1, brightnessFactor, 0);
        return brightImage;
    }

    // Contrast Adjustment
    cv::Mat applyContrast(const cv::Mat& image) {
        float contrastFactor = getRandomValue(1.0 - contrastRange, 1.0 + contrastRange);
        cv::Mat contrastImage = image.clone();
        contrastImage.convertTo(contrastImage, -1, contrastFactor, 0);
        return contrastImage;
    }

    // Noise Addition
    cv::Mat applyNoise(const cv::Mat& image) {
        cv::Mat noisyImage = image.clone();
        cv::Mat noise = cv::Mat(image.size(), image.type());
        cv::randn(noise, 0, noiseRange * 255);
        noisyImage += noise;
        return noisyImage;
    }
};

void displayImage(const std::string& windowName, const cv::Mat& image) {
    cv::imshow(windowName, image);
    cv::waitKey(0);
    cv::destroyWindow(windowName);
}

void saveImage(const std::string& filename, const cv::Mat& image) {
    cv::imwrite(filename, image);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./cpp_augment <image_path>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    // Create the augmentor with specific ranges for augmentations
    ImageAugmentor augmentor(30.0f, 0.2f, 0.1f, 0.2f, 0.2f, 0.05f);

    cv::Mat augmentedImage = augmentor.augment(image);

    // Display the original and augmented images
    displayImage("Original Image", image);
    displayImage("Augmented Image", augmentedImage);

    // Save the augmented image
    saveImage("augmented_image.jpg", augmentedImage);
    std::cout << "Augmented image saved as 'augmented_image.jpg'" << std::endl;

    return 0;
}