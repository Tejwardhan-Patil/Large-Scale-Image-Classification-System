#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <chrono>

// Configuration
#define TARGET_WIDTH 224
#define TARGET_HEIGHT 224
#define NORMALIZE_MIN 0.0
#define NORMALIZE_MAX 1.0
#define LOG_FILE "preprocess_log.txt"

std::mutex mtx; // For thread safety during logging and image saving

// Function to log messages
void log_message(const std::string &message) {
    std::lock_guard<std::mutex> lock(mtx);
    std::ofstream log(LOG_FILE, std::ios_base::app);
    if (log.is_open()) {
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        log << std::ctime(&now) << ": " << message << std::endl;
        log.close();
    } else {
        std::cerr << "Error: Unable to open log file." << std::endl;
    }
}

// Function to resize and normalize an image
cv::Mat preprocess_image(const cv::Mat &image) {
    cv::Mat resized_image, normalized_image;
    
    // Resize the image
    cv::resize(image, resized_image, cv::Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, cv::INTER_LINEAR);
    
    // Normalize the image to the range [NORMALIZE_MIN, NORMALIZE_MAX]
    resized_image.convertTo(normalized_image, CV_32F, (NORMALIZE_MAX - NORMALIZE_MIN) / 255.0, NORMALIZE_MIN);

    return normalized_image;
}

// Function to process a single image
void process_single_image(const std::string &input_path, const std::string &output_path) {
    cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        log_message("Error: Could not open image file: " + input_path);
        return;
    }

    cv::Mat processed_image = preprocess_image(image);

    // Save the processed image
    std::lock_guard<std::mutex> lock(mtx); // Ensures thread safety when saving the file
    processed_image.convertTo(processed_image, CV_8U, 255.0); // Convert back to [0, 255] range for saving
    cv::imwrite(output_path, processed_image);

    log_message("Successfully processed and saved image: " + output_path);
}

// Function to list all images in a directory (supports multiple formats)
std::vector<std::string> get_image_files(const std::string &directory) {
    std::vector<std::string> image_files;
    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
        std::string path = entry.path().string();
        if (path.find(".jpg") != std::string::npos || 
            path.find(".jpeg") != std::string::npos || 
            path.find(".png") != std::string::npos ||
            path.find(".bmp") != std::string::npos) {
            image_files.push_back(path);
        }
    }
    return image_files;
}

// Function to create output directories if they don't exist
void create_output_directory(const std::string &directory) {
    if (!std::filesystem::exists(directory)) {
        std::filesystem::create_directories(directory);
    }
}

// Multi-threaded batch preprocessing of images
void preprocess_batch(const std::vector<std::string> &input_paths, const std::vector<std::string> &output_paths) {
    std::vector<std::thread> threads;
    
    for (size_t i = 0; i < input_paths.size(); ++i) {
        threads.emplace_back(process_single_image, input_paths[i], output_paths[i]);
    }
    
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

// Main entry point
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> <output_directory>" << std::endl;
        log_message("Error: Insufficient arguments provided.");
        return -1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    log_message("Starting image preprocessing...");
    log_message("Input directory: " + input_dir);
    log_message("Output directory: " + output_dir);

    // Create the output directory if it doesn't exist
    create_output_directory(output_dir);

    // Get list of images to process
    std::vector<std::string> input_files = get_image_files(input_dir);
    if (input_files.empty()) {
        log_message("Error: No valid images found in the input directory.");
        return -1;
    }

    std::vector<std::string> output_files;
    for (const auto &file : input_files) {
        std::string output_path = output_dir + "/" + std::filesystem::path(file).filename().string();
        output_files.push_back(output_path);
    }

    // Preprocess images in parallel
    preprocess_batch(input_files, output_files);

    log_message("Image preprocessing completed successfully.");
    return 0;
}