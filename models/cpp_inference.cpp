#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/script.h>  // LibTorch header for loading PyTorch models
#include <chrono>
#include <stdexcept>  // For handling exceptions
#include <filesystem> // C++17 for checking file paths

// Check if a given file exists
bool fileExists(const std::string &path) {
    return std::filesystem::exists(path);
}

// Utility function to preprocess the input image (resizing, normalization)
cv::Mat preprocessImage(const cv::Mat &input_image, const int input_width, const int input_height) {
    if (input_image.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }

    cv::Mat processed_image;
    cv::resize(input_image, processed_image, cv::Size(input_width, input_height));

    // Convert to float and normalize the pixel values to [0, 1]
    processed_image.convertTo(processed_image, CV_32F, 1.0 / 255.0);

    // Convert BGR (OpenCV default) to RGB format
    cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB);

    return processed_image;
}

// Convert cv::Mat image to a tensor that can be used for inference in LibTorch
torch::Tensor convertToTensor(const cv::Mat &image) {
    if (image.empty()) {
        throw std::invalid_argument("Image passed for tensor conversion is empty.");
    }

    // Create a tensor from the image data
    torch::Tensor tensor_image = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kFloat);

    // Reorder dimensions to match [N, C, H, W] format
    tensor_image = tensor_image.permute({0, 3, 1, 2});

    // Ensure tensor memory is contiguous for efficient processing
    tensor_image = tensor_image.contiguous();

    // Clone the tensor to ensure safe memory handling
    return tensor_image.clone();
}

// Function to perform inference on a preprocessed image
std::vector<float> runInference(const torch::jit::script::Module &model, const torch::Tensor &input_tensor) {
    // Prepare input for model forward pass
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // Perform the forward pass to get the output tensor
    torch::Tensor output;
    try {
        output = model.forward(inputs).toTensor();
    } catch (const c10::Error &e) {
        std::cerr << "Error during model inference: " << e.what() << std::endl;
        throw;
    }

    // Convert output tensor to a standard vector of floats
    std::vector<float> result(output.sizes()[1]); 
    std::memcpy(result.data(), output.data_ptr<float>(), result.size() * sizeof(float));

    return result;
}

// Function to load the PyTorch model from a given file path
torch::jit::script::Module loadModel(const std::string &model_path) {
    if (!fileExists(model_path)) {
        throw std::invalid_argument("Model file does not exist at path: " + model_path);
    }

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        std::cout << "Model loaded successfully from " << model_path << std::endl;
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model from " << model_path << std::endl;
        throw;
    }

    return model;
}

// Function to load the input image from a file path
cv::Mat loadImage(const std::string &image_path) {
    if (!fileExists(image_path)) {
        throw std::invalid_argument("Image file does not exist at path: " + image_path);
    }

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image from: " + image_path);
    }

    return image;
}

int main(int argc, char** argv) {
    // Command-line argument parsing
    if (argc != 3) {
        std::cerr << "Usage: ./cpp_inference <model_path> <image_path>" << std::endl;
        return -1;
    }

    // Extract model and image paths from command-line arguments
    std::string model_path = argv[1];
    std::string image_path = argv[2];

    // Load the pre-trained PyTorch model
    torch::jit::script::Module model;
    try {
        model = loadModel(model_path);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return -1;
    }

    // Load the input image
    cv::Mat image;
    try {
        image = loadImage(image_path);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load image: " << e.what() << std::endl;
        return -1;
    }

    // Define the input size for the model (assumes 224x224 input size)
    const int input_width = 224;
    const int input_height = 224;

    // Preprocess the image
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    cv::Mat preprocessed_image;
    try {
        preprocessed_image = preprocessImage(image, input_width, input_height);
    } catch (const std::exception &e) {
        std::cerr << "Image preprocessing failed: " << e.what() << std::endl;
        return -1;
    }

    torch::Tensor input_tensor;
    try {
        input_tensor = convertToTensor(preprocessed_image);
    } catch (const std::exception &e) {
        std::cerr << "Failed to convert image to tensor: " << e.what() << std::endl;
        return -1;
    }

    // Measure the preprocessing time
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> preprocess_duration = end_preprocess - start_preprocess;
    std::cout << "Image preprocessing time: " << preprocess_duration.count() << " seconds" << std::endl;

    // Run inference on the input tensor
    std::vector<float> inference_results;
    auto start_inference = std::chrono::high_resolution_clock::now();
    try {
        inference_results = runInference(model, input_tensor);
    } catch (const std::exception &e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return -1;
    }

    // Measure the inference time
    auto end_inference = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inference_duration = end_inference - start_inference;
    std::cout << "Inference time: " << inference_duration.count() << " seconds" << std::endl;

    // Output the inference results
    std::cout << "Inference results: ";
    for (const float &result : inference_results) {
        std::cout << result << " ";
    }
    std::cout << std::endl;

    return 0;
}

// Function to normalize the inference results (e.g., applying softmax)
std::vector<float> softmax(const std::vector<float> &logits) {
    std::vector<float> probabilities(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());

    float sum_exp = 0.0;
    for (const float &logit : logits) {
        sum_exp += std::exp(logit - max_logit);
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit) / sum_exp;
    }

    return probabilities;
}

// Function to print the top N predictions with probabilities
void printTopNPredictions(const std::vector<float> &probabilities, int topN = 5) {
    if (topN > probabilities.size()) {
        topN = probabilities.size();
    }

    // Create a vector of index-probability pairs
    std::vector<std::pair<int, float>> index_probabilities;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        index_probabilities.emplace_back(i, probabilities[i]);
    }

    // Sort the vector in descending order by probability
    std::sort(index_probabilities.begin(), index_probabilities.end(),
              [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
                  return a.second > b.second;
              });

    // Print the top N predictions
    std::cout << "Top " << topN << " predictions:" << std::endl;
    for (int i = 0; i < topN; ++i) {
        std::cout << "Class: " << index_probabilities[i].first
                  << " | Probability: " << index_probabilities[i].second * 100 << "%" << std::endl;
    }
}

// Function to load class labels from a file
std::vector<std::string> loadClassLabels(const std::string &label_file_path) {
    if (!fileExists(label_file_path)) {
        throw std::invalid_argument("Label file does not exist at path: " + label_file_path);
    }

    std::ifstream label_file(label_file_path);
    if (!label_file.is_open()) {
        throw std::runtime_error("Failed to open label file: " + label_file_path);
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(label_file, line)) {
        labels.push_back(line);
    }

    label_file.close();
    return labels;
}

// Function to print the top N predictions with class labels and probabilities
void printTopNPredictionsWithLabels(const std::vector<float> &probabilities, 
                                    const std::vector<std::string> &labels, int topN = 5) {
    if (topN > probabilities.size()) {
        topN = probabilities.size();
    }

    // Create a vector of index-probability pairs
    std::vector<std::pair<int, float>> index_probabilities;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        index_probabilities.emplace_back(i, probabilities[i]);
    }

    // Sort the vector in descending order by probability
    std::sort(index_probabilities.begin(), index_probabilities.end(),
              [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
                  return a.second > b.second;
              });

    // Print the top N predictions with class labels
    std::cout << "Top " << topN << " predictions with labels:" << std::endl;
    for (int i = 0; i < topN; ++i) {
        int class_index = index_probabilities[i].first;
        float probability = index_probabilities[i].second * 100;
        std::string label = (class_index < labels.size()) ? labels[class_index] : "Unknown";

        std::cout << "Class: " << label << " | Probability: " << probability << "%" << std::endl;
    }
}

int main(int argc, char** argv) {
    // Ensure enough arguments are provided
    if (argc != 4) {
        std::cerr << "Usage: ./cpp_inference <model_path> <image_path> <label_path>" << std::endl;
        return -1;
    }

    // Extract model, image, and label file paths from command-line arguments
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string label_path = argv[3];

    // Load the pre-trained PyTorch model
    torch::jit::script::Module model;
    try {
        model = loadModel(model_path);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return -1;
    }

    // Load the input image
    cv::Mat image;
    try {
        image = loadImage(image_path);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load image: " << e.what() << std::endl;
        return -1;
    }

    // Load the class labels
    std::vector<std::string> class_labels;
    try {
        class_labels = loadClassLabels(label_path);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load class labels: " << e.what() << std::endl;
        return -1;
    }

    // Define the input size for the model (assumes 224x224 input size)
    const int input_width = 224;
    const int input_height = 224;

    // Preprocess the image
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    cv::Mat preprocessed_image;
    try {
        preprocessed_image = preprocessImage(image, input_width, input_height);
    } catch (const std::exception &e) {
        std::cerr << "Image preprocessing failed: " << e.what() << std::endl;
        return -1;
    }

    torch::Tensor input_tensor;
    try {
        input_tensor = convertToTensor(preprocessed_image);
    } catch (const std::exception &e) {
        std::cerr << "Failed to convert image to tensor: " << e.what() << std::endl;
        return -1;
    }

    // Measure the preprocessing time
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> preprocess_duration = end_preprocess - start_preprocess;
    std::cout << "Image preprocessing time: " << preprocess_duration.count() << " seconds" << std::endl;

    // Run inference on the input tensor
    std::vector<float> inference_results;
    auto start_inference = std::chrono::high_resolution_clock::now();
    try {
        inference_results = runInference(model, input_tensor);
    } catch (const std::exception &e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return -1;
    }

    // Measure the inference time
    auto end_inference = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inference_duration = end_inference - start_inference;
    std::cout << "Inference time: " << inference_duration.count() << " seconds" << std::endl;

    // Apply softmax to get probabilities
    std::vector<float> probabilities = softmax(inference_results);

    // Print the top 5 predictions with labels and probabilities
    printTopNPredictionsWithLabels(probabilities, class_labels, 5);

    return 0;
}

// Function to calculate the time difference in milliseconds
double calculateDuration(const std::chrono::time_point<std::chrono::high_resolution_clock> &start,
                         const std::chrono::time_point<std::chrono::high_resolution_clock> &end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Enhanced logging function to print detailed inference information
void logInferenceDetails(const std::string &model_path, const std::string &image_path,
                         double preprocess_time, double inference_time, double total_time) {
    std::cout << "\n====== Inference Summary ======" << std::endl;
    std::cout << "Model Path: " << model_path << std::endl;
    std::cout << "Image Path: " << image_path << std::endl;
    std::cout << "Preprocessing Time: " << preprocess_time << " ms" << std::endl;
    std::cout << "Inference Time: " << inference_time << " ms" << std::endl;
    std::cout << "Total Execution Time: " << total_time << " ms" << std::endl;
    std::cout << "==============================" << std::endl;
}

// Function to save the inference results to a file
void saveResultsToFile(const std::string &output_file, const std::vector<std::string> &labels,
                       const std::vector<float> &probabilities, int topN) {
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }

    // Write the top N predictions with labels and probabilities to the file
    for (int i = 0; i < topN && i < probabilities.size(); ++i) {
        out_file << "Class: " << labels[i] << ", Probability: " << probabilities[i] * 100 << "%" << std::endl;
    }

    out_file.close();
    std::cout << "Inference results saved to " << output_file << std::endl;
}

// Extended error handling to ensure smooth execution
void handleException(const std::exception &e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
    std::cerr << "Please check the file paths, input formats, and system configuration." << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: ./cpp_inference <model_path> <image_path> <label_path> [output_file]" << std::endl;
        return -1;
    }

    // Extract paths from command-line arguments
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string label_path = argv[3];
    std::string output_file = (argc == 5) ? argv[4] : "";

    // Timing start for overall execution
    auto start_total = std::chrono::high_resolution_clock::now();

    // Load the model
    torch::jit::script::Module model;
    try {
        model = loadModel(model_path);
    } catch (const std::exception &e) {
        handleException(e);
        return -1;
    }

    // Load the input image
    cv::Mat image;
    try {
        image = loadImage(image_path);
    } catch (const std::exception &e) {
        handleException(e);
        return -1;
    }

    // Load class labels
    std::vector<std::string> class_labels;
    try {
        class_labels = loadClassLabels(label_path);
    } catch (const std::exception &e) {
        handleException(e);
        return -1;
    }

    // Define input size for the model
    const int input_width = 224;
    const int input_height = 224;

    // Preprocess the image
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    cv::Mat preprocessed_image;
    try {
        preprocessed_image = preprocessImage(image, input_width, input_height);
    } catch (const std::exception &e) {
        handleException(e);
        return -1;
    }

    torch::Tensor input_tensor;
    try {
        input_tensor = convertToTensor(preprocessed_image);
    } catch (const std::exception &e) {
        handleException(e);
        return -1;
    }

    // Measure preprocessing time
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    double preprocess_time = calculateDuration(start_preprocess, end_preprocess);

    // Run inference
    std::vector<float> inference_results;
    auto start_inference = std::chrono::high_resolution_clock::now();
    try {
        inference_results = runInference(model, input_tensor);
    } catch (const std::exception &e) {
        handleException(e);
        return -1;
    }

    // Measure inference time
    auto end_inference = std::chrono::high_resolution_clock::now();
    double inference_time = calculateDuration(start_inference, end_inference);

    // Apply softmax to get probabilities
    std::vector<float> probabilities = softmax(inference_results);

    // Print the top 5 predictions with labels
    const int topN = 5;
    printTopNPredictionsWithLabels(probabilities, class_labels, topN);

    // Save the results to a file if specified
    if (!output_file.empty()) {
        try {
            saveResultsToFile(output_file, class_labels, probabilities, topN);
        } catch (const std::exception &e) {
            handleException(e);
            return -1;
        }
    }

    // Measure total execution time
    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time = calculateDuration(start_total, end_total);

    // Log the detailed inference information
    logInferenceDetails(model_path, image_path, preprocess_time, inference_time, total_time);

    return 0;
}