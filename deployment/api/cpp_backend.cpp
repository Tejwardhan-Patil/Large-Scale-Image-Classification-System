#include <iostream>
#include <vector>
#include <string>
#include <json/json.h> 
#include <opencv2/opencv.hpp> 
#include <httplib.h> 
#include <fstream> 
#include <mutex>

// Mutex for thread-safe logging
std::mutex log_mutex;

// Function to log messages to a file
void log_message(const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex);
    std::ofstream log_file("server_log.txt", std::ios_base::app);
    log_file << message << std::endl;
}

void load_model(const std::string& model_path) {
    // Check if model_path is empty
    if (model_path.empty()) {
        log_message("Error: Model path is empty.");
        throw std::invalid_argument("Model path cannot be empty.");
    }

    // Simulate model loading
    std::cout << "Loading model from " << model_path << std::endl;
    log_message("Model loading initiated from " + model_path);

    // Simulate a delay to represent model loading
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Log successful load
    log_message("Model successfully loaded from " + model_path);
}

// Preprocess input image before inference (resize, normalize, etc.)
cv::Mat preprocess_image(const cv::Mat& image) {
    cv::Mat processed;
    cv::resize(image, processed, cv::Size(224, 224)); // Resize to 224x224
    processed.convertTo(processed, CV_32F, 1.0 / 255.0); // Normalize pixel values to [0, 1]
    log_message("Image preprocessed");
    return processed;
}

std::vector<float> run_inference(const cv::Mat& image, const std::string& model_path, const std::string& config_path, const std::string& framework) {
    // Check if the input image is empty
    if (image.empty()) {
        log_message("Error: Input image is empty. Cannot run inference.");
        throw std::invalid_argument("Input image cannot be empty.");
    }

    // Load the pre-trained model
    cv::dnn::Net net = cv::dnn::readNet(model_path, config_path, framework);
    if (net.empty()) {
        log_message("Error: Failed to load the model.");
        throw std::runtime_error("Failed to load the model.");
    }
    log_message("Model loaded successfully.");

    // Preprocess the image to match the model's input requirements
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);

    // Set the input to the network
    net.setInput(blob);

    // Run the forward pass to obtain the output
    cv::Mat output = net.forward();

    // Convert the output to a vector of floats (confidence scores)
    std::vector<float> confidences;
    confidences.assign((float*)output.datastart, (float*)output.dataend);

    // Construct a log message dynamically based on the confidence scores
    std::string log_message_content = "Inference run completed. Confidence scores: [";
    for (size_t i = 0; i < confidences.size(); ++i) {
        log_message_content += std::to_string(confidences[i]);
        if (i != confidences.size() - 1) {
            log_message_content += ", ";
        }
    }
    log_message_content += "]";
    log_message(log_message_content);

    return confidences;
}

// Decode image from base64-encoded string in HTTP request
cv::Mat decode_image(const std::string& image_data) {
    std::vector<uchar> buffer(image_data.begin(), image_data.end());
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR); // Decode to OpenCV Mat
    if (image.empty()) {
        throw std::runtime_error("Image decoding failed.");
    }
    log_message("Image decoded successfully");
    return image;
}

// Generate JSON response with classification results
std::string generate_response(const std::vector<float>& confidences) {
    Json::Value response;
    for (size_t i = 0; i < confidences.size(); ++i) {
        response["class_" + std::to_string(i)] = confidences[i];
    }
    Json::StreamWriterBuilder writer;
    log_message("Response generated: " + Json::writeString(writer, response));
    return Json::writeString(writer, response);
}

// Handle classification requests
void handle_classification(const httplib::Request& req, httplib::Response& res) {
    try {
        // Parse JSON request
        Json::Value json_data;
        Json::CharReaderBuilder reader;
        std::string errs;
        std::istringstream s(req.body);
        if (!Json::parseFromStream(reader, s, &json_data, &errs)) {
            res.status = 400;
            res.set_content("Invalid JSON format", "text/plain");
            log_message("Error: Invalid JSON format.");
            return;
        }

        // Extract and decode the image
        std::string image_data = json_data["image"].asString();
        cv::Mat image = decode_image(image_data);

        // Preprocess the image
        cv::Mat preprocessed_image = preprocess_image(image);

        // Run inference on the image
        std::vector<float> confidences = run_inference(preprocessed_image);

        // Prepare and send response
        std::string response_body = generate_response(confidences);
        res.set_content(response_body, "application/json");
        log_message("Request successfully processed.");
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content("Internal server error: " + std::string(e.what()), "text/plain");
        log_message("Error: " + std::string(e.what()));
    }
}

// Function to handle invalid routes
void handle_invalid_route(const httplib::Request& req, httplib::Response& res) {
    res.status = 404;
    res.set_content("Route not found", "text/plain");
    log_message("404 Error: Route not found.");
}

int main() {
    // Load the model before starting the server
    const std::string model_path = "path/to/model.onnx";
    log_message("Starting server initialization...");
    load_model(model_path);

    // Start HTTP server
    httplib::Server svr;

    // Define the API endpoint for image classification
    svr.Post("/classify", handle_classification);

    // Handle invalid routes
    svr.set_error_handler(handle_invalid_route);

    // Start listening on port 8080
    std::cout << "Server started on port 8080..." << std::endl;
    log_message("Server started on port 8080");
    svr.listen("0.0.0.0", 8080);

    return 0;
}