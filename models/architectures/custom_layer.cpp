#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <algorithm>

// Utility function to generate random floats in a given range
float random_float(float lower, float upper) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(lower, upper);
    return dis(gen);
}

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Derivative of the Sigmoid function for backpropagation
float sigmoid_derivative(float x) {
    float sig = sigmoid(x);
    return sig * (1 - sig);
}

// Hyperbolic tangent activation function
float tanh_activation(float x) {
    return std::tanh(x);
}

// Derivative of the Tanh function for backpropagation
float tanh_derivative(float x) {
    float tanh_x = std::tanh(x);
    return 1.0f - tanh_x * tanh_x;
}

// Activation function enum for handling different types of activations
enum class ActivationFunction {
    RELU,
    SIGMOID,
    TANH
};

// Function to apply the chosen activation
float apply_activation(float x, ActivationFunction func) {
    switch (func) {
        case ActivationFunction::RELU:
            return std::max(0.0f, x);
        case ActivationFunction::SIGMOID:
            return sigmoid(x);
        case ActivationFunction::TANH:
            return tanh_activation(x);
        default:
            return x; // No activation
    }
}

// Function to apply the derivative of the chosen activation
float apply_activation_derivative(float x, ActivationFunction func) {
    switch (func) {
        case ActivationFunction::RELU:
            return (x > 0) ? 1.0f : 0.0f;
        case ActivationFunction::SIGMOID:
            return sigmoid_derivative(x);
        case ActivationFunction::TANH:
            return tanh_derivative(x);
        default:
            return 1.0f; // No derivative
    }
}

// Custom Layer class definition
class CustomLayer {
public:
    // Constructor for initializing layer with input and output sizes, and the desired activation function
    CustomLayer(int input_size, int output_size, ActivationFunction activation_function = ActivationFunction::RELU)
        : input_size(input_size), output_size(output_size), activation_function(activation_function) {
        
        initialize_weights_and_biases();
    }

    // Forward pass function
    std::vector<float> forward(const std::vector<float>& input) {
        assert(input.size() == input_size);
        
        std::vector<float> output(output_size);
        for (int i = 0; i < output_size; ++i) {
            float activation = biases[i];
            for (int j = 0; j < input_size; ++j) {
                activation += weights[i][j] * input[j];
            }
            output[i] = apply_activation(activation, activation_function);
        }
        return output;
    }

    // Backward pass for calculating gradients and updating weights
    std::vector<float> backward(const std::vector<float>& d_output, const std::vector<float>& input, float learning_rate) {
        assert(d_output.size() == output_size);
        assert(input.size() == input_size);

        std::vector<float> d_input(input_size, 0.0f);
        
        for (int i = 0; i < output_size; ++i) {
            float d_activation = d_output[i] * apply_activation_derivative(biases[i], activation_function);
            for (int j = 0; j < input_size; ++j) {
                d_input[j] += d_activation * weights[i][j];
                weights[i][j] -= learning_rate * d_activation * input[j];  // Update weights
            }
            biases[i] -= learning_rate * d_activation;  // Update biases
        }
        return d_input;
    }

    // Function to print the current weights and biases
    void print_weights_and_biases() const {
        std::cout << "Weights: \n";
        for (const auto& row : weights) {
            for (float weight : row) {
                std::cout << weight << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Biases: \n";
        for (float bias : biases) {
            std::cout << bias << " ";
        }
        std::cout << std::endl;
    }

private:
    int input_size;
    int output_size;
    ActivationFunction activation_function;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;

    // Initialize weights and biases (using Xavier initialization)
    void initialize_weights_and_biases() {
        weights.resize(output_size, std::vector<float>(input_size));
        biases.resize(output_size);

        float limit = std::sqrt(6.0f / (input_size + output_size));
        for (int i = 0; i < output_size; ++i) {
            biases[i] = random_float(-limit, limit);  // Initialize biases randomly
            for (int j = 0; j < input_size; ++j) {
                weights[i][j] = random_float(-limit, limit);  // Xavier initialization for weights
            }
        }
    }
};

int main() {
    // Instantiate a Custom Layer with ReLU activation
    CustomLayer layer(4, 3, ActivationFunction::RELU);

    // Forward pass
    std::vector<float> input = {0.5f, -0.1f, 0.8f, 1.2f};
    std::vector<float> output = layer.forward(input);

    std::cout << "Forward pass output: ";
    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Print the initialized weights and biases
    layer.print_weights_and_biases();

    return 0;
}

#include <iomanip>
#include <numeric>

// Batch normalization function
std::vector<float> batch_normalize(const std::vector<float>& input, const std::vector<float>& gamma, const std::vector<float>& beta, float epsilon = 1e-5) {
    int size = input.size();
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / size;
    float variance = 0.0f;

    for (float x : input) {
        variance += (x - mean) * (x - mean);
    }
    variance /= size;

    std::vector<float> normalized(size);
    for (int i = 0; i < size; ++i) {
        normalized[i] = gamma[i] * (input[i] - mean) / std::sqrt(variance + epsilon) + beta[i];
    }

    return normalized;
}

// Dropout layer function
std::vector<float> apply_dropout(const std::vector<float>& input, float dropout_rate) {
    std::vector<float> output(input.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(1.0f - dropout_rate);

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = d(gen) ? input[i] : 0.0f;
    }

    return output;
}

// Additional initialization methods for weights
enum class InitializationMethod {
    XAVIER,
    HE,
    NORMAL
};

// Custom Layer class extended with batch normalization and dropout
class ExtendedCustomLayer {
public:
    // Constructor for initializing layer with batch normalization and dropout
    ExtendedCustomLayer(int input_size, int output_size, ActivationFunction activation_function = ActivationFunction::RELU, 
                        InitializationMethod init_method = InitializationMethod::XAVIER, 
                        bool use_batch_norm = false, float dropout_rate = 0.0f)
        : input_size(input_size), output_size(output_size), activation_function(activation_function), 
          use_batch_norm(use_batch_norm), dropout_rate(dropout_rate) {
        
        initialize_weights_and_biases(init_method);
        
        if (use_batch_norm) {
            gamma.resize(output_size, 1.0f);
            beta.resize(output_size, 0.0f);
        }
    }

    // Forward pass function
    std::vector<float> forward(const std::vector<float>& input) {
        assert(input.size() == input_size);

        std::vector<float> output(output_size);
        for (int i = 0; i < output_size; ++i) {
            float activation = biases[i];
            for (int j = 0; j < input_size; ++j) {
                activation += weights[i][j] * input[j];
            }
            output[i] = apply_activation(activation, activation_function);
        }

        // Apply batch normalization if enabled
        if (use_batch_norm) {
            output = batch_normalize(output, gamma, beta);
        }

        // Apply dropout if dropout rate > 0
        if (dropout_rate > 0.0f) {
            output = apply_dropout(output, dropout_rate);
        }

        return output;
    }

    // Backward pass with batch normalization and dropout support
    std::vector<float> backward(const std::vector<float>& d_output, const std::vector<float>& input, float learning_rate) {
        assert(d_output.size() == output_size);
        assert(input.size() == input_size);

        std::vector<float> d_input(input_size, 0.0f);

        for (int i = 0; i < output_size; ++i) {
            float d_activation = d_output[i] * apply_activation_derivative(biases[i], activation_function);
            for (int j = 0; j < input_size; ++j) {
                d_input[j] += d_activation * weights[i][j];
                weights[i][j] -= learning_rate * d_activation * input[j];  // Update weights
            }
            biases[i] -= learning_rate * d_activation;  // Update biases
        }

        return d_input;
    }

    // Print weights, biases, gamma, and beta (for batch normalization)
    void print_layer_parameters() const {
        std::cout << "Layer Parameters: \n";

        std::cout << "Weights: \n";
        for (const auto& row : weights) {
            for (float weight : row) {
                std::cout << std::setw(8) << weight << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Biases: \n";
        for (float bias : biases) {
            std::cout << bias << " ";
        }
        std::cout << std::endl;

        if (use_batch_norm) {
            std::cout << "Gamma (Batch Norm): \n";
            for (float g : gamma) {
                std::cout << g << " ";
            }
            std::cout << std::endl;

            std::cout << "Beta (Batch Norm): \n";
            for (float b : beta) {
                std::cout << b << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    int input_size;
    int output_size;
    ActivationFunction activation_function;
    bool use_batch_norm;
    float dropout_rate;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::vector<float> gamma;
    std::vector<float> beta;

    // Initialize weights and biases with different methods (Xavier, He, Normal)
    void initialize_weights_and_biases(InitializationMethod method) {
        weights.resize(output_size, std::vector<float>(input_size));
        biases.resize(output_size);

        float limit = (method == InitializationMethod::HE) ? std::sqrt(2.0f / input_size) 
                    : (method == InitializationMethod::XAVIER) ? std::sqrt(6.0f / (input_size + output_size))
                    : 0.05f;  // Default for normal

        for (int i = 0; i < output_size; ++i) {
            biases[i] = random_float(-limit, limit);  // Random initialization
            for (int j = 0; j < input_size; ++j) {
                weights[i][j] = random_float(-limit, limit);
            }
        }
    }
};

int main() {
    // Instantiate an Extended Layer with ReLU activation, batch normalization, and dropout
    ExtendedCustomLayer extended_layer(4, 3, ActivationFunction::RELU, InitializationMethod::HE, true, 0.2f);

    // Forward pass with batch normalization and dropout
    std::vector<float> input = {0.6f, -0.2f, 1.0f, 1.5f};
    std::vector<float> output = extended_layer.forward(input);

    std::cout << "Extended Layer Forward pass output: ";
    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Print extended layer parameters
    extended_layer.print_layer_parameters();

    return 0;
}

#include <fstream>
#include <sstream>

// Momentum update for weights
void apply_momentum(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& velocity, float learning_rate, float momentum = 0.9f) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            velocity[i][j] = momentum * velocity[i][j] - learning_rate * weights[i][j];
            weights[i][j] += velocity[i][j];
        }
    }
}

// Learning rate scheduler with decay
class LearningRateScheduler {
public:
    LearningRateScheduler(float initial_lr, float decay_rate, int decay_steps)
        : initial_lr(initial_lr), decay_rate(decay_rate), decay_steps(decay_steps), current_step(0) {}

    float get_learning_rate() {
        return initial_lr * std::pow(decay_rate, current_step / decay_steps);
    }

    void step() {
        current_step++;
    }

private:
    float initial_lr;
    float decay_rate;
    int decay_steps;
    int current_step;
};

// Saving model parameters to file
void save_model(const std::string& filename, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const auto& row : weights) {
            for (float weight : row) {
                file << weight << " ";
            }
            file << "\n";
        }

        file << "Biases\n";
        for (float bias : biases) {
            file << bias << " ";
        }
        file.close();
        std::cout << "Model saved to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file for saving model." << std::endl;
    }
}

// Loading model parameters from file
void load_model(const std::string& filename, std::vector<std::vector<float>>& weights, std::vector<float>& biases) {
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        for (auto& row : weights) {
            if (std::getline(file, line)) {
                std::istringstream iss(line);
                for (float& weight : row) {
                    iss >> weight;
                }
            }
        }

        if (std::getline(file, line) && line == "Biases") {
            for (float& bias : biases) {
                file >> bias;
            }
        }

        file.close();
        std::cout << "Model loaded from " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file for loading model." << std::endl;
    }
}

// Extended Custom Layer with momentum and learning rate scheduler
class AdvancedCustomLayer {
public:
    AdvancedCustomLayer(int input_size, int output_size, ActivationFunction activation_function = ActivationFunction::RELU, 
                        InitializationMethod init_method = InitializationMethod::XAVIER, 
                        bool use_batch_norm = false, float dropout_rate = 0.0f)
        : input_size(input_size), output_size(output_size), activation_function(activation_function), 
          use_batch_norm(use_batch_norm), dropout_rate(dropout_rate) {
        
        initialize_weights_and_biases(init_method);
        velocity.resize(output_size, std::vector<float>(input_size, 0.0f));  // Initialize velocity for momentum

        if (use_batch_norm) {
            gamma.resize(output_size, 1.0f);
            beta.resize(output_size, 0.0f);
        }
    }

    // Forward pass
    std::vector<float> forward(const std::vector<float>& input) {
        assert(input.size() == input_size);

        std::vector<float> output(output_size);
        for (int i = 0; i < output_size; ++i) {
            float activation = biases[i];
            for (int j = 0; j < input_size; ++j) {
                activation += weights[i][j] * input[j];
            }
            output[i] = apply_activation(activation, activation_function);
        }

        if (use_batch_norm) {
            output = batch_normalize(output, gamma, beta);
        }

        if (dropout_rate > 0.0f) {
            output = apply_dropout(output, dropout_rate);
        }

        return output;
    }

    // Backward pass with momentum
    std::vector<float> backward(const std::vector<float>& d_output, const std::vector<float>& input, float learning_rate) {
        assert(d_output.size() == output_size);
        assert(input.size() == input_size);

        std::vector<float> d_input(input_size, 0.0f);

        for (int i = 0; i < output_size; ++i) {
            float d_activation = d_output[i] * apply_activation_derivative(biases[i], activation_function);
            for (int j = 0; j < input_size; ++j) {
                d_input[j] += d_activation * weights[i][j];
                velocity[i][j] = 0.9f * velocity[i][j] - learning_rate * d_activation * input[j];
                weights[i][j] += velocity[i][j];  // Apply momentum update to weights
            }
            biases[i] -= learning_rate * d_activation;  // Update biases
        }

        return d_input;
    }

    // Save the current model to file
    void save(const std::string& filename) const {
        save_model(filename, weights, biases);
    }

    // Load model from file
    void load(const std::string& filename) {
        load_model(filename, weights, biases);
    }

private:
    int input_size;
    int output_size;
    ActivationFunction activation_function;
    bool use_batch_norm;
    float dropout_rate;
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> velocity;  // Velocity for momentum
    std::vector<float> biases;
    std::vector<float> gamma;
    std::vector<float> beta;

    void initialize_weights_and_biases(InitializationMethod method) {
        weights.resize(output_size, std::vector<float>(input_size));
        biases.resize(output_size);

        float limit = (method == InitializationMethod::HE) ? std::sqrt(2.0f / input_size)
                    : (method == InitializationMethod::XAVIER) ? std::sqrt(6.0f / (input_size + output_size))
                    : 0.05f;

        for (int i = 0; i < output_size; ++i) {
            biases[i] = random_float(-limit, limit);
            for (int j = 0; j < input_size; ++j) {
                weights[i][j] = random_float(-limit, limit);
            }
        }
    }
};

int main() {
    // Initialize an advanced custom layer with batch normalization and momentum
    AdvancedCustomLayer advanced_layer(4, 3, ActivationFunction::RELU, InitializationMethod::HE, true, 0.2f);

    // Forward pass
    std::vector<float> input = {0.5f, -0.3f, 0.7f, 1.4f};
    std::vector<float> output = advanced_layer.forward(input);

    std::cout << "Advanced Layer Forward pass output: ";
    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Save the model
    advanced_layer.save("model_params.txt");

    // Load the model back
    advanced_layer.load("model_params.txt");

    return 0;
}