#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include "data/scripts/cpp_preprocess.cpp"
#include "data/scripts/cpp_augment.cpp"
#include "models/cpp_inference.cpp"

// Test for Preprocessing Functionality
TEST(PreprocessTest, ResizeImage) {
    int result = preprocess_image("image_path", 256, 256);
    EXPECT_EQ(result, 0); // 0 means success
}

TEST(PreprocessTest, ResizeImageNegativeSize) {
    int result = preprocess_image("image_path", -100, -100);
    EXPECT_EQ(result, -1); // Should fail due to invalid size
}

TEST(PreprocessTest, NormalizeImage) {
    std::vector<float> image_data = {0.2, 0.5, 0.9};
    std::vector<float> normalized_image = normalize_image(image_data);
    
    EXPECT_NEAR(normalized_image[0], 0.0, 0.001);
    EXPECT_NEAR(normalized_image[2], 1.0, 0.001);
}

TEST(PreprocessTest, NormalizeImageEdgeCases) {
    std::vector<float> empty_data = {};
    std::vector<float> normalized_empty = normalize_image(empty_data);
    EXPECT_TRUE(normalized_empty.empty());

    std::vector<float> constant_data = {1.0, 1.0, 1.0};
    std::vector<float> normalized_constant = normalize_image(constant_data);
    for (auto value : normalized_constant) {
        EXPECT_NEAR(value, 1.0, 0.001);
    }
}

TEST(PreprocessTest, PreprocessTimeLimit) {
    auto start = std::chrono::high_resolution_clock::now();
    preprocess_image("image_path", 512, 512);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LT(duration, 50); // Ensure processing takes less than 50ms
}

// Test for Augmentation Functionality
TEST(AugmentTest, RotateImage) {
    int result = augment_image("image_path", 90); 
    EXPECT_EQ(result, 0); // Success for 90 degree rotation
}

TEST(AugmentTest, RotateInvalidAngle) {
    int result = augment_image("image_path", 400); // Invalid angle
    EXPECT_EQ(result, -1); // Should return error for invalid angle
}

TEST(AugmentTest, FlipImage) {
    int result = flip_image("image_path", true); 
    EXPECT_EQ(result, 0); // Success for horizontal flip
}

TEST(AugmentTest, FlipImageVertical) {
    int result = flip_image("image_path", false); // Vertical flip
    EXPECT_EQ(result, 0);
}

TEST(AugmentTest, AugmentWithInvalidFile) {
    int result = augment_image("invalid_image_path", 90);
    EXPECT_EQ(result, -1); // Failure expected with an invalid file
}

TEST(AugmentTest, MultipleAugmentations) {
    int result_rotate = augment_image("image_path", 180);
    EXPECT_EQ(result_rotate, 0);

    int result_flip = flip_image("image_path", true);
    EXPECT_EQ(result_flip, 0);

    int result_flip_vertical = flip_image("image_path", false);
    EXPECT_EQ(result_flip_vertical, 0);
}

TEST(AugmentTest, AugmentPerformance) {
    auto start = std::chrono::high_resolution_clock::now();
    augment_image("image_path", 45);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LT(duration, 30); // Augmentation should complete within 30ms
}

// Test for Inference Functionality
TEST(InferenceTest, InferenceAccuracy) {
    float accuracy = run_inference("image_path", "model_path");
    EXPECT_GT(accuracy, 0.8); 
}

TEST(InferenceTest, InferenceSpeed) {
    auto start = std::chrono::high_resolution_clock::now();
    run_inference("image_path", "model_path");
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LT(duration, 100); 
}

TEST(InferenceTest, InferenceOnCorruptedFile) {
    float accuracy = run_inference("corrupted_image_path", "model_path");
    EXPECT_EQ(accuracy, 0.0); // Corrupted file should return 0 accuracy
}

TEST(InferenceTest, InferenceLargeImage) {
    float accuracy = run_inference("large_image_path", "model_path");
    EXPECT_GT(accuracy, 0.75); // Expect reasonable accuracy even for large images
}

TEST(InferenceTest, InferenceWithDifferentModel) {
    float accuracy = run_inference("image_path", "different_model_path");
    EXPECT_GT(accuracy, 0.7); // Expect accuracy with different model 
}

TEST(InferenceTest, BatchInferenceSpeed) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        run_inference("image_path", "model_path");
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LT(duration, 5000); // Ensure batch inference completes within 5 seconds
}

TEST(InferenceTest, InvalidModelPath) {
    float accuracy = run_inference("image_path", "invalid_model_path");
    EXPECT_EQ(accuracy, -1); // Invalid model path should return an error or invalid accuracy
}

TEST(InferenceTest, InferenceTimeLimit) {
    auto start = std::chrono::high_resolution_clock::now();
    run_inference("image_path", "model_path");
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LT(duration, 80); // Inference should be completed within 80ms for performance reasons
}

// Additional Utility Functionality
TEST(UtilsTest, ValidateImagePath) {
    bool valid = validate_image_path("image_path");
    EXPECT_TRUE(valid);

    bool invalid = validate_image_path("invalid_path");
    EXPECT_FALSE(invalid);
}

TEST(UtilsTest, LoadModelPerformance) {
    auto start = std::chrono::high_resolution_clock::now();
    load_model("model_path");
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LT(duration, 100); // Model loading should complete within 100ms
}

TEST(UtilsTest, MemoryFootprintInference) {
    size_t initial_memory = get_memory_usage();
    run_inference("image_path", "model_path");
    size_t final_memory = get_memory_usage();
    
    EXPECT_LT(final_memory - initial_memory, 50000); // Ensure inference doesn't exceed memory limits
}

TEST(UtilsTest, MemoryFootprintAugmentation) {
    size_t initial_memory = get_memory_usage();
    augment_image("image_path", 90);
    size_t final_memory = get_memory_usage();
    
    EXPECT_LT(final_memory - initial_memory, 20000); // Ensure augmentation doesn't consume too much memory
}

// Main function for running all the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}