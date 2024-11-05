#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <numeric>
#include <algorithm>
#include <execution>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <optional>

// Global mutex for thread-safe operations
std::mutex mtx;

// Caching for computational results to avoid redundant calculations
class ComputationCache {
public:
    std::optional<int> getCachedResult(int key) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void storeResult(int key, int result) {
        std::lock_guard<std::mutex> lock(mtx);
        cache[key] = result;
    }

private:
    std::unordered_map<int, int> cache;
};

// Pool allocator for optimizing memory management
template <typename T>
class MemoryPool {
public:
    MemoryPool(size_t poolSize) : poolSize(poolSize) {
        pool.resize(poolSize);
    }

    T* allocate() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!freeList.empty()) {
            T* ptr = freeList.back();
            freeList.pop_back();
            return ptr;
        }
        return new T;
    }

    void deallocate(T* ptr) {
        std::lock_guard<std::mutex> lock(mtx);
        freeList.push_back(ptr);
    }

private:
    size_t poolSize;
    std::vector<T*> pool;
    std::vector<T*> freeList;
};

// Simulate computationally heavy task with caching
int heavyComputation(int value, ComputationCache& cache) {
    auto cached = cache.getCachedResult(value);
    if (cached) {
        return cached.value();
    }

    // Simulate heavy computation
    int result = value * value - value / 2 + 1;
    cache.storeResult(value, result);

    return result;
}

// Optimized parallel computation using MemoryPool and cache
void parallelOptimizedComputation(std::vector<int>& data, size_t start, size_t end, ComputationCache& cache) {
    MemoryPool<int> pool(100);  // Memory pool

    for (size_t i = start; i < end; ++i) {
        int* ptr = pool.allocate();
        *ptr = heavyComputation(data[i], cache);
        data[i] = *ptr;
        pool.deallocate(ptr);
    }
}

// Task parallelism: Using futures to handle asynchronous tasks
std::future<void> runAsyncTask(std::vector<int>& data, size_t start, size_t end, ComputationCache& cache) {
    return std::async(std::launch::async, [&data, start, end, &cache]() {
        parallelOptimizedComputation(data, start, end, cache);
    });
}

// Parallel execution using async tasks
void asyncParallelExecution(std::vector<int>& data) {
    size_t dataSize = data.size();
    size_t batchSize = 10000000;  // Process 10 million elements per batch
    ComputationCache cache;

    for (size_t start = 0; start < dataSize; start += batchSize) {
        size_t end = std::min(start + batchSize, dataSize);
        
        size_t threadCount = std::thread::hardware_concurrency();
        size_t chunkSize = (end - start) / threadCount;

        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < threadCount; ++i) {
            size_t chunkStart = start + i * chunkSize;
            size_t chunkEnd = (i == threadCount - 1) ? end : chunkStart + chunkSize;
            futures.push_back(runAsyncTask(data, chunkStart, chunkEnd, cache));
        }

        for (auto& future : futures) {
            future.get();  // Wait for all tasks to complete
        }
    }
}

// Optimized SIMD Parallelism
void optimizedSIMDParallelism(std::vector<int>& data) {
    size_t dataSize = data.size();
    size_t batchSize = 10000000;  // Process 10 million elements per batch
    ComputationCache cache;

    for (size_t start = 0; start < dataSize; start += batchSize) {
        size_t end = std::min(start + batchSize, dataSize);

        // Apply SIMD parallelism to each batch
        std::for_each(std::execution::par_unseq, data.begin() + start, data.begin() + end, [&cache](int& x) {
            x = heavyComputation(x, cache);
        });
    }
}

// Time measurement utility for benchmarking
void benchmarkExecution(const std::string& description, const std::function<void()>& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << description << " execution time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << "ms" << std::endl;
}

// Lazy evaluation: Deferring computation until necessary
class LazyValue {
public:
    LazyValue(int initialValue) : value(initialValue), computed(false) {}

    int getValue() {
        if (!computed) {
            value = heavyComputation(value, cache);
            computed = true;
        }
        return value;
    }

private:
    int value;
    bool computed;
    ComputationCache cache;
};

// Lazy computation across a range
void lazyComputation(std::vector<LazyValue>& lazyValues) {
    std::for_each(lazyValues.begin(), lazyValues.end(), [](LazyValue& lazyVal) {
        std::cout << "Computed value: " << lazyVal.getValue() << std::endl;
    });
}

int main() {
    size_t dataSize = 100000000;  // 100 million elements
    std::vector<int> data(dataSize);
    std::iota(data.begin(), data.end(), 1);  // Fill the vector with sequential data

    ComputationCache cache;

    // 10 million elements
    size_t batchSize = 10000000;

    // Benchmark async parallel execution with batch processing
    benchmarkExecution("Async parallel with batching", [&]() {
        for (size_t start = 0; start < dataSize; start += batchSize) {
            size_t end = std::min(start + batchSize, dataSize);
            std::vector<int> batchData(data.begin() + start, data.begin() + end);
            // Perform parallel computation on this batch
            asyncParallelExecution(batchData);  // Run async task-based parallelism on the batch
            // Copy results back to main data vector
            std::copy(batchData.begin(), batchData.end(), data.begin() + start);
        }
    });

    // Benchmark SIMD optimized parallelism with batch processing
    benchmarkExecution("SIMD optimized parallelism with batching", [&]() {
        for (size_t start = 0; start < dataSize; start += batchSize) {
            size_t end = std::min(start + batchSize, dataSize);
            std::vector<int> batchData(data.begin() + start, data.begin() + end);
            optimizedSIMDParallelism(batchData);  // Run SIMD parallelism on the batch
            // Copy results back to main data vector
            std::copy(batchData.begin(), batchData.end(), data.begin() + start);
        }
    });

    // Lazy evaluation demo
    std::vector<LazyValue> lazyValues;
    for (int i = 0; i < 10; ++i) {
        lazyValues.emplace_back(i + 1);
    }
    lazyComputation(lazyValues);

    return 0;
}