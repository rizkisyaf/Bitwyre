#include "model/TradingModel.hpp"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

namespace trading {

TradingModel::TradingModel() {
    // Pre-allocate price history to avoid reallocations
    price_history_.reserve(price_history_length_);
    volatility_history_.reserve(price_history_length_);
    trade_intervals_.reserve(price_history_length_);
    
    // Initialize last trade time
    last_trade_time_ = std::chrono::high_resolution_clock::now();
}

TradingModel::TradingModel(const std::string& model_path, const std::string& mean_path, const std::string& std_path) {
    std::cout << "Initializing TradingModel with model path: " << model_path << std::endl;
    
    // Pre-allocate price history to avoid reallocations
    price_history_.reserve(price_history_length_);
    volatility_history_.reserve(price_history_length_);
    trade_intervals_.reserve(price_history_length_);
    
    // Initialize last trade time
    last_trade_time_ = std::chrono::high_resolution_clock::now();
    
    // Load the model
    if (!loadModel(model_path)) {
        std::cerr << "Failed to initialize model" << std::endl;
        throw std::runtime_error("Model initialization failed");
    }
    
    // Load normalization parameters if provided
    if (!mean_path.empty() && !std_path.empty()) {
        if (!loadNormalizationParams(mean_path, std_path)) {
            std::cerr << "Failed to load normalization parameters" << std::endl;
            std::cout << "Using default normalization" << std::endl;
        }
    }
    
    std::cout << "TradingModel initialization complete" << std::endl;
}

TradingModel::~TradingModel() {
    // No need to manually release the model, shared_ptr will handle it
}

bool TradingModel::loadModel(const std::string& model_path) {
    try {
        std::cout << "Loading model from path: " << model_path << std::endl;
        
        // Check if file exists
        std::ifstream f(model_path.c_str());
        if (!f.good()) {
            std::cerr << "Error: Model file does not exist at path: " << model_path << std::endl;
            return false;
        }
        
        // Load the model
        try {
            torch::jit::Module loaded_model = torch::jit::load(model_path);
            model_ = std::make_shared<torch::jit::Module>(std::move(loaded_model));
        } catch (const c10::Error& e) {
            std::cerr << "Failed to load the model: " << e.what() << std::endl;
            return false;
        }
        
        if (!model_) {
            std::cerr << "Error: Failed to load model" << std::endl;
            return false;
        }
        
        // Set to evaluation mode
        model_->eval();
        std::cout << "Model loaded and set to evaluation mode" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error loading model: " << e.what() << std::endl;
        return false;
    }
}

bool TradingModel::loadNormalizationParams(const std::string& mean_path, const std::string& std_path) {
    try {
        std::cout << "Loading normalization parameters from:" << std::endl;
        std::cout << "  Mean: " << mean_path << std::endl;
        std::cout << "  Std: " << std_path << std::endl;
        
        // Check if files exist
        std::ifstream mean_file(mean_path.c_str(), std::ios::binary);
        std::ifstream std_file(std_path.c_str(), std::ios::binary);
        
        if (!mean_file.good() || !std_file.good()) {
            std::cerr << "Error: Normalization parameter files do not exist" << std::endl;
            return false;
        }
        
        // Read the .npy files (simplified approach)
        // Skip the .npy header (first 128 bytes should be enough)
        mean_file.seekg(128);
        std_file.seekg(128);
        
        // Allocate vectors
        feature_mean_.resize(FEATURE_COUNT);
        feature_std_.resize(FEATURE_COUNT);
        
        // Read the data
        mean_file.read(reinterpret_cast<char*>(feature_mean_.data()), FEATURE_COUNT * sizeof(float));
        std_file.read(reinterpret_cast<char*>(feature_std_.data()), FEATURE_COUNT * sizeof(float));
        
        // Check if we read the correct amount of data
        if (mean_file.fail() || std_file.fail()) {
            std::cerr << "Error reading normalization parameters" << std::endl;
            return false;
        }
        
        std::cout << "Normalization parameters loaded successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading normalization parameters: " << e.what() << std::endl;
        return false;
    }
}

bool TradingModel::verifyModelIO(const torch::Tensor& input) {
    if (!model_) {
        std::cerr << "Error: Model not loaded" << std::endl;
        return false;
    }
    
    try {
        if (!input.defined()) {
            std::cerr << "Error: Input tensor is not defined" << std::endl;
            return false;
        }
        
        // Print detailed input tensor info
        std::cout << "\nInput Tensor Analysis:" << std::endl;
        std::cout << "Shape: [";
        for (int64_t i = 0; i < input.dim(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << input.size(i);
        }
        std::cout << "]" << std::endl;
        
        // Run test forward pass
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        
        std::cout << "\nRunning forward pass..." << std::endl;
        auto output = model_->forward(inputs);
        if (!output.isTensor()) {
            std::cerr << "Error: Model output is not a tensor" << std::endl;
            return false;
        }
        
        auto output_tensor = output.toTensor();
        
        // Print detailed output tensor info
        std::cout << "\nOutput Tensor Analysis:" << std::endl;
        std::cout << "Shape: [";
        for (int64_t i = 0; i < output_tensor.dim(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << output_tensor.size(i);
        }
        std::cout << "]" << std::endl;
        
        // Check for NaN values
        if (output_tensor.isnan().any().item<bool>()) {
            std::cerr << "Warning: NaN values detected in output" << std::endl;
        }
        
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error during model verification: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error during model verification: " << e.what() << std::endl;
        return false;
    }
}

double TradingModel::predict(const std::vector<float>& features) {
    if (!model_) {
        std::cerr << "❌ Error: Model not loaded" << std::endl;
        throw std::runtime_error("Model not loaded");
    }
    
    try {
        std::cout << "\nFeature Analysis:" << std::endl;
        std::cout << "Vector size: " << features.size() << std::endl;
        std::cout << "First 10 features: ";
        for (size_t i = 0; i < std::min(features.size(), (size_t)10); ++i) {
            std::cout << features[i] << " ";
        }
        std::cout << std::endl;
        
        // Check for invalid values in features and replace them
        bool has_invalid = false;
        std::vector<float> cleaned_features = features;
        for (size_t i = 0; i < cleaned_features.size(); ++i) {
            if (std::isnan(cleaned_features[i]) || std::isinf(cleaned_features[i])) {
                std::cout << "Invalid value at index " << i << ": " << cleaned_features[i] << " - replacing with 0" << std::endl;
                cleaned_features[i] = 0.0f;
                has_invalid = true;
            }
        }
        
        if (has_invalid) {
            std::cout << "Cleaned features (first 10): ";
            for (size_t i = 0; i < std::min(cleaned_features.size(), (size_t)10); ++i) {
                std::cout << cleaned_features[i] << " ";
            }
            std::cout << std::endl;
        }
        
        // Apply normalization
        std::vector<float> normalized_features = normalizeFeatures(cleaned_features);
        
        torch::NoGradGuard no_grad;
        auto input = torch::tensor(normalized_features).reshape({1, -1});
        
        if (!verifyModelIO(input)) {
            std::cerr << "Error: Model verification failed" << std::endl;
            throw std::runtime_error("Model verification failed");
        }
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        
        std::cout << "\nRunning prediction..." << std::endl;
        auto output = model_->forward(inputs).toTensor();
        
        // Handle potential NaN values
        if (output.isnan().any().item<bool>()) {
            std::cerr << "Error: Model produced NaN output" << std::endl;
            throw std::runtime_error("Model produced NaN output");
        }
        
        // Squeeze if necessary
        if (output.dim() == 2 && output.size(0) == 1 && output.size(1) == 1) {
            output = output.squeeze();
        }
        
        double raw_prediction = output.item<double>();
        
        // Check for invalid raw prediction
        if (std::isnan(raw_prediction) || std::isinf(raw_prediction)) {
            std::cerr << "Error: Invalid raw prediction value: " << raw_prediction << std::endl;
            throw std::runtime_error("Invalid raw prediction value");
        }
        
        // Apply sigmoid and ensure bounds
        double prediction = 1.0 / (1.0 + std::exp(-raw_prediction));
        prediction = std::max(0.0, std::min(1.0, prediction));
        
        std::cout << "Final prediction: " << prediction << std::endl;
        return prediction;
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error during prediction: " << e.what() << std::endl;
        throw std::runtime_error(std::string("PyTorch error: ") + e.what());
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error during prediction: " << e.what() << std::endl;
        throw;
    }
}

double TradingModel::predict(const Orderbook& orderbook) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    if (!model_) {
        std::cerr << "Error: Model not loaded" << std::endl;
        return 0.5;
    }
    
    try {
        auto features = extractFeatures(orderbook);
        std::vector<float> feature_vector;
        feature_vector.reserve(FEATURE_COUNT);
        
        // Convert features to vector
        // Add your feature extraction logic here based on the Python implementation
        
        return predict(feature_vector);
    } catch (const std::exception& e) {
        std::cerr << "Error during orderbook prediction: " << e.what() << std::endl;
        return 0.5;
    }
}

FeatureVector TradingModel::extractFeatures(const Orderbook& orderbook) {
    FeatureVector features;
    
    // Get bid and ask data
    const auto& bids = orderbook.getBids();
    const auto& asks = orderbook.getAsks();
    
    // Basic features
    features.mid_price = orderbook.getMidPrice();
    features.spread = orderbook.getSpread();
    
    // Extract bid and ask prices/quantities
    for (const auto& bid : bids) {
        features.bid_prices.push_back(bid.price);
        features.bid_quantities.push_back(bid.quantity);
    }
    
    for (const auto& ask : asks) {
        features.ask_prices.push_back(ask.price);
        features.ask_quantities.push_back(ask.quantity);
    }
    
    // Calculate additional features
    // Add your feature calculation logic here based on the Python implementation
    
    return features;
}

std::vector<float> TradingModel::normalizeFeatures(const std::vector<float>& features) {
    std::vector<float> normalized_features(features.size());
    
    // If normalization parameters are not loaded, return the original features
    if (feature_mean_.empty() || feature_std_.empty()) {
        return features;
    }
    
    // Apply normalization
    for (size_t i = 0; i < features.size(); ++i) {
        if (i < feature_mean_.size()) {
            // Handle NaN and Inf values
            if (std::isnan(features[i]) || std::isinf(features[i])) {
                normalized_features[i] = 0.0f;
            } else {
                // Z-score normalization
                normalized_features[i] = (features[i] - feature_mean_[i]) / feature_std_[i];
            }
        } else {
            normalized_features[i] = features[i];
        }
    }
    
    return normalized_features;
}

} // namespace trading