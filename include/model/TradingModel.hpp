#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>

// Include the full definition of c10::IValue
#include <ATen/core/ivalue.h>
#include <torch/torch.h>
#include <torch/script.h>

#include "orderbook/OrderbookFetcher.hpp"

namespace trading {

// Pre-defined constants for feature vector size
constexpr int MAX_DEPTH = 10;
constexpr int FEATURE_COUNT = 50; // Updated to include new temporal features

// Aligned feature vector for better memory access patterns
struct alignas(32) FeatureVector {
    double mid_price;
    double spread;
    std::vector<double> bid_prices;
    std::vector<double> bid_quantities;
    std::vector<double> ask_prices;
    std::vector<double> ask_quantities;
    
    // Additional features
    double bid_ask_imbalance;
    double volume_weighted_mid_price;
    double price_momentum;
    
    // New temporal features
    double book_depth;
    double price_volatility;
    double ema_volatility;
    double time_since_last_trade;
    double trade_frequency;
    
    // Pre-allocate vectors to avoid reallocations
    FeatureVector() {
        bid_prices.reserve(MAX_DEPTH);
        bid_quantities.reserve(MAX_DEPTH);
        ask_prices.reserve(MAX_DEPTH);
        ask_quantities.reserve(MAX_DEPTH);
    }
};

/**
 * @brief Trading model that uses PyTorch for prediction
 */
class TradingModel {
public:
    TradingModel();
    TradingModel(const std::string& model_path);
    ~TradingModel();
    
    /**
     * @brief Load a PyTorch model from a file
     * 
     * @param model_path Path to the model file
     * @return true if the model was loaded successfully
     */
    bool loadModel(const std::string& model_path);
    
    /**
     * @brief Make a prediction based on the orderbook
     * 
     * @param orderbook Current orderbook state
     * @return double Prediction value
     */
    double predict(const Orderbook& orderbook);
    
    /**
     * @brief Extract features from the orderbook
     * 
     * @param orderbook Current orderbook state
     * @return FeatureVector Features extracted from the orderbook
     */
    FeatureVector extractFeatures(const Orderbook& orderbook);
    
    /**
     * @brief Convert features to a tensor for model input
     * 
     * @param features Features extracted from the orderbook
     * @return torch::Tensor Tensor for model input
     */
    torch::Tensor featuresToTensor(const FeatureVector& features);
    
private:
    std::shared_ptr<torch::jit::script::Module> model_;
    std::mutex model_mutex_;
    
    // Cache for intermediate tensors to avoid allocations
    torch::Tensor input_tensor_ = torch::zeros({1, FEATURE_COUNT});
    
    // Constants for feature normalization (made constexpr for compile-time evaluation)
    static constexpr int max_depth_ = MAX_DEPTH;
    static constexpr int price_history_length_ = 10;
    std::vector<double> price_history_;
    
    // New members for temporal features
    std::vector<double> volatility_history_;
    std::chrono::high_resolution_clock::time_point last_trade_time_;
    std::vector<double> trade_intervals_;
    
    // Normalization parameters
    double price_mean_ = 0.0;
    double price_std_ = 1.0;
    double quantity_mean_ = 0.0;
    double quantity_std_ = 1.0;
};

} // namespace trading 