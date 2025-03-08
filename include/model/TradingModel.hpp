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
constexpr int FEATURE_COUNT = 50; // Confirmed to match the feature vector size used in TradingBot

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
 * @brief Trading model class
 * 
 * This class is responsible for making predictions based on orderbook data.
 */
class TradingModel {
public:
    /**
     * @brief Default constructor
     */
    TradingModel();
    
    /**
     * @brief Constructor with model path and normalization parameters
     * 
     * @param model_path Path to the model file
     * @param mean_path Path to the mean values file
     * @param std_path Path to the standard deviation values file
     */
    TradingModel(const std::string& model_path, const std::string& mean_path, const std::string& std_path);
    
    /**
     * @brief Destructor
     */
    ~TradingModel();
    
    /**
     * @brief Load model from file
     * 
     * @param model_path Path to the model file
     * @return true if the model was loaded successfully
     */
    bool loadModel(const std::string& model_path);
    
    /**
     * @brief Make a prediction based on orderbook data
     * 
     * @param orderbook Orderbook data
     * @return double Prediction value between 0 and 1
     */
    double predict(const Orderbook& orderbook);
    
    /**
     * @brief Make a prediction based on a feature vector
     * 
     * @param features Feature vector
     * @return double Prediction value between 0 and 1
     */
    double predict(const std::vector<float>& features);
    
    /**
     * @brief Extract features from orderbook data
     * 
     * @param orderbook Orderbook data
     * @return FeatureVector Extracted features
     */
    FeatureVector extractFeatures(const Orderbook& orderbook);
    
    /**
     * @brief Convert features to tensor
     * 
     * @param features Feature vector
     * @return torch::Tensor Tensor representation of features
     */
    torch::Tensor featuresToTensor(const FeatureVector& features);
    
private:
    // Model
    std::shared_ptr<torch::jit::script::Module> model_;
    std::mutex model_mutex_;
    
    // Feature extraction parameters
    static constexpr int max_depth_ = MAX_DEPTH;
    static constexpr int price_history_length_ = 10;
    std::vector<double> price_history_;
    std::vector<double> volatility_history_;
    std::chrono::high_resolution_clock::time_point last_trade_time_;
    std::vector<double> trade_intervals_;
    
    // Normalization parameters
    std::vector<float> feature_mean_;
    std::vector<float> feature_std_;
    
    /**
     * @brief Load normalization parameters from files
     * 
     * @param mean_path Path to the mean values file
     * @param std_path Path to the standard deviation values file
     * @return true if parameters were loaded successfully
     */
    bool loadNormalizationParams(const std::string& mean_path, const std::string& std_path);
    
    /**
     * @brief Normalize features using loaded parameters
     * 
     * @param features Raw feature vector
     * @return std::vector<float> Normalized features
     */
    std::vector<float> normalizeFeatures(const std::vector<float>& features);
    
    /**
     * @brief Verify model input/output compatibility
     * 
     * @param input Input tensor to verify
     * @return true if input is compatible with model
     */
    bool verifyModelIO(const torch::Tensor& input);
};

} // namespace trading 