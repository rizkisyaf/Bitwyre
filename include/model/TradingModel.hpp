#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>

// Include the full definition of c10::IValue
#include <ATen/core/ivalue.h>
#include <torch/torch.h>
#include <torch/script.h>

#include "orderbook/OrderbookFetcher.hpp"

namespace trading {

/**
 * @brief Feature vector for the trading model
 */
struct FeatureVector {
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
};

/**
 * @brief Trading model that uses PyTorch to make predictions
 */
class TradingModel {
public:
    TradingModel();
    ~TradingModel();
    
    /**
     * @brief Load a trained model from a file
     * @param model_path The path to the trained model
     * @return true if the model was loaded successfully, false otherwise
     */
    bool loadModel(const std::string& model_path);
    
    /**
     * @brief Make a prediction based on the current orderbook
     * @param orderbook The current orderbook
     * @return A prediction value between -1 and 1 (-1 for sell, 1 for buy)
     */
    double predict(const Orderbook& orderbook);
    
    /**
     * @brief Extract features from the orderbook
     * @param orderbook The orderbook to extract features from
     * @return A feature vector
     */
    FeatureVector extractFeatures(const Orderbook& orderbook);
    
    /**
     * @brief Convert a feature vector to a tensor
     * @param features The feature vector
     * @return A tensor
     */
    torch::Tensor featuresToTensor(const FeatureVector& features);
    
private:
    std::shared_ptr<torch::jit::script::Module> model_;
    std::mutex model_mutex_;
    
    // Feature extraction configuration
    int max_depth_ = 10;
    int price_history_length_ = 10;
    std::vector<double> price_history_;
    
    // Normalization parameters
    double price_mean_ = 0.0;
    double price_std_ = 1.0;
    double quantity_mean_ = 0.0;
    double quantity_std_ = 1.0;
};

} // namespace trading 