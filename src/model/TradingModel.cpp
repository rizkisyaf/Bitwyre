#include "model/TradingModel.hpp"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace trading {

TradingModel::TradingModel() {
    // Initialize price history with zeros
    price_history_.resize(price_history_length_, 0.0);
}

TradingModel::~TradingModel() {
    // No need to manually release the model, shared_ptr will handle it
}

bool TradingModel::loadModel(const std::string& model_path) {
    try {
        std::lock_guard<std::mutex> lock(model_mutex_);
        // Load the model
        model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));
        model_->eval();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

double TradingModel::predict(const Orderbook& orderbook) {
    try {
        std::lock_guard<std::mutex> lock(model_mutex_);
        
        if (!model_ || !model_->parameters().size()) {
            std::cerr << "Model not loaded or invalid" << std::endl;
            return 0.0;
        }
        
        // Extract features from the orderbook
        auto features = extractFeatures(orderbook);
        
        // Convert features to tensor
        auto inputs = featuresToTensor(features);
        
        // Make prediction using a better approach
        torch::NoGradGuard no_grad;
        auto output = model_->forward({inputs}).toTensor();
        
        // Return the prediction value
        return output.item<double>();
    } catch (const std::exception& e) {
        std::cerr << "Error making prediction: " << e.what() << std::endl;
        return 0.0;
    }
}

FeatureVector TradingModel::extractFeatures(const Orderbook& orderbook) {
    FeatureVector features;
    
    // Basic features
    features.mid_price = orderbook.getMidPrice();
    features.spread = orderbook.getSpread();
    
    const auto& bids = orderbook.getBids();
    const auto& asks = orderbook.getAsks();
    
    // Limit the depth
    int bid_depth = std::min(static_cast<int>(bids.size()), max_depth_);
    int ask_depth = std::min(static_cast<int>(asks.size()), max_depth_);
    
    // Extract bid and ask prices and quantities
    features.bid_prices.resize(max_depth_, 0.0);
    features.bid_quantities.resize(max_depth_, 0.0);
    features.ask_prices.resize(max_depth_, 0.0);
    features.ask_quantities.resize(max_depth_, 0.0);
    
    for (int i = 0; i < bid_depth; ++i) {
        features.bid_prices[i] = bids[i].price;
        features.bid_quantities[i] = bids[i].quantity;
    }
    
    for (int i = 0; i < ask_depth; ++i) {
        features.ask_prices[i] = asks[i].price;
        features.ask_quantities[i] = asks[i].quantity;
    }
    
    // Calculate additional features
    
    // Bid-ask imbalance
    double total_bid_quantity = std::accumulate(features.bid_quantities.begin(), 
                                               features.bid_quantities.end(), 0.0);
    double total_ask_quantity = std::accumulate(features.ask_quantities.begin(), 
                                               features.ask_quantities.end(), 0.0);
    
    features.bid_ask_imbalance = (total_bid_quantity - total_ask_quantity) / 
                                (total_bid_quantity + total_ask_quantity + 1e-10);
    
    // Volume-weighted mid price
    double weighted_bid_price = 0.0;
    double weighted_ask_price = 0.0;
    
    for (int i = 0; i < bid_depth; ++i) {
        weighted_bid_price += features.bid_prices[i] * features.bid_quantities[i];
    }
    
    for (int i = 0; i < ask_depth; ++i) {
        weighted_ask_price += features.ask_prices[i] * features.ask_quantities[i];
    }
    
    features.volume_weighted_mid_price = (weighted_bid_price + weighted_ask_price) / 
                                        (total_bid_quantity + total_ask_quantity + 1e-10);
    
    // Price momentum (using price history)
    if (features.mid_price > 0) {
        // Update price history
        price_history_.pop_back();
        price_history_.insert(price_history_.begin(), features.mid_price);
        
        // Calculate momentum as the slope of recent prices
        if (price_history_[0] > 0) {
            features.price_momentum = (price_history_[0] - price_history_[price_history_length_ - 1]) / 
                                     (price_history_[0] + 1e-10);
        }
    }
    
    return features;
}

torch::Tensor TradingModel::featuresToTensor(const FeatureVector& features) {
    // Create a vector to hold all features
    std::vector<double> feature_vector;
    
    // Add basic features
    feature_vector.push_back((features.mid_price - price_mean_) / (price_std_ + 1e-10));
    feature_vector.push_back(features.spread / (price_std_ + 1e-10));
    
    // Add bid and ask prices and quantities
    for (int i = 0; i < max_depth_; ++i) {
        feature_vector.push_back((features.bid_prices[i] - price_mean_) / (price_std_ + 1e-10));
        feature_vector.push_back((features.bid_quantities[i] - quantity_mean_) / (quantity_std_ + 1e-10));
        feature_vector.push_back((features.ask_prices[i] - price_mean_) / (price_std_ + 1e-10));
        feature_vector.push_back((features.ask_quantities[i] - quantity_mean_) / (quantity_std_ + 1e-10));
    }
    
    // Add additional features
    feature_vector.push_back(features.bid_ask_imbalance);
    feature_vector.push_back((features.volume_weighted_mid_price - price_mean_) / (price_std_ + 1e-10));
    feature_vector.push_back(features.price_momentum);
    
    // Convert to tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto tensor = torch::from_blob(feature_vector.data(), {1, static_cast<long>(feature_vector.size())}, options).clone();
    
    return tensor;
}

} // namespace trading 