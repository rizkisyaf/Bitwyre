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

TradingModel::TradingModel(const std::string& model_path) {
    // Pre-allocate price history to avoid reallocations
    price_history_.reserve(price_history_length_);
    volatility_history_.reserve(price_history_length_);
    trade_intervals_.reserve(price_history_length_);
    
    // Initialize last trade time
    last_trade_time_ = std::chrono::high_resolution_clock::now();
    
    // Load the model
    loadModel(model_path);
}

TradingModel::~TradingModel() {
    // No need to manually release the model, shared_ptr will handle it
}

bool TradingModel::loadModel(const std::string& model_path) {
    try {
        std::lock_guard<std::mutex> lock(model_mutex_);
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
        // Extract features and convert to tensor
        auto features = extractFeatures(orderbook);
        auto input = featuresToTensor(features);
        
        // Make prediction
        std::lock_guard<std::mutex> lock(model_mutex_);
        if (!model_) {
            std::cerr << "Model not loaded" << std::endl;
            return 0.0;
        }
        
        // Use no_grad to disable gradient calculation for inference
        torch::NoGradGuard no_grad;
        
        // Forward pass
        auto output = model_->forward({input}).toTensor();
        
        // Get prediction value
        return output[0].item<double>();
    } catch (const std::exception& e) {
        std::cerr << "Error making prediction: " << e.what() << std::endl;
        return 0.0;
    }
}

FeatureVector TradingModel::extractFeatures(const Orderbook& orderbook) {
    FeatureVector features;
    
    // Get basic orderbook features
    features.mid_price = orderbook.getMidPrice();
    features.spread = orderbook.getSpread();
    
    // Get bids and asks (use const references to avoid copies)
    const auto& bids = orderbook.getBids();
    const auto& asks = orderbook.getAsks();
    
    // Pre-calculate sizes to avoid repeated calls
    const size_t bid_size = bids.size();
    const size_t ask_size = asks.size();
    
    // Calculate bid-ask imbalance with more precision
    double total_bid_value = 0.0;
    double total_ask_value = 0.0;
    double total_bid_quantity = 0.0;
    double total_ask_quantity = 0.0;
    
    // Calculate total quantities and values
    for (size_t i = 0; i < std::min(bid_size, static_cast<size_t>(max_depth_)); ++i) {
        total_bid_quantity += bids[i].quantity;
        total_bid_value += bids[i].price * bids[i].quantity;
    }
    
    for (size_t i = 0; i < std::min(ask_size, static_cast<size_t>(max_depth_)); ++i) {
        total_ask_quantity += asks[i].quantity;
        total_ask_value += asks[i].price * asks[i].quantity;
    }
    
    // Calculate bid-ask imbalance
    features.bid_ask_imbalance = (total_bid_value - total_ask_value) / 
                               (total_bid_value + total_ask_value + 1e-10);
    
    // Calculate volume-weighted mid price
    double vwmp = 0.0;
    double total_quantity = total_bid_quantity + total_ask_quantity;
    
    if (total_quantity > 0) {
        vwmp = (features.mid_price * total_quantity + 
                (bid_size > 0 ? bids[0].price * total_bid_quantity : 0) + 
                (ask_size > 0 ? asks[0].price * total_ask_quantity : 0)) / (2 * total_quantity);
    } else {
        vwmp = features.mid_price;
    }
    
    features.volume_weighted_mid_price = vwmp;
    
    // Add book depth feature
    features.book_depth = std::min(bid_size, ask_size);
    
    // Calculate time since last trade
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - last_trade_time_).count();
    features.time_since_last_trade = static_cast<double>(time_diff);
    
    // Update last trade time
    last_trade_time_ = current_time;
    
    // Update trade intervals
    if (!trade_intervals_.empty()) {
        trade_intervals_.push_back(features.time_since_last_trade);
        if (trade_intervals_.size() > price_history_length_) {
            trade_intervals_.erase(trade_intervals_.begin());
        }
        
        // Calculate trade frequency (trades per second)
        double sum_intervals = std::accumulate(trade_intervals_.begin(), trade_intervals_.end(), 0.0);
        features.trade_frequency = trade_intervals_.size() * 1000.0 / (sum_intervals + 1e-10);
    } else {
        features.trade_frequency = 0.0;
    }
    
    // Update price history
    price_history_.push_back(features.mid_price);
    if (price_history_.size() > price_history_length_) {
        price_history_.erase(price_history_.begin());
    }
    
    // Calculate price momentum and volatility
    if (price_history_.size() >= 2) {
        features.price_momentum = (price_history_.back() - price_history_.front()) / 
                                 (price_history_.front() + 1e-10);
        
        // Calculate price volatility
        double sum_squared_diff = 0.0;
        double mean_price = std::accumulate(price_history_.begin(), price_history_.end(), 0.0) / 
                           price_history_.size();
        
        for (const auto& price : price_history_) {
            sum_squared_diff += std::pow(price - mean_price, 2);
        }
        
        features.price_volatility = std::sqrt(sum_squared_diff / price_history_.size());
        
        // Calculate EMA volatility
        double alpha = 0.1; // EMA factor
        features.ema_volatility = alpha * features.price_volatility + 
                                (1 - alpha) * (volatility_history_.empty() ? 0 : volatility_history_.back());
        
        // Update volatility history
        volatility_history_.push_back(features.ema_volatility);
        if (volatility_history_.size() > price_history_length_) {
            volatility_history_.erase(volatility_history_.begin());
        }
    } else {
        features.price_momentum = 0.0;
        features.price_volatility = 0.0;
        features.ema_volatility = 0.0;
    }
    
    // Extract price and quantity features
    for (size_t i = 0; i < max_depth_; ++i) {
        // Add bid prices and quantities
        features.bid_prices.push_back(i < bid_size ? bids[i].price : 0.0);
        features.bid_quantities.push_back(i < bid_size ? bids[i].quantity : 0.0);
        
        // Add ask prices and quantities
        features.ask_prices.push_back(i < ask_size ? asks[i].price : 0.0);
        features.ask_quantities.push_back(i < ask_size ? asks[i].quantity : 0.0);
    }
    
    return features;
}

torch::Tensor TradingModel::featuresToTensor(const FeatureVector& features) {
    // Use pre-allocated tensor to avoid memory allocation
    auto& tensor = input_tensor_;
    
    // Reset tensor values to zero
    tensor.zero_();
    
    // Get data pointer for direct access
    float* data_ptr = tensor.data_ptr<float>();
    
    // Fill tensor with normalized features
    // Using direct indexing instead of push_back to avoid bounds checking
    
    // Basic features
    data_ptr[0] = static_cast<float>((features.mid_price - price_mean_) / (price_std_ + 1e-10));
    data_ptr[1] = static_cast<float>(features.spread / (price_std_ + 1e-10));
    
    // Additional features
    data_ptr[2] = static_cast<float>(features.bid_ask_imbalance);
    data_ptr[3] = static_cast<float>((features.volume_weighted_mid_price - price_mean_) / (price_std_ + 1e-10));
    data_ptr[4] = static_cast<float>(features.price_momentum);
    
    // New temporal features
    data_ptr[5] = static_cast<float>(features.book_depth / max_depth_); // Normalize by max depth
    data_ptr[6] = static_cast<float>(features.price_volatility / (price_std_ + 1e-10));
    data_ptr[7] = static_cast<float>(features.ema_volatility / (price_std_ + 1e-10));
    data_ptr[8] = static_cast<float>(std::min(features.time_since_last_trade / 1000.0, 10.0)); // Cap at 10 seconds
    data_ptr[9] = static_cast<float>(features.trade_frequency / 100.0); // Normalize assuming max 100 trades/sec
    
    // Bid prices and quantities
    const size_t bid_size = features.bid_prices.size();
    for (size_t i = 0; i < max_depth_; ++i) {
        // Use ternary operator instead of if-else to reduce branches
        data_ptr[10 + i] = static_cast<float>((i < bid_size ? features.bid_prices[i] : 0.0) - price_mean_) / (price_std_ + 1e-10);
        data_ptr[10 + max_depth_ + i] = static_cast<float>((i < bid_size ? features.bid_quantities[i] : 0.0) - quantity_mean_) / (quantity_std_ + 1e-10);
    }
    
    // Ask prices and quantities
    const size_t ask_size = features.ask_prices.size();
    for (size_t i = 0; i < max_depth_; ++i) {
        data_ptr[10 + 2 * max_depth_ + i] = static_cast<float>((i < ask_size ? features.ask_prices[i] : 0.0) - price_mean_) / (price_std_ + 1e-10);
        data_ptr[10 + 3 * max_depth_ + i] = static_cast<float>((i < ask_size ? features.ask_quantities[i] : 0.0) - quantity_mean_) / (quantity_std_ + 1e-10);
    }
    
    return tensor;
}

} // namespace trading 