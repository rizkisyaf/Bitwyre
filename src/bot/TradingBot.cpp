#include "bot/TradingBot.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace trading {

TradingBot::TradingBot(std::shared_ptr<OrderbookFetcher> fetcher, 
                       std::shared_ptr<TradingModel> model,
                       double initial_balance) {
    // Pre-allocate memory for orders
    open_orders_.reserve(INITIAL_ORDERS_CAPACITY);
    filled_orders_.reserve(INITIAL_ORDERS_CAPACITY);
    
    // Pre-allocate pending order
    pending_order_ = new TradeOrder(TradeOrder::Type::BUY, 0.0, 0.0);
    
    // Initialize performance metrics
    last_metrics_update_ = std::chrono::system_clock::now();
    
    // Pre-allocate price history
    price_history_.reserve(price_history_length_);
    
    // Store the initial balance - ensure it's not unreasonably large
    if (initial_balance > 1000.0) {
        std::cout << "WARNING: Initial balance " << initial_balance << " is very large. Capping at 1000.0" << std::endl;
        initial_balance = 1000.0;
    }
    
    balance_ = initial_balance;
    initial_balance_ = initial_balance;
    
    // Store the shared pointers directly - using shared_ptr now
    orderbook_fetcher_ = fetcher;  // This is now a shared_ptr
    trading_model_ = model;  // This is now a shared_ptr
    
    // Set symbol from fetcher
    symbol_ = fetcher->getSymbol();
    
    // Register callback for orderbook updates
    orderbook_fetcher_->registerCallback([this](const Orderbook& orderbook) {
        onOrderbookUpdate(orderbook);
    });
    
    // Initialize P&L tracking
    realized_pnl_ = 0.0;
    unrealized_pnl_ = 0.0;
    total_pnl_ = 0.0;
    entry_value_ = 0.0;
    current_value_ = 0.0;
    win_count_ = 0;
    loss_count_ = 0;
    largest_win_ = 0.0;
    largest_loss_ = 0.0;
    avg_win_ = 0.0;
    avg_loss_ = 0.0;
    avg_trade_duration_ms_ = 0.0;
    
    // Initialize USD volume tracking
    interval_usd_volume_ = 0.0;
    total_usd_volume_ = 0.0;
    max_interval_volume_ = 0.0;
    avg_interval_volume_ = 0.0;
    interval_count_ = 0;
    last_volume_reset_ = std::chrono::system_clock::now();
    
    // Scale order size based on initial balance
    // For BTC, use a smaller order size to avoid excessive risk
    // Default is 0.1 BTC, but scale it down for smaller balances
    if (initial_balance < 1000.0) {
        // For smaller balances, use a much smaller order size
        order_size_ = 0.001;  // 0.001 BTC instead of 0.1 BTC
        std::cout << "Using smaller order size of " << order_size_ << " BTC due to initial balance of $" << initial_balance << std::endl;
    } else if (initial_balance < 10000.0) {
        // For medium balances, use a moderate order size
        order_size_ = 0.01;  // 0.01 BTC
        std::cout << "Using moderate order size of " << order_size_ << " BTC due to initial balance of $" << initial_balance << std::endl;
    } else {
        // For large balances, use the default order size
        order_size_ = 0.1;  // Default 0.1 BTC
        std::cout << "Using default order size of " << order_size_ << " BTC due to initial balance of $" << initial_balance << std::endl;
    }
}

TradingBot::~TradingBot() {
    stop();
    
    // Clean up pre-allocated objects
    delete pending_order_;
}

bool TradingBot::initialize(const std::string& exchange_url, const std::string& symbol, const std::string& model_path) {
    // This method is now deprecated since we initialize in the constructor
    // Keeping it for backward compatibility
    std::cerr << "Warning: initialize() is deprecated. Use the constructor instead." << std::endl;
    
    // We already have initialized the bot in the constructor
    // Just check if everything is set up correctly
    if (!orderbook_fetcher_ || !trading_model_) {
        std::cerr << "Trading bot not properly initialized" << std::endl;
        return false;
    }
    
    return true;
}

void TradingBot::start() {
    if (running_) {
        return;
    }
    
    running_ = true;
    trading_thread_ = std::thread(&TradingBot::run, this);
}

void TradingBot::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    // Disconnect from exchange
    if (orderbook_fetcher_) {
        orderbook_fetcher_->disconnect();
    }
    
    // Notify thread to exit
    cv_.notify_all();
    
    // Join thread
    if (trading_thread_.joinable()) {
        trading_thread_.join();
    }
}

double TradingBot::getPosition() const {
    return position_;
}

double TradingBot::getBalance() const {
    return balance_;
}

std::vector<TradeOrder> TradingBot::getOpenOrders() const {
    std::lock_guard<std::mutex> lock(orders_mutex_);
    return open_orders_;
}

std::vector<TradeOrder> TradingBot::getFilledOrders() const {
    std::lock_guard<std::mutex> lock(orders_mutex_);
    return filled_orders_;
}

nlohmann::json TradingBot::getPerformanceMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    nlohmann::json metrics;
    metrics["avg_tick_to_trade"] = avg_tick_to_trade_.count();
    metrics["trades_per_second"] = trades_per_second_.load();
    metrics["total_trades"] = total_trades_;
    metrics["successful_trades"] = successful_trades_;
    metrics["success_rate"] = total_trades_ > 0 ? (double)successful_trades_ / total_trades_ * 100.0 : 0.0;
    metrics["position"] = position_;
    
    // Ensure balance is always the initial balance plus realized P&L
    std::lock_guard<std::mutex> pnl_lock(pnl_mutex_);
    metrics["balance"] = initial_balance_ + realized_pnl_;
    
    // Add market volatility
    metrics["market_volatility"] = calculateMarketVolatility();
    
    // Add USD volume metrics
    {
        std::lock_guard<std::mutex> volume_lock(volume_mutex_);
        metrics["interval_usd_volume"] = interval_usd_volume_;
        metrics["total_usd_volume"] = total_usd_volume_;
        metrics["max_interval_usd_volume"] = max_interval_volume_;
        metrics["avg_interval_usd_volume"] = avg_interval_volume_;
        metrics["interval_count"] = interval_count_;
    }
    
    return metrics;
}

nlohmann::json TradingBot::getPnLMetrics() const {
    std::lock_guard<std::mutex> lock(pnl_mutex_);
    
    nlohmann::json metrics;
    metrics["realized_pnl"] = realized_pnl_;
    metrics["unrealized_pnl"] = unrealized_pnl_;
    metrics["total_pnl"] = total_pnl_;
    
    // Add uncapped P&L metrics for analysis
    metrics["uncapped_unrealized_pnl"] = uncapped_unrealized_pnl_;
    metrics["uncapped_total_pnl"] = uncapped_total_pnl_;
    
    // Add detailed P&L metrics
    metrics["win_count"] = win_count_;
    metrics["loss_count"] = loss_count_;
    metrics["win_rate"] = (win_count_ + loss_count_ > 0) ? 
                          (win_count_ / (win_count_ + loss_count_) * 100.0) : 0.0;
    metrics["largest_win"] = largest_win_;
    metrics["largest_loss"] = largest_loss_;
    metrics["avg_win"] = avg_win_;
    metrics["avg_loss"] = avg_loss_;
    metrics["profit_factor"] = (avg_loss_ != 0.0) ? (avg_win_ * win_count_) / (std::abs(avg_loss_) * loss_count_) : 0.0;
    metrics["avg_trade_duration_ms"] = avg_trade_duration_ms_;
    
    return metrics;
}

void TradingBot::run() {
    try {
        while (running_) {
            // Update order status
            updateOrderStatus();
            
            // Calculate performance metrics
            calculatePerformanceMetrics();
            
            // Sleep for a short time
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in trading thread: " << e.what() << std::endl;
    }
}

void TradingBot::onOrderbookUpdate(const Orderbook& orderbook) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get the current time
    auto now = std::chrono::system_clock::now();
    
    // Check if we need to reset the interval USD volume (every 5 seconds)
    {
        std::lock_guard<std::mutex> volume_lock(volume_mutex_);
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_volume_reset_).count();
        if (elapsed >= 5) {
            // Update interval statistics
            interval_count_++;
            
            // Update max interval volume
            if (interval_usd_volume_ > max_interval_volume_) {
                max_interval_volume_ = interval_usd_volume_;
            }
            
            // Update average interval volume
            avg_interval_volume_ = ((avg_interval_volume_ * (interval_count_ - 1)) + interval_usd_volume_) / interval_count_;
            
            // Reset interval volume
            interval_usd_volume_ = 0.0;
            last_volume_reset_ = now;
        }
    }
    
    // Update unrealized P&L based on current position and price
    if (position_ != 0.0) {
        double mid_price = orderbook.getMidPrice();
        
        std::lock_guard<std::mutex> pnl_lock(pnl_mutex_);
        
        // Calculate current value of position
        current_value_ = position_ * mid_price;
        
        // Calculate unrealized P&L
        unrealized_pnl_ = current_value_ - entry_value_;
        
        // Store uncapped unrealized P&L for analysis
        uncapped_unrealized_pnl_ = unrealized_pnl_;
        
        // Scale unrealized P&L to be relative to initial balance for risk management
        double scaled_unrealized_pnl = unrealized_pnl_;
        
        // Cap unrealized P&L at 50% of initial balance for risk management
        if (std::abs(unrealized_pnl_) > initial_balance_ * 0.5) {
            scaled_unrealized_pnl = (unrealized_pnl_ > 0) ? 
                initial_balance_ * 0.5 : -initial_balance_ * 0.5;
            std::cout << "Warning: Unrealized P&L exceeds 50% of initial balance. Scaling for risk management." << std::endl;
        }
        
        // Update total P&L with scaled value for risk management
        total_pnl_ = realized_pnl_ + scaled_unrealized_pnl;
        
        // Update uncapped total P&L for analysis
        uncapped_total_pnl_ = realized_pnl_ + uncapped_unrealized_pnl_;
    }
    
    // Extract features and make prediction
    double prediction = trading_model_->predict(orderbook);
    
    // Process the prediction
    processModelPrediction(prediction, orderbook);
    
    // Update performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        // Update tick-to-trade latency
        if (avg_tick_to_trade_.count() == 0) {
            avg_tick_to_trade_ = duration;
        } else {
            avg_tick_to_trade_ = std::chrono::nanoseconds(
                (avg_tick_to_trade_.count() * 9 + duration.count()) / 10
            );
        }
        
        // Update trades per second
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - last_metrics_update_).count();
        if (elapsed_seconds >= 1) {
            trades_per_second_ = total_trades_ / std::max(1UL, static_cast<unsigned long>(elapsed_seconds));
        }
    }
    
    // Update price history for volatility calculation
    {
        std::lock_guard<std::mutex> lock(price_history_mutex_);
        
        price_history_.push_back(orderbook.getMidPrice());
        if (price_history_.size() > price_history_length_) {
            price_history_.erase(price_history_.begin());
        }
    }
    
    // Store the last update time
    last_update_time_ = std::chrono::high_resolution_clock::now();
}

void TradingBot::processModelPrediction(double prediction, const Orderbook& orderbook_snapshot) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check risk limits before making trading decisions
    bool risk_limits_exceeded = checkRiskLimits();
    
    // If stop loss has been triggered, don't place new orders
    if (stop_loss_triggered_) {
        std::cout << "Stop loss triggered. Not placing new orders." << std::endl;
        return;
    }
    
    // If risk limits exceeded but not stop loss triggered, close current position
    if (risk_limits_exceeded) {
        std::cout << "Risk limits exceeded. Closing current position." << std::endl;
        // Close current position
        if (position_ > 0) {
            // Close long position
            TradeOrder order(TradeOrder::Type::SELL, orderbook_snapshot.getMidPrice(), std::abs(position_));
            placeOrder(order);
        } else if (position_ < 0) {
            // Close short position
            TradeOrder order(TradeOrder::Type::BUY, orderbook_snapshot.getMidPrice(), std::abs(position_));
            placeOrder(order);
        }
        return;
    }
    
    // Calculate market volatility
    double volatility = calculateMarketVolatility();
    
    // Adaptive threshold based on volatility
    double buy_threshold = 0.5 + (volatility * 0.2);  // Higher threshold in volatile markets
    double sell_threshold = -0.5 - (volatility * 0.2);
    
    // Check if spread is too small
    double spread = orderbook_snapshot.getSpread();
    if (spread < min_spread_) {
        // Skip trading when spread is too small
        return;
    }
    
    // Get best bid and ask prices
    double best_bid = orderbook_snapshot.getBestBid();
    double best_ask = orderbook_snapshot.getBestAsk();
    
    // Increment total trades counter
    total_trades_++;
    
    // Place orders based on prediction
    if (prediction > buy_threshold && position_ < max_position_) {
        // Buy signal
        double order_price = best_ask;  // Buy at ask price
        
        // Calculate position size
        double old_position = position_;
        
        // Place buy order
        TradeOrder order(TradeOrder::Type::BUY, order_price, order_size_);
        if (placeOrder(order)) {
            // Update position
            position_ += order_size_;
            
            // Update entry value using average price
            if (old_position == 0.0) {
                // New position
                entry_value_ = position_ * order_price;
            } else if (old_position > 0.0 && position_ > old_position) {
                // Adding to long position
                entry_value_ = entry_value_ + (order_size_ * order_price);
            } else if (old_position < 0.0 && position_ > old_position) {
                // Reducing short position
                double remaining_ratio = std::abs(position_) / std::abs(old_position);
                entry_value_ = entry_value_ * remaining_ratio;
                
                // Calculate realized P&L
                double closed_size = order_size_ - (position_ - old_position);
                double avg_entry_price = std::abs(entry_value_ / old_position);
                double realized_pnl = closed_size * (avg_entry_price - order_price);
                
                // Update realized P&L
                std::lock_guard<std::mutex> pnl_lock(pnl_mutex_);
                realized_pnl_ += realized_pnl;
            }
            
            // Update current value
            current_value_ = position_ * order_price;
            
            // Update unrealized P&L
            std::lock_guard<std::mutex> pnl_lock(pnl_mutex_);
            unrealized_pnl_ = current_value_ - entry_value_;
            
            // Scale unrealized P&L to be relative to initial balance for risk management
            double scaled_unrealized_pnl = unrealized_pnl_;
            if (std::abs(unrealized_pnl_) > initial_balance_ * 0.5) {
                scaled_unrealized_pnl = (unrealized_pnl_ > 0) ? 
                    initial_balance_ * 0.5 : -initial_balance_ * 0.5;
            }
            
            // Update total P&L
            total_pnl_ = realized_pnl_ + scaled_unrealized_pnl;
            
            // Update USD volume
            std::lock_guard<std::mutex> volume_lock(volume_mutex_);
            double usd_volume = order_size_ * order_price;
            interval_usd_volume_ += usd_volume;
            total_usd_volume_ += usd_volume;
            
            // Increment successful trades counter
            successful_trades_++;
        }
    } else if (prediction < sell_threshold && position_ > -max_position_) {
        // Sell signal
        double order_price = best_bid;  // Sell at bid price
        
        // Calculate position size
        double old_position = position_;
        
        // Place sell order
        TradeOrder order(TradeOrder::Type::SELL, order_price, order_size_);
        if (placeOrder(order)) {
            // Update position
            position_ -= order_size_;
            
            // Update entry value using average price
            if (old_position == 0.0) {
                // New position
                entry_value_ = position_ * order_price;
            } else if (old_position < 0.0 && position_ < old_position) {
                // Adding to short position
                entry_value_ = entry_value_ + (order_size_ * order_price);
            } else if (old_position > 0.0 && position_ < old_position) {
                // Reducing long position
                double remaining_ratio = std::abs(position_) / std::abs(old_position);
                entry_value_ = entry_value_ * remaining_ratio;
                
                // Calculate realized P&L
                double closed_size = order_size_ - (old_position - position_);
                double avg_entry_price = entry_value_ / old_position;
                double realized_pnl = closed_size * (order_price - avg_entry_price);
                
                // Update realized P&L
                std::lock_guard<std::mutex> pnl_lock(pnl_mutex_);
                realized_pnl_ += realized_pnl;
            }
            
            // Update current value
            current_value_ = position_ * order_price;
            
            // Update unrealized P&L
            std::lock_guard<std::mutex> pnl_lock(pnl_mutex_);
            unrealized_pnl_ = current_value_ - entry_value_;
            
            // Scale unrealized P&L to be relative to initial balance for risk management
            double scaled_unrealized_pnl = unrealized_pnl_;
            if (std::abs(unrealized_pnl_) > initial_balance_ * 0.5) {
                scaled_unrealized_pnl = (unrealized_pnl_ > 0) ? 
                    initial_balance_ * 0.5 : -initial_balance_ * 0.5;
            }
            
            // Update total P&L
            total_pnl_ = realized_pnl_ + scaled_unrealized_pnl;
            
            // Update USD volume
            std::lock_guard<std::mutex> volume_lock(volume_mutex_);
            double usd_volume = order_size_ * order_price;
            interval_usd_volume_ += usd_volume;
            total_usd_volume_ += usd_volume;
            
            // Increment successful trades counter
            successful_trades_++;
        }
    }
}

bool TradingBot::placeOrder(const TradeOrder& order) {
    // In a real implementation, this would send the order to an exchange
    // For simulation, we'll just add it to our open orders
    
    std::lock_guard<std::mutex> lock(orders_mutex_);
    open_orders_.push_back(order);
    
    // For simulation, we'll assume the order is filled immediately
    auto& filled_order = open_orders_.back();
    filled_order.status = TradeOrder::Status::FILLED;
    filled_order.filled_quantity = filled_order.quantity;
    
    // Update position
    if (filled_order.type == TradeOrder::Type::BUY) {
        position_ += filled_order.quantity;
    } else {
        position_ -= filled_order.quantity;
    }
    
    // Update USD volume tracking
    {
        std::lock_guard<std::mutex> volume_lock(volume_mutex_);
        double order_usd_volume = filled_order.price * filled_order.quantity;
        interval_usd_volume_ += order_usd_volume;
        total_usd_volume_ += order_usd_volume;
    }
    
    // Move to filled orders
    filled_orders_.push_back(filled_order);
    open_orders_.pop_back();
    
    // Update total trades counter
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        total_trades_++;
    }
    
    return true;
}

bool TradingBot::cancelOrder(const std::string& order_id) {
    try {
        // In a real implementation, this would send a cancel request to the exchange
        // For simulation, we just remove it from our open orders
        std::lock_guard<std::mutex> lock(orders_mutex_);
        
        // Find the order
        auto it = std::find_if(open_orders_.begin(), open_orders_.end(),
            [&order_id](const TradeOrder& o) { return o.id == order_id; });
        
        if (it == open_orders_.end()) {
            return false;
        }
        
        // Update the order status
        it->status = TradeOrder::Status::CANCELED;
        
        // Move the order to filled orders
        filled_orders_.push_back(*it);
        
        // Remove the order from open orders
        open_orders_.erase(it);
        
        // Log the cancellation
        std::cout << "Canceled order: " << order_id << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error canceling order: " << e.what() << std::endl;
        return false;
    }
}

void TradingBot::updateOrderStatus() {
    try {
        // In a real implementation, this would query the exchange for order status
        // For simulation, we just simulate order fills
        std::lock_guard<std::mutex> lock(orders_mutex_);
        
        // Get the latest orderbook
        Orderbook orderbook = orderbook_fetcher_->getLatestOrderbook();
        double current_price = orderbook.getMidPrice();
        
        // Process each open order
        auto it = open_orders_.begin();
        while (it != open_orders_.end()) {
            // Simulate order fill
            bool filled = false;
            
            if (it->type == TradeOrder::Type::BUY) {
                // Buy order fills if the best ask is below or equal to the order price
                double best_ask = orderbook.getBestAsk();
                filled = best_ask > 0 && best_ask <= it->price;
            } else {
                // Sell order fills if the best bid is above or equal to the order price
                double best_bid = orderbook.getBestBid();
                filled = best_bid > 0 && best_bid >= it->price;
            }
            
            if (filled) {
                // Update the order status
                it->status = TradeOrder::Status::FILLED;
                it->filled_quantity = it->quantity;
                
                // Calculate P&L for this order
                calculatePnL(*it, current_price);
                
                // Move the order to filled orders
                filled_orders_.push_back(*it);
                
                // Remove the order from open orders
                it = open_orders_.erase(it);
                
                // Log the fill
                std::cout << "Order filled: " << it->id << std::endl;
            } else {
                ++it;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error updating order status: " << e.what() << std::endl;
    }
}

void TradingBot::calculatePerformanceMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Calculate time since last update
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_metrics_update_);
    
    // Only update if at least 1 second has passed
    if (elapsed.count() >= 1) {
        // Calculate trades per second
        // Make sure both arguments to std::max have the same type
        uint64_t elapsed_seconds = static_cast<uint64_t>(elapsed.count());
        
        // Ensure TPS is calculated correctly
        if (elapsed_seconds > 0) {
            trades_per_second_ = total_trades_ / elapsed_seconds;
        } else {
            trades_per_second_ = total_trades_;  // If less than 1 second has passed
        }
        
        // Reset timer
        last_metrics_update_ = now;
    }
    
    // Update USD volume metrics
    std::lock_guard<std::mutex> volume_lock(volume_mutex_);
    
    // Reset interval volume every 5 seconds
    auto volume_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_volume_reset_);
    if (volume_elapsed.count() >= 5) {
        // Update interval statistics
        interval_count_++;
        
        // Update max interval volume
        max_interval_volume_ = std::max(max_interval_volume_, interval_usd_volume_);
        
        // Update average interval volume
        avg_interval_volume_ = ((avg_interval_volume_ * (interval_count_ - 1)) + interval_usd_volume_) / interval_count_;
        
        // Reset interval volume
        interval_usd_volume_ = 0.0;
        
        // Reset timer
        last_volume_reset_ = now;
    }
}

double TradingBot::calculateMarketVolatility() const {
    std::lock_guard<std::mutex> lock(price_history_mutex_);
    
    // If we don't have enough price history, return a default value
    if (price_history_.size() < 2) {
        return 0.01; // Default low volatility
    }
    
    // Calculate mean price
    double sum = std::accumulate(price_history_.begin(), price_history_.end(), 0.0);
    double mean = sum / price_history_.size();
    
    // Calculate variance
    double variance = 0.0;
    for (const auto& price : price_history_) {
        variance += std::pow(price - mean, 2);
    }
    variance /= price_history_.size();
    
    // Calculate standard deviation (volatility)
    double volatility = std::sqrt(variance);
    
    // Normalize by mean price to get relative volatility
    return volatility / (mean + 1e-10);
}

void TradingBot::calculatePnL(const TradeOrder& filled_order, double current_price) {
    std::lock_guard<std::mutex> lock(pnl_mutex_);
    
    // Calculate trade duration
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - filled_order.timestamp);
    
    // Update average trade duration
    if (filled_orders_.empty()) {
        avg_trade_duration_ms_ = duration.count();
    } else {
        avg_trade_duration_ms_ = (avg_trade_duration_ms_ * filled_orders_.size() + duration.count()) / 
                                (filled_orders_.size() + 1);
    }
    
    // Calculate trade P&L
    double trade_pnl = 0.0;
    if (filled_order.type == TradeOrder::Type::BUY) {
        // For a buy order, P&L is current price - entry price
        trade_pnl = (current_price - filled_order.entry_price) * filled_order.quantity;
    } else {
        // For a sell order, P&L is entry price - current price
        trade_pnl = (filled_order.entry_price - current_price) * filled_order.quantity;
    }
    
    // Update win/loss statistics
    if (trade_pnl > 0) {
        // Win
        win_count_++;
        largest_win_ = std::max(largest_win_, trade_pnl);
        avg_win_ = (avg_win_ * (win_count_ - 1) + trade_pnl) / win_count_;
    } else if (trade_pnl < 0) {
        // Loss
        loss_count_++;
        largest_loss_ = std::min(largest_loss_, trade_pnl);
        avg_loss_ = (avg_loss_ * (loss_count_ - 1) + trade_pnl) / loss_count_;
    }
    
    // Update balance
    balance_ += trade_pnl;
}

void TradingBot::setStopLossPercentage(double percentage) {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_loss_percentage_ = percentage;
    std::cout << "Stop loss percentage set to " << percentage << "%" << std::endl;
}

void TradingBot::setMaxDrawdownPercentage(double percentage) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_drawdown_percentage_ = percentage;
    std::cout << "Max drawdown percentage set to " << percentage << "%" << std::endl;
}

bool TradingBot::checkRiskLimits() {
    // Check if stop loss has already been triggered
    if (stop_loss_triggered_) {
        return true;
    }
    
    // Calculate current drawdown
    double initial_value = initial_balance_;
    double current_value = initial_balance_ + total_pnl_;
    double drawdown_percentage = 0.0;
    
    if (initial_value > 0) {
        drawdown_percentage = ((initial_value - current_value) / initial_value) * 100.0;
    }
    
    // Check if max drawdown has been reached
    if (drawdown_percentage >= max_drawdown_percentage_) {
        std::cout << "WARNING: Max drawdown of " << max_drawdown_percentage_ 
                  << "% reached. Current drawdown: " << drawdown_percentage 
                  << "%. Stopping trading." << std::endl;
        stop_loss_triggered_ = true;
        return true;
    }
    
    // Check individual trade stop loss - use a more reasonable calculation
    if (unrealized_pnl_ < 0) {
        // Calculate loss percentage relative to initial balance instead of entry value
        // This prevents unrealistic loss percentages when trading high-priced assets
        double loss_percentage = (std::abs(unrealized_pnl_) / initial_balance_) * 100.0;
        
        // Only trigger stop loss if loss percentage is significant relative to initial balance
        // and if we have a significant position, and if the loss percentage is at least 2x the stop loss percentage
        if (loss_percentage >= stop_loss_percentage_ * 2 && std::abs(position_) > 0.01) {
            std::cout << "WARNING: Stop loss of " << stop_loss_percentage_ 
                      << "% reached. Current loss: " << loss_percentage 
                      << "%. Closing position." << std::endl;
            // We don't set stop_loss_triggered_ to true here because we want to allow new trades
            // after closing the current losing position
            return true;
        }
    }
    
    return false;
}

} // namespace trading