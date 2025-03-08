#include "bot/TradingBot.hpp"
#include "bot/BinanceApiTrader.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <sstream>

namespace trading {

TradingBot::TradingBot(std::shared_ptr<OrderbookFetcher> fetcher, 
                       std::shared_ptr<TradingModel> model,
                       double initial_balance,
                       const std::string& api_key,
                       const std::string& secret_key)
    : orderbook_fetcher_(fetcher),
      trading_model_(model),
      initial_balance_(initial_balance),
      balance_(initial_balance),
      last_update_time_(std::chrono::high_resolution_clock::now()),
      last_volume_reset_(std::chrono::system_clock::now())
{
    // Get the symbol from the orderbook fetcher
    symbol_ = orderbook_fetcher_->getSymbol();
    
    // Initialize price history
    price_history_.resize(price_history_length_, 0.0);
    
    // Set default order size based on initial balance
    // Increase order size to 0.1 BTC to ensure minimum notional value of 100 USDT
    order_size_ = 0.1;  // Fixed at 0.1 BTC to meet minimum notional requirements
    
    // If API keys are provided, initialize the API trader
    if (!api_key.empty() && !secret_key.empty()) {
        api_trader_ = std::make_shared<BinanceApiTrader>(api_key, secret_key);
        
        // Initialize the API trader
        if (!api_trader_->initialize()) {
            std::cerr << "❌ Failed to initialize API trader" << std::endl;
            return;
        }
        
        std::cout << "✔️ API trader initialized successfully" << std::endl;
        
        // Test API connection by getting account info
        auto account_info = api_trader_->getAccountInfo();
        if (account_info.is_null()) {
            std::cerr << "❌ Failed to connect to API" << std::endl;
            return;
        }
        
        std::cout << "✔️ API connection successful" << std::endl;
        
        // Check if futures trading is enabled
        if (api_trader_->getTradingMode() == BinanceApiTrader::TradingMode::USDM_FUTURES) {
            std::cout << "🚀 Futures trading enabled" << std::endl;
            
            // Set leverage to 5x
            if (!api_trader_->setLeverage(symbol_, leverage_)) {
                std::cerr << "❌ Failed to set leverage" << std::endl;
                return;
            }
            std::cout << "✅ Leverage set successfully" << std::endl;
            
            // Get account information
            if (!account_info.is_null()) {
                // For futures, get the USDT balance and positions
                if (account_info.contains("availableBalance")) {
                    double available_balance = std::stod(account_info["availableBalance"].get<std::string>());
                    std::cout << "💰 Available Balance: " << available_balance << " USDT" << std::endl;
                    
                    // Update initial balance if it's different
                    if (std::abs(available_balance - initial_balance_) > 0.01) {
                        std::cout << "🔄 Updating initial balance from " << initial_balance_ << " to " << available_balance << std::endl;
                        initial_balance_ = available_balance;
                        balance_ = available_balance;
                    }
                }

                // Get current positions
                if (account_info.contains("positions")) {
                    for (const auto& pos : account_info["positions"]) {
                        std::string pos_symbol = pos["symbol"].get<std::string>();
                        if (pos_symbol == symbol_) {
                            double pos_amt = std::stod(pos["positionAmt"].get<std::string>());
                            position_ = pos_amt;
                            
                            // Get unrealized PnL
                            if (pos.contains("unrealizedProfit")) {
                                unrealized_pnl_ = std::stod(pos["unrealizedProfit"].get<std::string>());
                            }
                            
                            // Get entry price
                            if (pos.contains("entryPrice")) {
                                double entry_price = std::stod(pos["entryPrice"].get<std::string>());
                                entry_value_ = position_ * entry_price;
                            }
                            
                            std::cout << "📊 Current Position: " << position_ << " BTC"
                                      << "\n💵 Entry Value: " << entry_value_ << " USDT"
                                      << "\n📈 Unrealized PnL: " << unrealized_pnl_ << " USDT" << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    // Pre-allocate memory for orders
    open_orders_.reserve(INITIAL_ORDERS_CAPACITY);
    filled_orders_.reserve(INITIAL_ORDERS_CAPACITY);
    
    // Pre-allocate pending order
    pending_order_ = new TradeOrder(TradeOrder::Type::BUY, 0.0, 0.0);
    
    // Initialize system time
    last_update_time_ = std::chrono::high_resolution_clock::now();
    
    // Pre-allocate price history
    price_history_.reserve(price_history_length_);
    
    // Store the initial balance - ensure it's not unreasonably large
    if (initial_balance > 1000.0) {
        std::cout << "WARNING: Initial balance " << initial_balance << " is very large. Capping at 1000.0" << std::endl;
        initial_balance = 1000.0;
    }
    
    balance_ = initial_balance;
    initial_balance_ = initial_balance;
    
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
    
    std::cout << "======= INITIALIZING REAL TRADING =======" << std::endl;
    
    std::cout << "=======================================" << std::endl;
}

TradingBot::~TradingBot() {
    stop();
    
    // Clean up pre-allocated objects
    delete pending_order_;
}

void TradingBot::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (running_) {
        return;
    }
    
    // Reset all flags
    running_ = true;
    emergency_stop_ = false;
    force_stop_ = false;
    active_operations_ = 0;
    websocket_connected_ = false;
    
    // Start the WebSocket thread
    websocket_thread_ = std::thread([this]() {
        try {
            websocket_connected_ = true;
            while (running_ && !emergency_stop_) {
                if (!websocket_connected_ && !force_stop_) {
                    orderbook_fetcher_->connect(exchange_url_, symbol_);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (const std::exception& e) {
            std::cerr << "WebSocket thread error: " << e.what() << std::endl;
        }
        websocket_connected_ = false;
    });
    
    // Start the trading thread
    trading_thread_ = std::thread(&TradingBot::run, this);
}

void TradingBot::stop() {
    bool was_running = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        was_running = running_;
        running_ = false;
    }
    
    if (was_running) {
        cv_.notify_all();
        shutdownThreads();
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
        std::cout << "\n=== Trading Bot Status ===\n"
                  << "Symbol: " << symbol_ << "\n"
                  << "Initial Balance: " << initial_balance_ << " USDT\n"
                  << "Stop Loss: " << stop_loss_percentage_ << "%\n"
                  << "Max Drawdown: " << max_drawdown_percentage_ << "%\n"
                  << "Position: " << position_ << "\n"
                  << "=====================\n" << std::endl;
                  
        while (running_ && !emergency_stop_) {
            // Increment active operations counter
            ++active_operations_;
            
            try {
                // Wait for a short time or until stop signal
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    if (cv_.wait_for(lock, std::chrono::milliseconds(100), 
                        [this] { return !running_ || emergency_stop_; })) {
                        break;
                    }
                }
                
                if (force_stop_) break;
                
                // Update order status with timeout
                auto update_future = std::async(std::launch::async, [this]() {
                    this->updateOrderStatus();
                });
                
                if (update_future.wait_for(std::chrono::milliseconds(WEBSOCKET_TIMEOUT_MS)) 
                    != std::future_status::ready) {
                    std::cerr << "Warning: Order status update timed out" << std::endl;
                }
                
                // Check risk limits
                if (checkRiskLimits()) {
                    std::cout << "⚠️ Risk limits reached. Stopping trading." << std::endl;
                    emergency_stop_ = true;
                    break;
                }
                
                // Print trading status every few iterations
                static int status_counter = 0;
                if (++status_counter % 30 == 0) {  // Adjust frequency as needed
                    std::cout << "\n--- Trading Status ---\n"
                              << "Position: " << position_ << " | Balance: " << balance_ << "\n"
                              << "P&L: Realized=" << realized_pnl_ << " | Unrealized=" << unrealized_pnl_ 
                              << " | Total=" << total_pnl_ << "\n"
                              << "Detailed P&L: Win/Loss=" << win_count_ << "/" << loss_count_
                              << " | Win Rate=" << (win_count_ + loss_count_ > 0 ? 
                                 (win_count_ * 100.0 / (win_count_ + loss_count_)) : 0) << "%"
                              << " | Profit Factor=" << (avg_loss_ != 0.0 ? 
                                 (avg_win_ * win_count_) / (std::abs(avg_loss_) * loss_count_) : 0.0)
                              << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in trading loop: " << e.what() << std::endl;
            }
            
            // Decrement active operations counter
            --active_operations_;
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal trading error: " << e.what() << std::endl;
    }
    
    // Ensure active_operations is zero when exiting
    active_operations_ = 0;
}

void TradingBot::onOrderbookUpdate(const Orderbook& orderbook) {
    try {
        // Record the update time
        auto now = std::chrono::high_resolution_clock::now();
        last_update_time_ = now;
        
        // Extract features from the orderbook
        std::vector<float> features;
        features.reserve(50); // Pre-allocate space for 50 features
        
        // Add basic features
        double best_bid = orderbook.getBestBid();
        double best_ask = orderbook.getBestAsk();
        double mid_price = orderbook.getMidPrice();
        double spread = orderbook.getSpread();
        
        std::cout << "\n📊 Orderbook Update:"
                  << "\n- Best Bid: " << best_bid
                  << "\n- Best Ask: " << best_ask
                  << "\n- Mid Price: " << mid_price
                  << "\n- Spread: " << spread << std::endl;
        
        features.push_back(static_cast<float>(best_bid));
        features.push_back(static_cast<float>(best_ask));
        features.push_back(static_cast<float>(mid_price));
        features.push_back(static_cast<float>(spread));
        
        // Add bid/ask imbalance
        auto bids = orderbook.getBids();
        auto asks = orderbook.getAsks();
        
        double bid_volume = 0.0;
        double ask_volume = 0.0;
        
        for (const auto& bid : bids) {
            bid_volume += bid.quantity;
        }
        
        for (const auto& ask : asks) {
            ask_volume += ask.quantity;
        }
        
        double imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume);
        features.push_back(static_cast<float>(imbalance));
        
        std::cout << "📈 Market Metrics:"
                  << "\n- Bid Volume: " << bid_volume
                  << "\n- Ask Volume: " << ask_volume
                  << "\n- Imbalance: " << imbalance << std::endl;
        
        // Add market volatility
        double volatility = calculateMarketVolatility();
        features.push_back(static_cast<float>(volatility));
        
        // Add position as a feature
        features.push_back(static_cast<float>(position_));
        
        // Add bid prices and quantities (10 levels)
        for (size_t i = 0; i < 10; i++) {
            if (i < bids.size()) {
                features.push_back(static_cast<float>(bids[i].price));
                features.push_back(static_cast<float>(bids[i].quantity));
            } else {
                features.push_back(0.0f);
                features.push_back(0.0f);
            }
        }
        
        // Add ask prices and quantities (10 levels)
        for (size_t i = 0; i < 10; i++) {
            if (i < asks.size()) {
                features.push_back(static_cast<float>(asks[i].price));
                features.push_back(static_cast<float>(asks[i].quantity));
            } else {
                features.push_back(0.0f);
                features.push_back(0.0f);
            }
        }
        
        // Pad remaining features
        while (features.size() < 50) {
            features.push_back(0.0f);
        }
        
        // Get prediction from the model
        double prediction;
        try {
            prediction = trading_model_->predict(features);
        } catch (const std::exception& e) {
            std::cerr << "❌ Model prediction error: " << e.what() << std::endl;
            std::cerr << "Skipping trade due to model error" << std::endl;
            return;
        }
        
        // Validate prediction
        if (std::isnan(prediction)) {
            std::cerr << "❌ Model returned NaN prediction - skipping trade" << std::endl;
            return;
        }
        
        std::string direction;
        if (prediction > 0.51) {
            direction = "Bullish";
        } else if (prediction < 0.49) {
            direction = "Bearish";
        } else {
            direction = "Neutral";
        }
        
        std::cout << "\n🤖 Model Prediction: " << std::fixed << std::setprecision(4) << prediction 
                  << " (" << direction << ")" << std::endl;
        
        // Update account info before processing prediction
        if (api_trader_) {
            auto account_info = api_trader_->getAccountInfo();
            if (account_info.contains("availableBalance")) {
                balance_ = std::stod(account_info["availableBalance"].get<std::string>());
                std::cout << "💰 Updated Balance: " << balance_ << " USDT" << std::endl;
            }
        }
        
        // Only process valid predictions
        if (prediction >= 0.0 && prediction <= 1.0) {
        processModelPrediction(prediction, orderbook);
        } else {
            std::cerr << "❌ Invalid prediction value: " << prediction << " - must be between 0 and 1" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error processing orderbook update: " << e.what() << std::endl;
    }
}

void TradingBot::processModelPrediction(double prediction, const Orderbook& orderbook_snapshot) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update price history for volatility calculation
    {
        std::lock_guard<std::mutex> lock(price_history_mutex_);
        double mid_price = orderbook_snapshot.getMidPrice();
        price_history_.push_back(mid_price);
        if (price_history_.size() > price_history_length_) {
        price_history_.erase(price_history_.begin());
        }
    }
    
    // Check risk limits before placing any orders
    if (checkRiskLimits()) {
        std::cout << "⚠️  Risk limits exceeded - no new orders" << std::endl;
        return;
    }
    
    const double MIN_NOTIONAL = 100.0;  // Minimum notional value in USDT
    const double TICK_SIZE = 0.1;       // Price tick size for BTCUSDT
    const double MIN_QTY = 0.001;       // Minimum quantity for BTCUSDT
    
    // Calculate available margin considering leverage
    double available_margin = balance_ * leverage_;
    
    // Use a fixed percentage of available margin per trade (e.g., 20%)
    double max_trade_margin = available_margin * 0.2;
    
    std::cout << "\n💫 Signal Analysis:"
              << "\n- Raw Prediction: " << prediction
              << "\n- Available Margin: " << available_margin << " USDT"
              << "\n- Max Trade Margin: " << max_trade_margin << " USDT"
              << "\n- Current Position: " << position_ << " BTC" << std::endl;
    
    // Get current market price from orderbook
    double market_price;
    if (prediction > 0.51) {  // For buy orders, use best ask
        market_price = std::ceil(orderbook_snapshot.getBestAsk() / TICK_SIZE) * TICK_SIZE;
    } else if (prediction < 0.49) {  // For sell orders, use best bid
        market_price = std::floor(orderbook_snapshot.getBestBid() / TICK_SIZE) * TICK_SIZE;
    } else {
        // For neutral predictions, use mid price
        market_price = std::round(orderbook_snapshot.getMidPrice() / TICK_SIZE) * TICK_SIZE;
    }
    
    // Calculate minimum quantity needed for MIN_NOTIONAL
    double min_quantity = std::ceil((MIN_NOTIONAL / market_price) * 1000.0) / 1000.0;
    
    // Calculate maximum quantity based on available margin
    double max_quantity = std::floor((max_trade_margin / market_price) * 1000.0) / 1000.0;
    
    // Use the minimum between max_quantity and min_quantity, but ensure it's at least MIN_QTY
    double order_quantity = std::max(MIN_QTY, std::min(max_quantity, min_quantity));
    
    std::cout << "📊 Order Calculation:"
              << "\n- Market Price: " << market_price
              << "\n- Min Quantity: " << min_quantity
              << "\n- Max Quantity: " << max_quantity
              << "\n- Final Quantity: " << order_quantity << std::endl;
    
    // Normalize prediction to [0,1]
    double normalized_prediction = std::min(1.0, std::max(0.0, prediction));
    double signal_strength = std::abs(normalized_prediction - 0.5) * 2.0;  // Convert to 0-1 scale
    
    // Strong signals only (>60% confidence)
    if (signal_strength > 0.2) {  // Requires at least 60% confidence
        if (normalized_prediction > 0.51) {  // BUY Signal
            std::cout << "\n🔵 BUY Signal [Confidence: " << std::fixed << std::setprecision(2) 
                      << signal_strength * 100 << "%]" << std::endl;
            
            // Check if we have enough margin for the trade
            double notional_value = order_quantity * market_price;
            if (notional_value < MIN_NOTIONAL) {
                std::cerr << "❌ Insufficient margin - Required: " << MIN_NOTIONAL 
                          << " USDT, Available: " << notional_value << " USDT" << std::endl;
                return;
            }
            
            // Place the buy order
            TradeOrder order(TradeOrder::Type::BUY, market_price, order_quantity);
            std::cout << "📝 Placing BUY order:" << std::endl
                      << "- Price: " << market_price << std::endl
                      << "- Quantity: " << order_quantity << std::endl
                      << "- Notional: " << notional_value << " USDT" << std::endl;
                      
            if (placeOrder(order)) {
                std::lock_guard<std::mutex> pnl_lock(pnl_mutex_);
                entry_value_ = notional_value;
                std::cout << "✅ BUY order placed successfully" << std::endl;
            }
            
        } else if (normalized_prediction < 0.49) {  // SELL Signal
            std::cout << "\n🔴 SELL Signal [Confidence: " << std::fixed << std::setprecision(2) 
                      << signal_strength * 100 << "%]" << std::endl;
            
            double notional_value = order_quantity * market_price;
            if (notional_value < MIN_NOTIONAL) {
                std::cerr << "❌ Insufficient margin - Required: " << MIN_NOTIONAL 
                          << " USDT, Available: " << notional_value << " USDT" << std::endl;
                return;
            }
            
            TradeOrder order(TradeOrder::Type::SELL, market_price, order_quantity);
            std::cout << "📝 Placing SELL order:" << std::endl
                      << "- Price: " << market_price << std::endl
                      << "- Quantity: " << order_quantity << std::endl
                      << "- Notional: " << notional_value << " USDT" << std::endl;
                      
        if (placeOrder(order)) {
                std::lock_guard<std::mutex> pnl_lock(pnl_mutex_);
                entry_value_ = notional_value;
                std::cout << "✅ SELL order placed successfully" << std::endl;
            }
        } else {
            std::cout << "\n⚪ NEUTRAL Signal - No trade" << std::endl;
        }
    } else {
        std::cout << "\n⚪ Weak Signal [Confidence: " << std::fixed << std::setprecision(2) 
                  << signal_strength * 100 << "%] - No trade" << std::endl;
    }
}

bool TradingBot::placeOrder(const TradeOrder& order) {
    // If we have an API trader, use it to place the order
    if (api_trader_) {
        std::cout << "Placing order via API trader: " 
                  << (order.type == TradeOrder::Type::BUY ? "BUY" : "SELL") 
                  << " " << order.quantity << " at " << order.price << std::endl;
        
        // Place the order using the API trader, passing the symbol
        if (api_trader_->placeOrder(order, symbol_)) {
            std::cout << "Order placed successfully via API" << std::endl;
            
            // Add to open orders
            {
                std::lock_guard<std::mutex> lock(orders_mutex_);
                open_orders_.push_back(order);
            }
            
            return true;
        } else {
            std::cerr << "Failed to place order via API" << std::endl;
            return false;
        }
    }
    
    // If we don't have an API trader, just simulate the order
    std::cout << "No API trader available, simulating order" << std::endl;
    
    // Add to open orders
    {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        open_orders_.push_back(order);
    }
    
    return true;
}

bool TradingBot::cancelOrder(const std::string& order_id) {
    if (api_trader_) {
        // Use the API trader for real trading
        std::cout << "REAL TRADE: Canceling order " << order_id << std::endl;
        bool success = api_trader_->cancelOrder(order_id, symbol_);
        
        if (success) {
            std::cout << "REAL TRADE: Order canceled successfully" << std::endl;
        } else {
            std::cerr << "REAL TRADE: Failed to cancel order" << std::endl;
        }
        
        return success;
    } else {
        std::cerr << "ERROR: API trader not initialized" << std::endl;
        return false;
    }
}

void TradingBot::updateOrderStatus() {
    if (!api_trader_) return;
    
    try {
        // Get account information
        auto account_info = api_trader_->getAccountInfo();
        
        // Update balance and positions
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Update balance from available balance
        if (account_info.contains("availableBalance")) {
            balance_ = std::stod(account_info["availableBalance"].get<std::string>());
        }
        
        // Update position from positions array
        position_ = 0.0;
        if (account_info.contains("positions")) {
            for (const auto& pos : account_info["positions"]) {
                if (pos["symbol"] == symbol_) {
                    position_ = std::stod(pos["positionAmt"].get<std::string>());
                    
                    // Update entry price and current value
                    double entry_price = std::stod(pos["entryPrice"].get<std::string>());
                    double mark_price = std::stod(pos["markPrice"].get<std::string>());
                    entry_value_ = std::abs(position_) * entry_price;
                    current_value_ = std::abs(position_) * mark_price;
                    
                    // Update unrealized PnL
                    std::lock_guard<std::mutex> pnl_lock(pnl_mutex_);
                    unrealized_pnl_ = std::stod(pos["unrealizedPnl"].get<std::string>());
                    total_pnl_ = realized_pnl_ + unrealized_pnl_;
                    break;
                }
            }
        }
        
        // Get open orders
        auto open_orders = api_trader_->getOpenOrders(symbol_);
        
        // Update open orders
        std::lock_guard<std::mutex> orders_lock(orders_mutex_);
        open_orders_ = open_orders;
        
    } catch (const std::exception& e) {
        std::cerr << "Error updating order status: " << e.what() << std::endl;
    }
}

double TradingBot::calculateMarketVolatility() const {
    std::lock_guard<std::mutex> lock(price_history_mutex_);
    
    if (price_history_.size() < 2) {
        return 0.0;
    }
    
    // Calculate returns
    std::vector<double> returns;
    returns.reserve(price_history_.size() - 1);
    
    for (size_t i = 1; i < price_history_.size(); ++i) {
        double ret = (price_history_[i] - price_history_[i - 1]) / price_history_[i - 1];
        returns.push_back(ret);
    }
    
    // Calculate standard deviation of returns
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    
    double sq_sum = std::inner_product(returns.begin(), returns.end(), returns.begin(), 0.0,
        std::plus<>(), [mean](double x, double y) { return (x - mean) * (y - mean); });
    
    double std_dev = std::sqrt(sq_sum / returns.size());
    
    return std_dev;
}

void TradingBot::calculatePnL(const TradeOrder& filled_order, double current_price) {
    std::lock_guard<std::mutex> lock(pnl_mutex_);
    
    // Calculate P&L for this trade
    double trade_pnl = 0.0;
    
    if (filled_order.type == TradeOrder::Type::BUY) {
        // For a buy, P&L is positive if current price > entry price
        trade_pnl = (current_price - filled_order.entry_price) * filled_order.filled_quantity;
    } else {
        // For a sell, P&L is positive if entry price > current price
        trade_pnl = (filled_order.entry_price - current_price) * filled_order.filled_quantity;
    }
    
    // Update realized P&L
    realized_pnl_ += trade_pnl;
    
    // Update win/loss statistics
    if (trade_pnl > 0.0) {
        win_count_++;
        largest_win_ = std::max(largest_win_, trade_pnl);
        avg_win_ = (avg_win_ * (win_count_ - 1) + trade_pnl) / win_count_;
    } else if (trade_pnl < 0.0) {
        loss_count_++;
        largest_loss_ = std::min(largest_loss_, trade_pnl);
        avg_loss_ = (avg_loss_ * (loss_count_ - 1) + trade_pnl) / loss_count_;
    }
    
    // Update total P&L
    total_pnl_ = realized_pnl_ + unrealized_pnl_;
    uncapped_total_pnl_ = realized_pnl_ + uncapped_unrealized_pnl_;
    
    // Calculate trade duration
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - filled_order.timestamp);
    
    // Update average trade duration
    double total_trades = win_count_ + loss_count_;
    if (total_trades > 0) {
        avg_trade_duration_ms_ = (avg_trade_duration_ms_ * (total_trades - 1) + duration.count()) / total_trades;
    }
}

bool TradingBot::checkRiskLimits() {
    // Check stop loss
    if (unrealized_pnl_ < -stop_loss_percentage_ * initial_balance_ / 100.0) {
        std::cout << "🛑 Stop loss triggered:\n"
                  << "  Unrealized P&L: " << unrealized_pnl_ << " USDT\n"
                  << "  Threshold: " << -stop_loss_percentage_ * initial_balance_ / 100.0 << " USDT"
                  << std::endl;
        stop_loss_triggered_ = true;
        return true;
    }
    
    // Check max drawdown
    if (total_pnl_ < -max_drawdown_percentage_ * initial_balance_ / 100.0) {
        std::cout << "📉 Max drawdown triggered:\n"
                  << "  Total P&L: " << total_pnl_ << " USDT\n"
                  << "  Threshold: " << -max_drawdown_percentage_ * initial_balance_ / 100.0 << " USDT"
                  << std::endl;
        return true;
    }
    
    return false;
}

void TradingBot::setStopLossPercentage(double percentage) {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_loss_percentage_ = percentage;
    std::cout << "Stop loss percentage set to " << stop_loss_percentage_ << "%" << std::endl;
}

void TradingBot::setMaxDrawdownPercentage(double percentage) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_drawdown_percentage_ = percentage;
    std::cout << "Max drawdown percentage set to " << max_drawdown_percentage_ << "%" << std::endl;
}

bool TradingBot::emergencyStop(int timeout_ms) {
    std::cout << "🚨 EMERGENCY STOP INITIATED" << std::endl;
    
    // Set emergency stop flag
    emergency_stop_ = true;
    running_ = false;
    cv_.notify_all();
    
    try {
        // Step 1: Try to cancel all orders with timeout
        std::cout << "1️⃣ Attempting to cancel all orders..." << std::endl;
        auto cancel_start = std::chrono::steady_clock::now();
        bool orders_cancelled = false;
        
        std::future<bool> cancel_future = std::async(std::launch::async, [this]() {
            return cancelAllOrders();
        });
        
        if (cancel_future.wait_for(std::chrono::milliseconds(ORDER_CANCEL_TIMEOUT_MS)) == std::future_status::ready) {
            orders_cancelled = cancel_future.get();
        }
        
        if (orders_cancelled) {
            std::cout << "✅ All orders cancelled successfully" << std::endl;
        } else {
            std::cout << "⚠️ Failed to cancel all orders or timed out" << std::endl;
        }
        
        // Step 2: Close WebSocket connection
        std::cout << "2️⃣ Closing WebSocket connection..." << std::endl;
        closeWebSocket();
        
        // Step 3: Wait for threads to finish gracefully
        std::cout << "3️⃣ Waiting for threads to finish..." << std::endl;
        bool graceful_shutdown = waitForThreads(timeout_ms);
        
        if (!graceful_shutdown) {
            std::cout << "⚠️ Graceful shutdown timed out, forcing stop..." << std::endl;
            force_stop_ = true;
            shutdownThreads(true);
            return false;
        }
        
        std::cout << "✅ Emergency stop completed successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during emergency stop: " << e.what() << std::endl;
        force_stop_ = true;
        shutdownThreads(true);
        return false;
    }
}

void TradingBot::shutdownThreads(bool force) {
    // Set flags to stop threads
    running_ = false;
    if (force) {
        force_stop_ = true;
        emergency_stop_ = true;
    }
    
    // Notify any waiting threads
    cv_.notify_all();
    
    // Close WebSocket first
    closeWebSocket();
    
    // Wait for threads to finish
    if (websocket_thread_.joinable()) {
        try {
            // Create a timeout mechanism for websocket thread
            auto start_time = std::chrono::high_resolution_clock::now();
            bool joined = false;
            
            // Try to join with a timeout
            std::thread temp_thread([this, &joined]() { 
                websocket_thread_.join(); 
                joined = true;
            });
            
            // Wait for the join to complete or timeout
            while (!joined) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - start_time).count();
                    
                if (elapsed > THREAD_JOIN_TIMEOUT_MS) {
                    std::cerr << "Timeout waiting for websocket thread" << std::endl;
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Clean up the temporary thread
            if (temp_thread.joinable()) {
                temp_thread.detach();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error joining websocket thread: " << e.what() << std::endl;
        }
    }
    
    if (trading_thread_.joinable()) {
        try {
            // Create a timeout mechanism for trading thread
            auto start_time = std::chrono::high_resolution_clock::now();
            bool joined = false;
            
            // Try to join with a timeout
            std::thread temp_thread([this, &joined]() { 
                trading_thread_.join(); 
                joined = true;
            });
            
            // Wait for the join to complete or timeout
            while (!joined) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - start_time).count();
                    
                if (elapsed > THREAD_JOIN_TIMEOUT_MS) {
                    std::cerr << "Timeout waiting for trading thread" << std::endl;
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Clean up the temporary thread
            if (temp_thread.joinable()) {
                temp_thread.detach();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error joining trading thread: " << e.what() << std::endl;
        }
    }
}

bool TradingBot::waitForThreads(int timeout_ms) {
    bool success = true;
    
    // Wait for trading thread to finish
    if (trading_thread_.joinable()) {
        std::cout << "Waiting for trading thread to finish..." << std::endl;
        
        // Create a timeout mechanism
        auto start_time = std::chrono::high_resolution_clock::now();
        while (trading_thread_.joinable()) {
            // Check if timeout has been reached
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();
                
            if (elapsed > timeout_ms) {
                std::cerr << "Timeout waiting for trading thread" << std::endl;
                success = false;
                break;
            }
            
            // Try to join with a small timeout
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    // Wait for websocket thread to finish
    if (websocket_thread_.joinable()) {
        std::cout << "Waiting for websocket thread to finish..." << std::endl;
        
        // Create a timeout mechanism
        auto start_time = std::chrono::high_resolution_clock::now();
        while (websocket_thread_.joinable()) {
            // Check if timeout has been reached
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();
                
            if (elapsed > timeout_ms) {
                std::cerr << "Timeout waiting for websocket thread" << std::endl;
                success = false;
                break;
            }
            
            // Try to join with a small timeout
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    return success;
}

void TradingBot::closeWebSocket() {
    if (orderbook_fetcher_) {
        orderbook_fetcher_->disconnect();
    }
}

bool TradingBot::cancelAllOrders() {
    std::lock_guard<std::mutex> lock(orders_mutex_);
    
    if (api_trader_) {
        // Use the API trader to cancel all orders
        return api_trader_->cancelAllOrders(symbol_);
    } else {
        std::cerr << "ERROR: API trader not initialized" << std::endl;
        return false;
    }
}

} // namespace trading