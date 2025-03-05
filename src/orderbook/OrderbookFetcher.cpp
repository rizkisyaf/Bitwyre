#include "orderbook/OrderbookFetcher.hpp"

#include <algorithm>
#include <iostream>

namespace trading {

// Orderbook implementation
void Orderbook::updateBids(const std::vector<Order>& bids) {
    std::lock_guard<std::mutex> lock(mutex_);
    bids_ = bids;
    // Sort bids in descending order by price
    std::sort(bids_.begin(), bids_.end(), [](const Order& a, const Order& b) {
        return a.price > b.price;
    });
}

void Orderbook::updateAsks(const std::vector<Order>& asks) {
    std::lock_guard<std::mutex> lock(mutex_);
    asks_ = asks;
    // Sort asks in ascending order by price
    std::sort(asks_.begin(), asks_.end(), [](const Order& a, const Order& b) {
        return a.price < b.price;
    });
}

double Orderbook::getBestBid() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bids_.empty()) {
        return 0.0;
    }
    return bids_[0].price;
}

double Orderbook::getBestAsk() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (asks_.empty()) {
        return 0.0;
    }
    return asks_[0].price;
}

double Orderbook::getMidPrice() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bids_.empty() || asks_.empty()) {
        return 0.0;
    }
    return (bids_[0].price + asks_[0].price) / 2.0;
}

double Orderbook::getSpread() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bids_.empty() || asks_.empty()) {
        return 0.0;
    }
    return asks_[0].price - bids_[0].price;
}

// OrderbookFetcher implementation
OrderbookFetcher::OrderbookFetcher() : running_(false) {
    client_ = std::make_unique<websocket_client>();
    
    // Set up websocket client
    client_->clear_access_channels(websocketpp::log::alevel::all);
    client_->clear_error_channels(websocketpp::log::elevel::all);
    
    client_->init_asio();
    
    // Set TLS handler to accept all certificates (for testing only)
    client_->set_tls_init_handler([](websocketpp::connection_hdl) {
        return websocketpp::lib::make_shared<asio::ssl::context>(asio::ssl::context::sslv23);
    });
    
    // Register handlers
    client_->set_message_handler([this](websocketpp::connection_hdl hdl, websocket_client::message_ptr msg) {
        this->onMessage(client_.get(), hdl, msg);
    });
    
    client_->set_open_handler([this](websocketpp::connection_hdl hdl) {
        this->onOpen(client_.get(), hdl);
    });
    
    client_->set_close_handler([this](websocketpp::connection_hdl hdl) {
        this->onClose(client_.get(), hdl);
    });
    
    client_->set_fail_handler([this](websocketpp::connection_hdl hdl) {
        this->onFail(client_.get(), hdl);
    });
}

OrderbookFetcher::~OrderbookFetcher() {
    disconnect();
}

bool OrderbookFetcher::connect(const std::string& exchange_url, const std::string& symbol) {
    if (running_) {
        return false;
    }
    
    symbol_ = symbol;
    
    try {
        std::error_code ec;
        websocket_client::connection_ptr con = client_->get_connection(exchange_url, ec);
        
        if (ec) {
            std::cerr << "Could not create connection: " << ec.message() << std::endl;
            return false;
        }
        
        connection_ = con->get_handle();
        client_->connect(con);
        
        // Start the websocket thread
        running_ = true;
        websocket_thread_ = std::thread(&OrderbookFetcher::run, this);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return false;
    }
}

void OrderbookFetcher::disconnect() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    try {
        // Close the websocket connection
        std::error_code ec;
        client_->close(connection_, websocketpp::close::status::normal, "Disconnecting", ec);
        
        if (ec) {
            std::cerr << "Error closing connection: " << ec.message() << std::endl;
        }
        
        // Notify the websocket thread to exit
        cv_.notify_one();
        
        // Wait for the websocket thread to exit
        if (websocket_thread_.joinable()) {
            websocket_thread_.join();
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}

void OrderbookFetcher::registerCallback(OrderbookCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.push_back(callback);
}

Orderbook OrderbookFetcher::getLatestOrderbook() const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    // Create a copy of the orderbook
    Orderbook copy;
    copy.updateBids(orderbook_.getBids());
    copy.updateAsks(orderbook_.getAsks());
    return copy;
}

void OrderbookFetcher::run() {
    try {
        client_->run();
    } catch (const std::exception& e) {
        std::cerr << "Exception in websocket thread: " << e.what() << std::endl;
    }
    
    running_ = false;
}

void OrderbookFetcher::onMessage(websocket_client* client, websocketpp::connection_hdl hdl, 
                               websocket_client::message_ptr msg) {
    try {
        // Parse the message
        json data = json::parse(msg->get_payload());
        
        // Log the received message for debugging
        std::cout << "Received message from exchange: " << std::endl;
        std::cout << "Message type: " << (data.contains("e") ? data["e"].get<std::string>() : "unknown") << std::endl;
        
        // Process the orderbook update
        processOrderbookUpdate(data);
    } catch (const std::exception& e) {
        std::cerr << "Error in onMessage: " << e.what() << std::endl;
        std::cerr << "Message payload: " << msg->get_payload() << std::endl;
    }
}

void OrderbookFetcher::onOpen(websocket_client* client, websocketpp::connection_hdl hdl) {
    try {
        std::cout << "WebSocket connection established" << std::endl;
        
        // Store the connection handle
        connection_ = hdl;
        
        // Subscribe to the orderbook stream
        json subscription = {
            {"method", "SUBSCRIBE"},
            {"params", {symbol_ + "@depth"}},
            {"id", 1}
        };
        
        std::cout << "Sending subscription message: " << subscription.dump() << std::endl;
        
        client->send(connection_, subscription.dump(), websocketpp::frame::opcode::text);
        
        // Set the running flag
        running_ = true;
    } catch (const std::exception& e) {
        std::cerr << "Error in onOpen: " << e.what() << std::endl;
    }
}

void OrderbookFetcher::onClose(websocket_client* client, websocketpp::connection_hdl hdl) {
    std::cout << "Connection closed" << std::endl;
    running_ = false;
    cv_.notify_one();
}

void OrderbookFetcher::onFail(websocket_client* client, websocketpp::connection_hdl hdl) {
    std::cerr << "Connection failed" << std::endl;
    running_ = false;
    cv_.notify_one();
}

void OrderbookFetcher::processOrderbookUpdate(const json& data) {
    try {
        std::cout << "Processing orderbook update: " << std::endl;
        
        // Check if this is a partial or update message
        if (data.contains("lastUpdateId")) {
            // This is a partial orderbook
            std::cout << "Received partial orderbook" << std::endl;
            
            // Extract bids and asks
            std::vector<Order> bids;
            std::vector<Order> asks;
            
            if (data.contains("bids") && data["bids"].is_array()) {
                for (const auto& bid : data["bids"]) {
                    if (bid.is_array() && bid.size() >= 2) {
                        double price = std::stod(bid[0].get<std::string>());
                        double quantity = std::stod(bid[1].get<std::string>());
                        bids.emplace_back(price, quantity);
                    }
                }
                std::cout << "Processed " << bids.size() << " bids" << std::endl;
            }
            
            if (data.contains("asks") && data["asks"].is_array()) {
                for (const auto& ask : data["asks"]) {
                    if (ask.is_array() && ask.size() >= 2) {
                        double price = std::stod(ask[0].get<std::string>());
                        double quantity = std::stod(ask[1].get<std::string>());
                        asks.emplace_back(price, quantity);
                    }
                }
                std::cout << "Processed " << asks.size() << " asks" << std::endl;
            }
            
            // Update the orderbook
            {
                std::lock_guard<std::mutex> lock(orderbook_mutex_);
                orderbook_.updateBids(bids);
                orderbook_.updateAsks(asks);
            }
            
            // Notify callbacks
            {
                std::lock_guard<std::mutex> lock(callbacks_mutex_);
                for (const auto& callback : callbacks_) {
                    callback(orderbook_);
                }
            }
        } else if (data.contains("e") && data["e"] == "depthUpdate") {
            // This is an orderbook update
            std::cout << "Received orderbook update" << std::endl;
            
            // Get the current orderbook
            Orderbook current_orderbook;
            {
                std::lock_guard<std::mutex> lock(orderbook_mutex_);
                current_orderbook = orderbook_;
            }
            
            // Extract bids and asks
            std::vector<Order> bids = current_orderbook.getBids();
            std::vector<Order> asks = current_orderbook.getAsks();
            
            // Process bid updates
            if (data.contains("b") && data["b"].is_array()) {
                for (const auto& bid : data["b"]) {
                    if (bid.is_array() && bid.size() >= 2) {
                        double price = std::stod(bid[0].get<std::string>());
                        double quantity = std::stod(bid[1].get<std::string>());
                        
                        // Find the bid at this price level
                        auto it = std::find_if(bids.begin(), bids.end(),
                                            [price](const Order& order) {
                                                return order.price == price;
                                            });
                        
                        if (quantity > 0) {
                            // Update or add the bid
                            if (it != bids.end()) {
                                it->quantity = quantity;
                            } else {
                                bids.emplace_back(price, quantity);
                            }
                        } else {
                            // Remove the bid
                            if (it != bids.end()) {
                                bids.erase(it);
                            }
                        }
                    }
                }
                std::cout << "Processed " << data["b"].size() << " bid updates" << std::endl;
            }
            
            // Process ask updates
            if (data.contains("a") && data["a"].is_array()) {
                for (const auto& ask : data["a"]) {
                    if (ask.is_array() && ask.size() >= 2) {
                        double price = std::stod(ask[0].get<std::string>());
                        double quantity = std::stod(ask[1].get<std::string>());
                        
                        // Find the ask at this price level
                        auto it = std::find_if(asks.begin(), asks.end(),
                                            [price](const Order& order) {
                                                return order.price == price;
                                            });
                        
                        if (quantity > 0) {
                            // Update or add the ask
                            if (it != asks.end()) {
                                it->quantity = quantity;
                            } else {
                                asks.emplace_back(price, quantity);
                            }
                        } else {
                            // Remove the ask
                            if (it != asks.end()) {
                                asks.erase(it);
                            }
                        }
                    }
                }
                std::cout << "Processed " << data["a"].size() << " ask updates" << std::endl;
            }
            
            // Update the orderbook
            {
                std::lock_guard<std::mutex> lock(orderbook_mutex_);
                orderbook_.updateBids(bids);
                orderbook_.updateAsks(asks);
            }
            
            // Notify callbacks
            {
                std::lock_guard<std::mutex> lock(callbacks_mutex_);
                for (const auto& callback : callbacks_) {
                    callback(orderbook_);
                }
            }
        } else {
            // This is some other message
            std::cout << "Received other message type: " << data.dump() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in processOrderbookUpdate: " << e.what() << std::endl;
    }
}

} // namespace trading 