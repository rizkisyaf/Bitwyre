#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <csignal>
#include <nlohmann/json.hpp>

#include "orderbook/OrderbookFetcher.hpp"
#include "bot/TradingBot.hpp"
#include "model/TradingModel.hpp"

// Global variables for signal handling
volatile sig_atomic_t g_running = 1;

// Signal handler
void signalHandler(int signal) {
    g_running = 0;
}

int main(int argc, char* argv[]) {
    // Register signal handler
    std::signal(SIGINT, signalHandler);
    
    // Parse command line arguments
    std::string exchange_url = "wss://stream.binance.com:9443/ws";
    std::string symbol = "btcusdt";
    std::string model_path = "model.pt";
    
    if (argc > 1) {
        exchange_url = argv[1];
    }
    
    if (argc > 2) {
        symbol = argv[2];
    }
    
    if (argc > 3) {
        model_path = argv[3];
    }
    
    std::cout << "Starting trading bot..." << std::endl;
    std::cout << "Exchange URL: " << exchange_url << std::endl;
    std::cout << "Symbol: " << symbol << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    
    // Create the trading bot
    trading::TradingBot bot;
    
    // Initialize the bot
    if (!bot.initialize(exchange_url, symbol, model_path)) {
        std::cerr << "Failed to initialize trading bot" << std::endl;
        return 1;
    }
    
    // Start the bot
    bot.start();
    
    std::cout << "Trading bot started" << std::endl;
    
    // Main loop
    while (g_running) {
        // Print performance metrics every 5 seconds
        nlohmann::json metrics = bot.getPerformanceMetrics();
        
        std::cout << "Performance metrics:" << std::endl;
        std::cout << "  Avg tick-to-trade: " << metrics["avg_tick_to_trade_ns"].get<uint64_t>() << " ns" << std::endl;
        std::cout << "  Trades per second: " << metrics["trades_per_second"].get<uint64_t>() << std::endl;
        std::cout << "  Total trades: " << metrics["total_trades"].get<uint64_t>() << std::endl;
        std::cout << "  Successful trades: " << metrics["successful_trades"].get<uint64_t>() << std::endl;
        std::cout << "  Success rate: " << metrics["success_rate"].get<double>() * 100.0 << "%" << std::endl;
        std::cout << "  Position: " << bot.getPosition() << std::endl;
        std::cout << "  Balance: " << bot.getBalance() << std::endl;
        
        // Sleep for 5 seconds
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    
    std::cout << "Stopping trading bot..." << std::endl;
    
    // Stop the bot
    bot.stop();
    
    std::cout << "Trading bot stopped" << std::endl;
    
    return 0;
} 