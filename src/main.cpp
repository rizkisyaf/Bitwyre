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
    std::string symbol = "btcusdt";
    std::string exchange_url = "wss://stream.binance.com:9443/ws";
    int duration_seconds = 60;
    double initial_balance = 100.0;  // Default initial balance
    double stop_loss_pct = 0.5;        // Default stop loss percentage
    double max_drawdown_pct = 5.0;     // Default max drawdown percentage
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--symbol" && i + 1 < argc) {
            symbol = argv[++i];
        } else if (arg == "--exchange-url" && i + 1 < argc) {
            exchange_url = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            duration_seconds = std::stoi(argv[++i]);
        } else if (arg == "--balance" && i + 1 < argc) {
            initial_balance = std::stod(argv[++i]);
            // Ensure balance is not unreasonably large
            if (initial_balance > 1000.0) {
                std::cout << "WARNING: Initial balance " << initial_balance << " is very large. Capping at 1000.0" << std::endl;
                initial_balance = 1000.0;
            }
        } else if (arg == "--stop-loss" && i + 1 < argc) {
            stop_loss_pct = std::stod(argv[++i]);
        } else if (arg == "--max-drawdown" && i + 1 < argc) {
            max_drawdown_pct = std::stod(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --symbol SYMBOL       Trading symbol (default: btcusdt)" << std::endl;
            std::cout << "  --exchange-url URL    Exchange WebSocket URL (default: wss://stream.binance.com:9443/ws)" << std::endl;
            std::cout << "  --duration SECONDS    Test duration in seconds (default: 60)" << std::endl;
            std::cout << "  --balance AMOUNT      Initial balance (default: 100.0)" << std::endl;
            std::cout << "  --stop-loss PCT       Stop loss percentage (default: 0.5%)" << std::endl;
            std::cout << "  --max-drawdown PCT    Max drawdown percentage (default: 5.0%)" << std::endl;
            std::cout << "  --help                Show this help message" << std::endl;
            return 0;
        }
    }
    
    std::cout << "Starting trading bot with:" << std::endl;
    std::cout << "  Symbol: " << symbol << std::endl;
    std::cout << "  Exchange URL: " << exchange_url << std::endl;
    std::cout << "  Duration: " << duration_seconds << " seconds" << std::endl;
    std::cout << "  Initial balance: " << initial_balance << std::endl;
    std::cout << "  Stop loss: " << stop_loss_pct << "%" << std::endl;
    std::cout << "  Max drawdown: " << max_drawdown_pct << "%" << std::endl;
    
    // Create the trading bot
    auto fetcher = std::make_shared<trading::OrderbookFetcher>(symbol);
    auto model = std::make_shared<trading::TradingModel>("model.pt");
    auto bot = std::make_shared<trading::TradingBot>(fetcher, model, initial_balance);
    
    // Connect to the exchange
    if (!fetcher->connect(exchange_url, symbol)) {
        std::cerr << "Failed to connect to exchange" << std::endl;
        return 1;
    }
    
    // Set risk management parameters
    bot->setStopLossPercentage(stop_loss_pct);
    bot->setMaxDrawdownPercentage(max_drawdown_pct);
    
    // Start the bot
    bot->start();
    
    std::cout << "Trading bot started" << std::endl;
    
    // Main loop to display metrics
    auto start_time = std::chrono::system_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration_seconds);
    
    // Static counter for detailed P&L metrics
    static int counter = 0;
    
    while (std::chrono::system_clock::now() < end_time) {
        // Get performance metrics
        auto metrics = bot->getPerformanceMetrics();
        
        // Print performance metrics
        std::cout << "Performance metrics:" << std::endl;
        std::cout << "  Avg tick-to-trade: " << metrics["avg_tick_to_trade"].get<uint64_t>() << " ns" << std::endl;
        std::cout << "  Trades per second: " << metrics["trades_per_second"].get<uint64_t>() << std::endl;
        std::cout << "  Total trades: " << metrics["total_trades"].get<uint64_t>() << std::endl;
        std::cout << "  Successful trades: " << metrics["successful_trades"].get<uint64_t>() << std::endl;
        std::cout << "  Success rate: " << metrics["success_rate"].get<double>() << "%" << std::endl;
        std::cout << "  Position: " << metrics["position"].get<double>() << std::endl;
        std::cout << "  Balance: " << metrics["balance"].get<double>() << std::endl;
        
        // Print USD volume metrics
        std::cout << "USD Volume metrics:" << std::endl;
        std::cout << "  Current interval volume: $" << metrics["interval_usd_volume"].get<double>() << std::endl;
        std::cout << "  Total USD volume: $" << metrics["total_usd_volume"].get<double>() << std::endl;
        std::cout << "  Max interval volume: $" << metrics["max_interval_usd_volume"].get<double>() << std::endl;
        std::cout << "  Avg interval volume: $" << metrics["avg_interval_usd_volume"].get<double>() << std::endl;
        
        // Print P&L metrics
        auto pnl_metrics = bot->getPnLMetrics();
        std::cout << "P&L metrics:" << std::endl;
        std::cout << "  Realized P&L: " << pnl_metrics["realized_pnl"].get<double>() << std::endl;
        std::cout << "  Unrealized P&L: " << pnl_metrics["unrealized_pnl"].get<double>() << std::endl;
        std::cout << "  Total P&L: " << pnl_metrics["total_pnl"].get<double>() << std::endl;
        
        // Print uncapped P&L metrics for analysis
        std::cout << "Uncapped P&L metrics (for analysis):" << std::endl;
        std::cout << "  Uncapped Unrealized P&L: " << pnl_metrics["uncapped_unrealized_pnl"].get<double>() << std::endl;
        std::cout << "  Uncapped Total P&L: " << pnl_metrics["uncapped_total_pnl"].get<double>() << std::endl;
        
        // Display detailed P&L metrics every 15 seconds
        if (++counter % 15 == 0) {
            std::cout << "Detailed P&L metrics:" << std::endl;
            std::cout << "  Win count: " << pnl_metrics["win_count"].get<double>() << std::endl;
            std::cout << "  Loss count: " << pnl_metrics["loss_count"].get<double>() << std::endl;
            std::cout << "  Win rate: " << pnl_metrics["win_rate"].get<double>() << "%" << std::endl;
            std::cout << "  Largest win: " << pnl_metrics["largest_win"].get<double>() << std::endl;
            std::cout << "  Largest loss: " << pnl_metrics["largest_loss"].get<double>() << std::endl;
            std::cout << "  Avg win: " << pnl_metrics["avg_win"].get<double>() << std::endl;
            std::cout << "  Avg loss: " << pnl_metrics["avg_loss"].get<double>() << std::endl;
            std::cout << "  Profit factor: " << pnl_metrics["profit_factor"].get<double>() << std::endl;
            std::cout << "  Avg trade duration: " << pnl_metrics["avg_trade_duration_ms"].get<double>() << " ms" << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "Stopping trading bot..." << std::endl;
    
    // Stop the bot
    bot->stop();
    
    std::cout << "Trading bot stopped" << std::endl;
    
    return 0;
} 