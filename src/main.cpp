#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <csignal>
#include <nlohmann/json.hpp>
#include <atomic>

#include "orderbook/OrderbookFetcher.hpp"
#include "bot/TradingBot.hpp"
#include "model/TradingModel.hpp"
#include "bot/BinanceApiTrader.hpp"

// Global variables for signal handling
volatile sig_atomic_t g_running = 1;
volatile sig_atomic_t g_emergency_stop = 0;
std::shared_ptr<trading::TradingBot> g_bot = nullptr;

// Signal handler
void signalHandler(int signal) {
    static std::atomic<bool> handling_signal{false};
    static std::atomic<int> signal_count{0};
    
    // Prevent multiple signal handlers from running simultaneously
    if (handling_signal.exchange(true)) {
        signal_count++;
        if (signal_count >= 3) {
            std::cerr << "\n🚨 Forcing immediate termination!" << std::endl;
            std::quick_exit(1);
        }
        return;
    }
    
    try {
        if (signal == SIGINT) {
            std::cout << "\nReceived SIGINT, gracefully stopping..." << std::endl;
            g_running = 0;
            if (g_bot) {
                g_bot->stop();
            }
        } else if (signal == SIGTERM || signal == SIGUSR1) {
            std::cout << "\nReceived emergency stop signal" << std::endl;
            g_emergency_stop = 1;
            g_running = 0;
            
            if (g_bot) {
                // Try emergency stop with 5 second timeout
                if (!g_bot->emergencyStop(5000)) {
                    std::cerr << "Emergency stop timed out or failed, forcing exit..." << std::endl;
                    std::quick_exit(1);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in signal handler: " << e.what() << std::endl;
        std::quick_exit(1);
    }
    
    handling_signal = false;
}

// Function to test API connection
bool testApiConnection(const std::string& api_key, const std::string& secret_key, bool use_futures) {
    std::cout << "\n======= TESTING API CONNECTION ONLY =======" << std::endl;
    
    trading::BinanceApiTrader::TradingMode mode = 
        use_futures ? trading::BinanceApiTrader::TradingMode::USDM_FUTURES : 
                     trading::BinanceApiTrader::TradingMode::SPOT;
    
    std::cout << "Creating API trader with mode: " << (use_futures ? "USDM Futures" : "Spot") << std::endl;
    auto api_trader = std::make_shared<trading::BinanceApiTrader>(api_key, secret_key, mode);
    
    std::cout << "Attempting to initialize API connection..." << std::endl;
    bool success = api_trader->initialize();
    
    if (success) {
        std::cout << "✅ API connection test SUCCESSFUL!" << std::endl;
        
        // Try to get account info
        try {
            auto account_info = api_trader->getAccountInfo();
            
            // Check if the response contains an error code
            if (account_info.contains("code") && account_info["code"].is_number() && 
                account_info["code"].get<int>() < 0) {
                std::cerr << "❌ ERROR: " << account_info["msg"].get<std::string>() << 
                             " (code: " << account_info["code"].get<int>() << ")" << std::endl;
                success = false;
            } else {
                std::cout << "✅ Account info retrieved successfully" << std::endl;
                
                // Check trading permissions
                if (use_futures) {
                    if (account_info.contains("canTrade")) {
                        if (account_info["canTrade"].get<bool>()) {
                            std::cout << "✅ Futures trading is ENABLED for this account" << std::endl;
                        } else {
                            std::cout << "❌ Futures trading is NOT enabled for this account" << std::endl;
                            success = false;
                        }
                    }
                    
                    // Check if account has USDT balance
                    if (account_info.contains("assets")) {
                        bool found_usdt = false;
                        for (const auto& asset : account_info["assets"]) {
                            if (asset["asset"] == "USDT") {
                                double free_balance = std::stod(asset["availableBalance"].get<std::string>());
                                std::cout << "✅ USDT balance: " << free_balance << std::endl;
                                found_usdt = true;
                                break;
                            }
                        }
                        
                        if (!found_usdt) {
                            std::cout << "⚠️ WARNING: No USDT balance found in futures account" << std::endl;
                        }
                    }
                } else {
                    // Check spot trading permissions
                    if (account_info.contains("canTrade")) {
                        if (account_info["canTrade"].get<bool>()) {
                            std::cout << "✅ Spot trading is ENABLED for this account" << std::endl;
                        } else {
                            std::cout << "❌ Spot trading is NOT enabled for this account" << std::endl;
                            success = false;
                        }
                    }
                    
                    // Check if account has USDT balance
                    if (account_info.contains("balances")) {
                        bool found_usdt = false;
                        for (const auto& asset : account_info["balances"]) {
                            if (asset["asset"] == "USDT") {
                                double free_balance = std::stod(asset["free"].get<std::string>());
                                std::cout << "✅ USDT balance: " << free_balance << std::endl;
                                found_usdt = true;
                                break;
                            }
                        }
                        
                        if (!found_usdt) {
                            std::cout << "⚠️ WARNING: No USDT balance found in spot account" << std::endl;
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ ERROR: Failed to get account info: " << e.what() << std::endl;
            success = false;
        }
    } else {
        std::cerr << "❌ API connection test FAILED!" << std::endl;
    }
    
    std::cout << "\nSUMMARY: " << (success ? "✅ Connection test PASSED" : "❌ Connection test FAILED") << std::endl;
    std::cout << "=======================================" << std::endl;
    return success;
}

// Separate main function for test connection mode
int runTestConnection(const std::string& api_key, const std::string& secret_key, bool use_futures) {
    bool success = testApiConnection(api_key, secret_key, use_futures);
    return success ? 0 : 1;
}

// Main function for trading bot
int runTradingBot(int argc, char* argv[]) {
    // Register signal handlers
    std::signal(SIGINT, signalHandler);   // Ctrl+C
    std::signal(SIGTERM, signalHandler);  // Termination signal
    std::signal(SIGUSR1, signalHandler);  // User-defined signal
    
    // Parse command line arguments
    std::string symbol = "btcusdt";
    std::string exchange_url = "wss://stream.binance.com:9443/ws";
    int duration_seconds = 60;
    double initial_balance = 100.0;  // Default initial balance
    double stop_loss_pct = 0.5;        // Default stop loss percentage
    double max_drawdown_pct = 5.0;     // Default max drawdown percentage
    std::string api_key = "";          // Default empty API key
    std::string secret_key = "";       // Default empty secret key
    bool use_futures = true;           // Default to USDM futures
    int leverage = 5;                  // Default leverage for futures
    bool isolated_margin = true;       // Default to isolated margin
    
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
        } else if (arg == "--api-key" && i + 1 < argc) {
            api_key = argv[++i];
        } else if (arg == "--secret-key" && i + 1 < argc) {
            secret_key = argv[++i];
        } else if (arg == "--spot") {
            use_futures = false;
        } else if (arg == "--leverage" && i + 1 < argc) {
            leverage = std::stoi(argv[++i]);
            if (leverage < 1 || leverage > 125) {
                std::cout << "WARNING: Leverage must be between 1 and 125. Setting to default (5)." << std::endl;
                leverage = 5;
            }
        } else if (arg == "--cross-margin") {
            isolated_margin = false;
        }
    }
    
    // Check if API keys are missing
    if (api_key.empty() || secret_key.empty()) {
        std::cerr << "Error: Trading requires both API key and secret key" << std::endl;
        return 1;
    }
    
    // From here on, we're running the actual trading bot
    std::cout << "\n======= STARTING TRADING BOT =======" << std::endl;
    std::cout << "Starting trading bot with:" << std::endl;
    std::cout << "  Symbol: " << symbol << std::endl;
    std::cout << "  Exchange URL: " << exchange_url << std::endl;
    std::cout << "  Duration: " << duration_seconds << " seconds" << std::endl;
    std::cout << "  Initial balance: " << initial_balance << std::endl;
    std::cout << "  Stop loss: " << stop_loss_pct << "%" << std::endl;
    std::cout << "  Max drawdown: " << max_drawdown_pct << "%" << std::endl;
    std::cout << "  Market type: " << (use_futures ? "USDM Futures" : "Spot") << std::endl;
    
    if (use_futures) {
        std::cout << "  Leverage: " << leverage << "x" << std::endl;
        std::cout << "  Margin type: " << (isolated_margin ? "Isolated" : "Cross") << std::endl;
    }
    
    // Create the trading bot
    auto fetcher = std::make_shared<trading::OrderbookFetcher>(symbol);
    auto model = std::make_shared<trading::TradingModel>("model_fixed.pt", "mean.npy", "std.npy");
    
    // Create the bot with API keys
    std::shared_ptr<trading::TradingBot> bot;
    
    // Set the trading mode (spot or futures)
    trading::BinanceApiTrader::TradingMode trading_mode = 
        use_futures ? trading::BinanceApiTrader::TradingMode::USDM_FUTURES : 
                     trading::BinanceApiTrader::TradingMode::SPOT;
    
    bot = std::make_shared<trading::TradingBot>(fetcher, model, initial_balance, api_key, secret_key);
    
    // If using futures, set leverage and margin type
    if (use_futures) {
        auto api_trader = std::make_shared<trading::BinanceApiTrader>(api_key, secret_key, trading_mode);
        if (api_trader->initialize()) {
            std::cout << "Setting leverage to " << leverage << "x for " << symbol << std::endl;
            if (api_trader->setLeverage(symbol, leverage)) {
                std::cout << "✅ Leverage set successfully" << std::endl;
            } else {
                std::cerr << "❌ Failed to set leverage" << std::endl;
            }
            
            std::cout << "Setting margin type to " << (isolated_margin ? "Isolated" : "Cross") << " for " << symbol << std::endl;
            if (api_trader->setMarginType(symbol, isolated_margin)) {
                std::cout << "✅ Margin type set successfully" << std::endl;
            } else {
                std::cerr << "❌ Failed to set margin type" << std::endl;
            }
        } else {
            std::cerr << "❌ Failed to initialize API trader for leverage/margin configuration" << std::endl;
        }
    }
    
    // Store bot in global variable for signal handler
    g_bot = bot;
    
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
    
    std::cout << "Trading bot started. Press Ctrl+C to stop normally, or send SIGUSR1 signal for emergency stop." << std::endl;
    std::cout << "To cancel all orders from another terminal: kill -SIGUSR1 " << getpid() << std::endl;
    
    // Main loop to display metrics
    auto start_time = std::chrono::system_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration_seconds);
    
    // Reduce logging frequency for less clutter
    int log_interval = 10; // Log every 10 iterations
    int log_counter = 0;
    
    while (std::chrono::system_clock::now() < end_time && g_running) {
        // Get P&L metrics
        auto pnl_metrics = bot->getPnLMetrics();
        
        // Only log periodically to reduce clutter
        if (++log_counter >= log_interval) {
            log_counter = 0;
            
            // Print a more concise summary
            std::cout << "\n--- TRADING STATUS UPDATE ---" << std::endl;
            std::cout << "Position: " << bot->getPosition() 
                      << " | Balance: " << bot->getBalance() << std::endl;
            
            std::cout << "P&L: Realized=" << pnl_metrics["realized_pnl"].get<double>() 
                      << " | Unrealized=" << pnl_metrics["unrealized_pnl"].get<double>() 
                      << " | Total=" << pnl_metrics["total_pnl"].get<double>() << std::endl;
            
            // Display detailed P&L metrics
            std::cout << "Detailed P&L: Win/Loss=" << pnl_metrics["win_count"].get<double>() << "/" 
                      << pnl_metrics["loss_count"].get<double>() 
                      << " | Win Rate=" << pnl_metrics["win_rate"].get<double>() << "%" 
                      << " | Profit Factor=" << pnl_metrics["profit_factor"].get<double>() << std::endl;
        }
        
        // Check for emergency stop
        if (g_emergency_stop) {
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    if (!g_emergency_stop) {
        std::cout << "\nStopping trading bot..." << std::endl;
        
        // Stop the bot
        bot->stop();
        
        std::cout << "Trading bot stopped" << std::endl;
    }
    
    // Clear global pointer
    g_bot = nullptr;
    
    return 0;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments for test connection mode
    std::string api_key = "";
    std::string secret_key = "";
    bool use_futures = true;
    bool test_connection_only = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--api-key" && i + 1 < argc) {
            api_key = argv[++i];
        } else if (arg == "--secret-key" && i + 1 < argc) {
            secret_key = argv[++i];
        } else if (arg == "--spot") {
            use_futures = false;
        } else if (arg == "--test-connection") {
            test_connection_only = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --symbol SYMBOL       Trading symbol (default: btcusdt)" << std::endl;
            std::cout << "  --exchange-url URL    Exchange WebSocket URL (default: wss://stream.binance.com:9443/ws)" << std::endl;
            std::cout << "  --duration SECONDS    Test duration in seconds (default: 60)" << std::endl;
            std::cout << "  --balance AMOUNT      Initial balance (default: 100.0)" << std::endl;
            std::cout << "  --stop-loss PCT       Stop loss percentage (default: 0.5%)" << std::endl;
            std::cout << "  --max-drawdown PCT    Max drawdown percentage (default: 5.0%)" << std::endl;
            std::cout << "  --api-key KEY         Binance API key for real trading" << std::endl;
            std::cout << "  --secret-key KEY      Binance secret key for real trading" << std::endl;
            std::cout << "  --spot                Use spot trading instead of USDM futures" << std::endl;
            std::cout << "  --leverage NUM        Set leverage for futures trading (1-125, default: 5)" << std::endl;
            std::cout << "  --cross-margin        Use cross margin instead of isolated margin for futures" << std::endl;
            std::cout << "  --test-connection     Only test API connection and exit" << std::endl;
            std::cout << "  --help                Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Check if test connection mode is requested
    if (test_connection_only) {
        // Check if API keys are provided
        if (api_key.empty() || secret_key.empty()) {
            std::cerr << "Error: API connection testing requires both API key and secret key" << std::endl;
            return 1;
        }
        
        // Run in test connection mode
        return runTestConnection(api_key, secret_key, use_futures);
    }
    
    // Run in trading bot mode
    return runTradingBot(argc, argv);
} 