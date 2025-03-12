#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <csignal>
#include <iomanip>
#include <map>
#include <mutex>
#include <limits.h>  // For PATH_MAX
#include "orderbook/OrderbookFetcher.hpp"
#include <nlohmann/json.hpp>
#include <sys/stat.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>  // For _NSGetExecutablePath
#elif defined(_WIN32)
#include <windows.h>  // For GetModuleFileName
#endif

using json = nlohmann::json;

// Global variables for signal handling
volatile sig_atomic_t g_running = 1;

// Structure to store trade data
struct TradeData {
    int64_t timestamp;
    double price;
    double quantity;
    bool is_buyer_maker;
};

// Global variables for trade data
std::map<int64_t, TradeData> g_trades;
std::mutex g_trades_mutex;
int64_t g_last_trade_timestamp = 0;

// Signal handler
void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\nReceived signal " << signal << ", stopping data collection..." << std::endl;
        g_running = 0;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <symbol> <duration_minutes> <output_file>" << std::endl;
        return 1;
    }

    // Parse command line arguments
    std::string symbol = argv[1];
    int duration_minutes = std::stoi(argv[2]);
    std::string output_file = argv[3];
    
    // Ensure output file is in the data folder if not an absolute path
    if (output_file.find('/') == std::string::npos) {
        // Get the executable path
        char exec_path[PATH_MAX];
        #ifdef __APPLE__
            uint32_t size = sizeof(exec_path);
            if (_NSGetExecutablePath(exec_path, &size) != 0) {
                std::cerr << "Failed to get executable path" << std::endl;
                return 1;
            }
        #elif defined(_WIN32)
            GetModuleFileName(NULL, exec_path, PATH_MAX);
        #else
            ssize_t count = readlink("/proc/self/exe", exec_path, PATH_MAX);
            if (count == -1) {
                std::cerr << "Failed to get executable path" << std::endl;
                return 1;
            }
            exec_path[count] = '\0';
        #endif
        
        // Get the directory of the executable
        std::string exec_dir = exec_path;
        size_t last_slash = exec_dir.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            exec_dir = exec_dir.substr(0, last_slash);
        }
        
        // Navigate to project root (assuming executable is in build/src)
        std::string project_root = exec_dir;
        if (project_root.find("/build/") != std::string::npos) {
            project_root = project_root.substr(0, project_root.find("/build/"));
        } else if (project_root.find("/src/") != std::string::npos) {
            project_root = project_root.substr(0, project_root.find("/src/"));
        }
        
        // Create data directory if it doesn't exist
        std::string data_dir = project_root + "/data";
        struct stat info;
        if (stat(data_dir.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
            // Directory doesn't exist, create it
            #ifdef _WIN32
                int result = mkdir(data_dir.c_str());
            #else
                int result = mkdir(data_dir.c_str(), 0755);
            #endif
            if (result != 0) {
                std::cerr << "Failed to create data directory: " << data_dir << std::endl;
                return 1;
            }
        }
        
        output_file = data_dir + "/" + output_file;
    }
    
    std::cout << "Data will be saved to: " << output_file << std::endl;

    // Register signal handler
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Create orderbook fetcher
    auto fetcher = std::make_shared<trading::OrderbookFetcher>(symbol);

    // Open output file
    std::ofstream csv_file(output_file);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return 1;
    }

    // Write CSV header
    csv_file << "timestamp,";
    // Add bid columns (10 levels)
    for (int i = 1; i <= 10; i++) {
        csv_file << "bid_price" << i << ",bid_qty" << i << ",";
    }
    // Add ask columns (10 levels)
    for (int i = 1; i <= 10; i++) {
        csv_file << "ask_price" << i << ",ask_qty" << i;
        if (i < 10) csv_file << ",";
    }
    // Add trade data columns
    csv_file << ",taker_buy_base_volume,taker_sell_base_volume,trade_count,avg_trade_price";
    csv_file << std::endl;

    // Connect to Binance Futures WebSocket (for perpetual contracts)
    std::string ws_url = "wss://fstream.binance.com/ws";
    if (!fetcher->connect(ws_url, symbol)) {
        std::cerr << "Failed to connect to WebSocket" << std::endl;
        return 1;
    }

    std::cout << "Connected to Binance WebSocket for " << symbol << std::endl;
    
    // Wait a short time to ensure connection is fully established
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Subscribe to trade stream
    std::string trade_stream = symbol.substr(0, symbol.find_first_of("@/")) + "@aggTrade";
    std::cout << "Subscribing to trade stream: " << trade_stream << std::endl;
    
    if (!fetcher->subscribeToStream(trade_stream, [&symbol](const std::string& data) {
        try {
            // Parse JSON data properly using nlohmann/json
            json trade_data = json::parse(data);
            
            // Extract data using proper JSON parsing
            if (trade_data.contains("e") && trade_data["e"] == "aggTrade") {
                int64_t timestamp = trade_data["T"];
                double price = std::stod(trade_data["p"].get<std::string>());
                double quantity = std::stod(trade_data["q"].get<std::string>());
                bool is_buyer_maker = trade_data["m"];
                
                // Store trade data
                std::lock_guard<std::mutex> lock(g_trades_mutex);
                TradeData trade = {timestamp, price, quantity, is_buyer_maker};
                g_trades[timestamp] = trade;
                g_last_trade_timestamp = std::max(g_last_trade_timestamp, timestamp);
                
                // Log first few trades for debugging
                static int trade_debug_count = 0;
                if (trade_debug_count < 5) {
                    std::cout << "Trade received - Symbol: " << symbol 
                              << ", Price: " << price 
                              << ", Quantity: " << quantity 
                              << ", Buyer is maker: " << (is_buyer_maker ? "true" : "false") << std::endl;
                    trade_debug_count++;
                }
            } else {
                // Log unexpected message format
                static int unexpected_count = 0;
                if (unexpected_count < 3) {
                    std::cerr << "Unexpected trade message format: " << data.substr(0, 100) << "..." << std::endl;
                    unexpected_count++;
                }
            }
        } catch (const json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            std::cerr << "Raw data (first 100 chars): " << data.substr(0, 100) << "..." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error processing trade data: " << e.what() << std::endl;
        }
    })) {
        std::cerr << "Failed to subscribe to trade stream: " << trade_stream << std::endl;
        fetcher->disconnect();
        return 1;
    }

    std::cout << "Subscribed to trade stream for " << symbol << std::endl;
    std::cout << "Collecting data for " << duration_minutes << " minutes..." << std::endl;

    // Register callback for orderbook updates
    fetcher->registerCallback([&csv_file](const trading::Orderbook& orderbook) {
        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        // Get bids and asks
        auto bids = orderbook.getBids();
        auto asks = orderbook.getAsks();

        // Calculate trade statistics for the last interval (e.g., 1 second)
        double taker_buy_volume = 0.0;
        double taker_sell_volume = 0.0;
        int trade_count = 0;
        double total_trade_price = 0.0;
        
        {
            std::lock_guard<std::mutex> lock(g_trades_mutex);
            int64_t cutoff_time = timestamp - 1000; // Last 1 second
            
            // Process trades in the last interval
            for (const auto& trade_pair : g_trades) {
                const auto& trade = trade_pair.second;
                if (trade.timestamp >= cutoff_time && trade.timestamp <= timestamp) {
                    if (trade.is_buyer_maker) {
                        // If buyer is maker, seller is taker (market sell)
                        taker_sell_volume += trade.quantity;
                    } else {
                        // If buyer is taker (market buy)
                        taker_buy_volume += trade.quantity;
                    }
                    trade_count++;
                    total_trade_price += trade.price;
                }
            }
            
            // Clean up old trades (keep only last 10 seconds)
            int64_t cleanup_cutoff = timestamp - 10000;
            auto it = g_trades.begin();
            while (it != g_trades.end() && it->first < cleanup_cutoff) {
                it = g_trades.erase(it);
            }
            
            // Log trade statistics periodically (every 10 seconds)
            static int64_t last_log_time = 0;
            if (timestamp - last_log_time > 10000) {
                std::cout << "Trade stats - Count: " << trade_count 
                          << ", Buy volume: " << taker_buy_volume 
                          << ", Sell volume: " << taker_sell_volume 
                          << ", Trades in memory: " << g_trades.size() << std::endl;
                last_log_time = timestamp;
            }
        }
        
        double avg_trade_price = (trade_count > 0) ? (total_trade_price / trade_count) : 0.0;

        // Write row to CSV
        csv_file << timestamp << ",";

        // Write bid data (10 levels)
        for (size_t i = 0; i < 10; i++) {
            if (i < bids.size()) {
                csv_file << std::fixed << std::setprecision(8) 
                        << bids[i].price << "," << bids[i].quantity;
            } else {
                csv_file << "0.0,0.0";
            }
            csv_file << ",";
        }

        // Write ask data (10 levels)
        for (size_t i = 0; i < 10; i++) {
            if (i < asks.size()) {
                csv_file << std::fixed << std::setprecision(8) 
                        << asks[i].price << "," << asks[i].quantity;
            } else {
                csv_file << "0.0,0.0";
            }
            if (i < 9) csv_file << ",";
        }
        
        // Write trade data
        csv_file << "," << std::fixed << std::setprecision(8)
                << taker_buy_volume << ","
                << taker_sell_volume << ","
                << trade_count << ","
                << avg_trade_price;
                
        csv_file << std::endl;
        csv_file.flush(); // Ensure data is written to disk immediately
    });

    // Wait for specified duration
    auto start_time = std::chrono::system_clock::now();
    while (g_running) {
        auto current_time = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(
            current_time - start_time).count();

        if (elapsed >= duration_minutes) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Cleanup
    fetcher->disconnect();
    csv_file.close();

    std::cout << "\nData collection complete. Data saved to: " << output_file << std::endl;
    return 0;
} 