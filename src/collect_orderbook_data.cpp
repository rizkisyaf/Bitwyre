#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <csignal>
#include <iomanip>
#include "orderbook/OrderbookFetcher.hpp"

// Global variables for signal handling
volatile sig_atomic_t g_running = 1;

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
    csv_file << std::endl;

    // Connect to Binance WebSocket
    std::string ws_url = "wss://stream.binance.com:9443/ws";
    if (!fetcher->connect(ws_url, symbol)) {
        std::cerr << "Failed to connect to WebSocket" << std::endl;
        return 1;
    }

    std::cout << "Connected to Binance WebSocket for " << symbol << std::endl;
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
        csv_file << std::endl;
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