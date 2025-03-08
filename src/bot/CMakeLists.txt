add_library(bot
    TradingBot.cpp
    BinanceApiTrader.cpp
)

target_include_directories(bot
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Find required packages
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

target_link_libraries(bot
    PUBLIC
    orderbook_lib
    model_lib
    ${CURL_LIBRARIES}
    ${OPENSSL_LIBRARIES}
)

target_include_directories(bot
    PRIVATE
    ${CURL_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

target_compile_features(bot PRIVATE cxx_std_20) 