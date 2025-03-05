add_library(orderbook_lib
    OrderbookFetcher.cpp
)

target_include_directories(orderbook_lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(orderbook_lib
    PRIVATE
    OpenSSL::SSL
    OpenSSL::Crypto
)

target_compile_features(orderbook_lib PRIVATE cxx_std_20) 