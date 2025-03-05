add_library(bot_lib
    TradingBot.cpp
)

target_include_directories(bot_lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bot_lib
    PRIVATE
    orderbook_lib
)

target_compile_features(bot_lib PRIVATE cxx_std_20) 