# Set torch libraries directly with full paths
set(TORCH_LIBRARY "${LIBTORCH_PATH}/lib/libtorch.dylib")
set(TORCH_CPU_LIBRARY "${LIBTORCH_PATH}/lib/libtorch_cpu.dylib")
set(C10_LIBRARY "${LIBTORCH_PATH}/lib/libc10.dylib")

# Set torch libraries
set(TORCH_LIBRARIES ${TORCH_LIBRARY} ${TORCH_CPU_LIBRARY} ${C10_LIBRARY})

# Add subdirectories
add_subdirectory(orderbook)
add_subdirectory(model)
add_subdirectory(bot)

# Find required packages
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

# Add the main executable
add_executable(trading_bot main.cpp)

# Add collect_orderbook_data executable
add_executable(collect_orderbook_data collect_orderbook_data.cpp)
target_link_libraries(collect_orderbook_data
    PRIVATE
    orderbook_lib
    nlohmann_json::nlohmann_json
)
target_compile_features(collect_orderbook_data PRIVATE cxx_std_20)

# Link libraries
target_link_libraries(trading_bot
    PRIVATE
    orderbook_lib
    model_lib
    bot
    ${CURL_LIBRARIES}
    ${OPENSSL_LIBRARIES}
    ${TORCH_LIBRARIES}
)

# Include directories
target_include_directories(trading_bot
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CURL_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

# Set C++ standard
target_compile_features(trading_bot PRIVATE cxx_std_20) 