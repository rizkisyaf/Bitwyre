cmake_minimum_required(VERSION 3.20)
project(TradingBot VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include FetchContent for dependency management
include(FetchContent)

# Set FetchContent options
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# Find OpenSSL for TLS support
find_package(OpenSSL REQUIRED)

# Fetch dependencies
FetchContent_Declare(
    websocketpp
    GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
    GIT_TAG 0.8.2
)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)

FetchContent_Declare(
    asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG asio-1-28-0
)

# Set LibTorch path to virtual environment
set(LIBTORCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/venv/lib/python3.13/site-packages/torch")

# Make dependencies available
FetchContent_MakeAvailable(websocketpp)
FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(asio)

# Configure WebSocket++ to use standalone ASIO
add_definitions(-DASIO_STANDALONE)

# Add include directories
include_directories(
    ${websocketpp_SOURCE_DIR}
    ${json_SOURCE_DIR}/include
    ${asio_SOURCE_DIR}/asio/include
    ${LIBTORCH_PATH}/include
    ${LIBTORCH_PATH}/include/torch/csrc/api/include
    ${OPENSSL_INCLUDE_DIR}
    include
)

# Link directories for libtorch
link_directories(${LIBTORCH_PATH}/lib)

# Add subdirectories for our project components
add_subdirectory(src)

# Enable testing
enable_testing()
add_subdirectory(tests) 