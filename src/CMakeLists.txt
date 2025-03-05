# Set torch libraries directly with full paths
set(TORCH_LIBRARY "${LIBTORCH_PATH}/lib/libtorch.dylib")
set(TORCH_CPU_LIBRARY "${LIBTORCH_PATH}/lib/libtorch_cpu.dylib")
set(C10_LIBRARY "${LIBTORCH_PATH}/lib/libc10.dylib")

# Set torch libraries
set(TORCH_LIBRARIES ${TORCH_LIBRARY} ${TORCH_CPU_LIBRARY} ${C10_LIBRARY})

add_subdirectory(orderbook)
add_subdirectory(bot)
add_subdirectory(model)

# Main executable
add_executable(trading_bot main.cpp)
target_link_libraries(trading_bot 
    PRIVATE 
    orderbook_lib
    bot_lib
    model_lib
    ${TORCH_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
) 