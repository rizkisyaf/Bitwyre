add_library(model_lib
    TradingModel.cpp
)

target_include_directories(model_lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${libtorch_SOURCE_DIR}/include
    ${libtorch_SOURCE_DIR}/include/torch/csrc/api/include
)

target_link_libraries(model_lib
    PRIVATE
    ${TORCH_LIBRARIES}
)

target_compile_features(model_lib PRIVATE cxx_std_20) 