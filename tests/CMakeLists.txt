FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

add_executable(orderbook_tests OrderbookTests.cpp)
target_link_libraries(orderbook_tests PRIVATE orderbook_lib gtest_main)
add_test(NAME orderbook_tests COMMAND orderbook_tests)

add_executable(bot_tests BotTests.cpp)
target_link_libraries(bot_tests PRIVATE bot_lib model_lib orderbook_lib ${TORCH_LIBRARIES} gtest_main)
add_test(NAME bot_tests COMMAND bot_tests)

add_executable(model_tests ModelTests.cpp)
target_link_libraries(model_tests PRIVATE model_lib orderbook_lib ${TORCH_LIBRARIES} gtest_main)
add_test(NAME model_tests COMMAND model_tests) 