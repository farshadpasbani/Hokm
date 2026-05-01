cmake_minimum_required(VERSION 3.14)
project(HokmCPP VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directories
include_directories(include)

# Find LibTorch (PyTorch C++ API)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Core library
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
if(CORE_SOURCES)
    add_library(HokmCore ${CORE_SOURCES})
else()
    # Dummy to allow cmake to pass initially
    add_library(HokmCore INTERFACE)
endif()

file(GLOB_RECURSE RL_SOURCES "src/rl/*.cpp")
if(RL_SOURCES)
    add_library(HokmRL ${RL_SOURCES})
    target_link_libraries(HokmRL PRIVATE HokmCore "${TORCH_LIBRARIES}")
    target_include_directories(HokmRL PRIVATE include)
endif()

add_executable(HokmApp src/app.cpp)
target_link_libraries(HokmApp PRIVATE HokmCore HokmRL "${TORCH_LIBRARIES}")
target_include_directories(HokmApp PRIVATE include)

add_executable(TrainHokm src/train_hokm.cpp)
target_link_libraries(TrainHokm PRIVATE HokmCore HokmRL "${TORCH_LIBRARIES}")
target_include_directories(TrainHokm PRIVATE include)

if(APPLE)
  set_property(TARGET TrainHokm PROPERTY BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endif()

# Tests
enable_testing()

file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
if(TEST_SOURCES)
    add_executable(HokmTests ${TEST_SOURCES})
    target_link_libraries(HokmTests gtest_main HokmCore)
    include(GoogleTest)
    gtest_discover_tests(HokmTests)
endif()
