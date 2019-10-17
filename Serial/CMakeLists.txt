cmake_minimum_required(VERSION 3.12)
project(shallow_water)

set(CMAKE_CXX_STANDARD 14)
set(GCC_OPTIMIZATION_FLAGS -O3)
add_definitions(${GCC_OPTIMIZATION_FLAGS})
add_executable(shallow_water shallow_water.cpp helpers.cpp)
