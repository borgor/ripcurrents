cmake_minimum_required(VERSION 2.8)
project(RipCurrents)
find_package(OpenCV REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

add_executable( stabilize stabilize.cpp )
target_compile_features(stabilize PUBLIC cxx_lambdas)
target_link_libraries( stabilize ${OpenCV_LIBS} )
