cmake_minimum_required(VERSION 2.8)
project(RipCurrents)
find_package(OpenCV REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

add_executable( ripcurrents ripcurrents.hpp main.cpp ripcurrents_module.cpp Streakline.hpp Streakline.cpp)
target_compile_features(ripcurrents PUBLIC cxx_lambdas)
target_link_libraries( ripcurrents ${OpenCV_LIBS} )

