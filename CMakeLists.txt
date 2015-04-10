project(CompressiveTracking)

cmake_minimum_required(VERSION 2.6)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH bin)

INCLUDE_DIRECTORIES (${OpenCV_INCLUDE_DIRS})

add_executable(main
	main.cpp
	ObstacleAvoid.cpp ObstacleAvoid.h KeyPointHistory.cpp KeyPointHistory.h)

target_link_libraries(main ${OpenCV_LIBS})
