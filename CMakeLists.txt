cmake_minimum_required(VERSION 3.9)
project(StiefelManifoldExample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

include_directories(/usr/include/suitesparse)

find_package(Eigen3 3.3.7 REQUIRED)
if(EIGEN3_FOUND)
    message(STATUS "Found Eigen3 library: ${EIGEN3_INCLUDE_DIR}")
endif()



# Find GTSAM components
set(GTSAM_DIR /home/nikolas/gtsam/cmake-build-release)
set(GTSAM_UNSTABLE_DIR /home/nikolas/gtsam/cmake-build-release)
#set(GTSAM_DIR /home/jason/DPGO/gtsam_personal/cmake-build-release)
#set(GTSAM_UNSTABLE_DIR /home/jason/DPGO/gtsam_personal/cmake-build-release)
find_package(GTSAM REQUIRED) # Uses installed package
if(GTSAM_FOUND)
    message(STATUS "Found GTSAM library: ${GTSAM_INCLUDE_DIR}\n")
endif()
include_directories(${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBRARIES gtsam)   # TODO: automatic search libs

find_package(GTSAMCMakeTools)
if(GTSAMCMakeTools_FOUND)
    message(STATUS "Found GTSAMCMakeTools library: ${GTSAMCMakeTools_DIR}\n")
endif()

include(GtsamMakeConfigFile)
include(GtsamBuildTypes)
include(GtsamTesting)

# for unittest scripts
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${GTSAM_DIR}/../GTSAMCMakeTools")

# Process source subdirs
add_subdirectory(cpp)


#From fast verifications
# Find Optimization library
set(OPTIMIZATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Optimization/ CACHE PATH "Path to top-level directory of Optimization library (the one containing CMakeLists.txt)")

# Run the Optimization package's CMake file
add_subdirectory(${OPTIMIZATION_DIR})

# Find Preconditioners library
set(PRECONDITIONERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Preconditioners/ CACHE PATH "Path to top-level directory of Preconditioners library (the one containing CMakeLists.txt)")

# Run the Preconditioners package's CMake file
add_subdirectory(${PRECONDITIONERS_DIR})