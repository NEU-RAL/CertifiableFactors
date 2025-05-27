cmake_minimum_required(VERSION 3.9)
project(StiefelManifoldExample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall -Wextra -Wno-error")

# Set global output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(/usr/include/suitesparse)

# ========== Find Eigen ==========
find_package(Eigen3 3.3.7 REQUIRED)
if(EIGEN3_FOUND)
    message(STATUS "Found Eigen3 library: ${EIGEN3_INCLUDE_DIR}")
endif()

# ========== Find GTSAM ==========
find_package(GTSAM REQUIRED) 
if(GTSAM_FOUND)
    message(STATUS "Found GTSAM library: ${GTSAM_INCLUDE_DIR}\n")
endif()
include_directories(${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBRARIES gtsam)  

# ========== Find GTSAMCMakeTools ==========
find_package(GTSAMCMakeTools)
if(GTSAMCMakeTools_FOUND)
    message(STATUS "Found GTSAMCMakeTools library: ${GTSAMCMakeTools_DIR}\n")
endif()

# We are using some MACRO OF GTSAM, like gtsamAddTestsGlob and gtsamAddExamplesGlob.
# There we need to include(GtsamBuildTypes), where will define the compklier flag like -Werror.
# For now, we just override GTSAM's default compile flags to avoid Werror.
# But will fix the complier warning in the coming weeks. 
set(GTSAM_COMPILE_OPTIONS_PRIVATE_COMMON
    -Wall -Wextra -Wpedantic -Wno-error -Wno-unused-parameter
    CACHE STRING "No Werror" FORCE)

set(GTSAM_COMPILE_OPTIONS_PRIVATE_RELEASE
    -O2
    CACHE STRING "Safe optimization" FORCE)

set(GTSAM_COMPILE_OPTIONS_PUBLIC "" CACHE STRING "Public flags clean" FORCE)
set(GTSAM_BUILD_WITH_MARCH_NATIVE OFF CACHE BOOL "Disable native arch" FORCE)

include(GtsamMakeConfigFile)
include(GtsamBuildTypes)
include(GtsamTesting)

# for unittest scripts
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${GTSAM_DIR}/../GTSAMCMakeTools")

# ========== Process source subdirs ==========
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
