# install header
file(GLOB ex_headers "*.h")
#install(FILES ${ex_headers} DESTINATION include/StiefelManifoldExample)

# Add unit tests
gtsamAddTestsGlob(tests "tests/*.cpp" "" ${PROJECT_NAME})

# Add executables
gtsamAddExamplesGlob("examples/*.cpp" "" "${PROJECT_NAME}" ON)

file(GLOB ex_srcs "*.cpp")

# export cmake setting to the source code
add_library(${PROJECT_NAME} SHARED ${ex_srcs} ${ex_headers}
)

# linking with dependences
target_link_libraries(${PROJECT_NAME} gtsam Optimization ILDL ${BLAS_LIBRARIES} ${SPQR_LIBRARIES} ${LAPACK} cholmod)