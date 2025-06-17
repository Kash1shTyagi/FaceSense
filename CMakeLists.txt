cmake_minimum_required(VERSION 3.15)
project(FaceEmotionRecognition LANGUAGES CXX)

# CMake settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configuration options
option(BUILD_TESTS "Build test executables" ON)
option(USE_OPENMP "Enable OpenMP parallelization" ON)
option(USE_AVX "Enable AVX optimizations" ON)
option(DEPLOY_MODE "Optimize for deployment" OFF)

# OpenCV dependency
find_package(OpenCV REQUIRED)

# Set deployment flags
if(DEPLOY_MODE)
    add_compile_definitions(DEPLOY_MODE)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")
endif()

# Optimizations
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(USE_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx -mfma")
endif()

# Main executable
add_executable(emotion_recognition src/main.cpp)
target_include_directories(emotion_recognition PUBLIC include)
target_link_libraries(emotion_recognition PRIVATE ${OpenCV_LIBS})

# Testing targets
if(BUILD_TESTS)
    enable_testing()
    
    # Matrix tests
    add_executable(test_matrix tests/test_matrix.cpp)
    target_include_directories(test_matrix PUBLIC include)
    add_test(NAME MatrixTests COMMAND test_matrix)
    
    # PCA tests
    add_executable(test_pca tests/test_pca.cpp)
    target_include_directories(test_pca PUBLIC include)
    target_link_libraries(test_pca PRIVATE ${OpenCV_LIBS})
    add_test(NAME PCATests COMMAND test_pca)
    
    # GMM tests
    add_executable(test_gmm tests/test_gmm.cpp)
    target_include_directories(test_gmm PUBLIC include)
    add_test(NAME GMMTests COMMAND test_gmm)
    
    # FFNN tests
    add_executable(test_ffnn tests/test_ffnn.cpp)
    target_include_directories(test_ffnn PUBLIC include)
    add_test(NAME FFNNTests COMMAND test_ffnn)
    
    # Integration test
    add_executable(test_integration tests/test_integration.cpp)
    target_include_directories(test_integration PUBLIC include)
    target_link_libraries(test_integration PRIVATE ${OpenCV_LIBS})
    add_test(NAME IntegrationTest COMMAND test_integration)
endif()

# Installation for deployment
install(TARGETS emotion_recognition DESTINATION bin)
install(DIRECTORY data/ DESTINATION share/face_emotion_cpp/data)