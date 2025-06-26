cmake_minimum_required(VERSION 3.15)
project(face_emotion_cpp
    VERSION 1.0
    DESCRIPTION "Real-Time Face Feature Extraction & Emotion Classification in C++"
    LANGUAGES CXX
)

# ──────────────────────────────────────────────────────────────────────────────
# 1) GLOBAL SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BUILD_TESTS "Enable building of unit tests" ON)

# ──────────────────────────────────────────────────────────────────────────────
# 2) THIRD-PARTY DEPENDENCIES
# ──────────────────────────────────────────────────────────────────────────────
# 2a) GoogleTest
if(BUILD_TESTS)
  add_subdirectory(third_party/googletest)
endif()

# 2b) OpenCV
# We only care about a few core modules: core, imgproc, objdetect, highgui
set(BUILD_LIST core imgproc objdetect highgui)
# Turn off the parts of OpenCV you don't need (docs, tests, samples)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/opencv)   # registers OpenCV::opencv_core, etc.

# ──────────────────────────────────────────────────────────────────────────────
# 3) CORE LIBRARY
# ──────────────────────────────────────────────────────────────────────────────
add_library(face_core
    src/Matrix.cpp
    src/PCA.cpp
    src/GMM.cpp
    src/FFNN.cpp
    src/Utils.cpp
)

target_include_directories(face_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(face_core
  PUBLIC
    opencv_core
    opencv_imgproc
    opencv_objdetect
)

# ──────────────────────────────────────────────────────────────────────────────
# 4) APPLICATION
# ──────────────────────────────────────────────────────────────────────────────
add_executable(face_emotion_app src/main.cpp)

target_link_libraries(face_emotion_app
  PRIVATE
    face_core
    opencv_highgui   # for imshow, waitKey
)

set_target_properties(face_emotion_app PROPERTIES
  OUTPUT_NAME face_emotion
)

# ──────────────────────────────────────────────────────────────────────────────
# 5) INSTALLATION & EXPORT
# ──────────────────────────────────────────────────────────────────────────────
# ... your existing install rules ...

# ──────────────────────────────────────────────────────────────────────────────
# 6) TESTS
# ──────────────────────────────────────────────────────────────────────────────
if(BUILD_TESTS)
  enable_testing()

  # Matrix tests
  add_executable(test_matrix tests/test_matrix.cpp)
  target_link_libraries(test_matrix PRIVATE face_core gtest_main)
  add_test(NAME matrix_tests COMMAND test_matrix)

  # PCA tests
  add_executable(test_pca tests/test_pca.cpp)
  target_link_libraries(test_pca PRIVATE face_core gtest_main)
  add_test(NAME pca_tests COMMAND test_pca)

  # GMM tests
  add_executable(test_gmm tests/test_gmm.cpp)
  target_link_libraries(test_gmm PRIVATE face_core gtest_main)
  add_test(NAME gmm_tests COMMAND test_gmm)

  # FFNN tests
  add_executable(test_ffnn tests/test_ffnn.cpp)
  target_link_libraries(test_ffnn PRIVATE face_core gtest_main)
  add_test(NAME ffnn_tests COMMAND test_ffnn)
endif()
