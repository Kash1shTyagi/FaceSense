cmake_minimum_required(VERSION 3.15)
project(face_emotion_cpp
    VERSION 1.0
    DESCRIPTION "Real‐Time Face Feature Extraction & Emotion Classification in C++"
    LANGUAGES CXX
)

# ──────────────────────────────────────────────────────────────────────────────
# 1) GLOBAL SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BUILD_TESTS "Enable building of unit tests" ON)

# Make install prefix relocatable-friendly
if(NOT CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  # user-provided, leave as is
else()
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "" FORCE)
endif()

# ──────────────────────────────────────────────────────────────────────────────
# 2) DEPENDENCIES
# ──────────────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

if(BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
  )
  FetchContent_MakeAvailable(googletest)
endif()

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
      # Project headers
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      # OpenCV headers
      ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(face_core
    PUBLIC
      ${OpenCV_LIBS}        # legacy OpenCV variables
      # or for modern CMake if OpenCV provides imported targets:
      # OpenCV::opencv_core OpenCV::opencv_imgproc OpenCV::opencv_objdetect …
)
target_compile_features(face_core PUBLIC cxx_std_17)
set_target_properties(face_core PROPERTIES EXPORT_NAME core)

# ──────────────────────────────────────────────────────────────────────────────
# 4) APPLICATION
# ──────────────────────────────────────────────────────────────────────────────
add_executable(face_emotion_app
    src/main.cpp
)
target_link_libraries(face_emotion_app
    PRIVATE
      face_core
)
set_target_properties(face_emotion_app PROPERTIES OUTPUT_NAME face_emotion)

# ──────────────────────────────────────────────────────────────────────────────
# 5) INSTALLATION & EXPORT
# ──────────────────────────────────────────────────────────────────────────────
install(TARGETS face_core face_emotion_app
    EXPORT face_emotion_cppTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

# Generate a Config file for find_package()
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/face_emotion_cppConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/face_emotion_cppConfig.cmake"
  INSTALL_DESTINATION lib/cmake/face_emotion_cpp
)
install(EXPORT face_emotion_cppTargets
    FILE face_emotion_cppTargets.cmake
    NAMESPACE face_emotion_cpp::
    DESTINATION lib/cmake/face_emotion_cpp
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/face_emotion_cppConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/face_emotion_cppConfigVersion.cmake"
  DESTINATION lib/cmake/face_emotion_cpp
)

# ──────────────────────────────────────────────────────────────────────────────
# 6) TESTS
# ──────────────────────────────────────────────────────────────────────────────
if(BUILD_TESTS)
  enable_testing()

  add_executable(test_matrix tests/test_matrix.cpp)
  target_link_libraries(test_matrix PRIVATE face_core gtest_main)
  add_test(NAME matrix_tests COMMAND test_matrix)

  add_executable(test_pca tests/test_pca.cpp)
  target_link_libraries(test_pca PRIVATE face_core gtest_main)
  add_test(NAME pca_tests COMMAND test_pca)

  add_executable(test_gmm tests/test_gmm.cpp)
  target_link_libraries(test_gmm PRIVATE face_core gtest_main)
  add_test(NAME gmm_tests COMMAND test_gmm)

  add_executable(test_ffnn tests/test_ffnn.cpp)
  target_link_libraries(test_ffnn PRIVATE face_core gtest_main)
  add_test(NAME ffnn_tests COMMAND test_ffnn)
endif()
