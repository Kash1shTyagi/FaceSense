```markdown
# 🎭 Face Emotion Recognition in C++

Real-Time Facial Feature Extraction & Emotion Classification using C++ and OpenCV. This project implements PCA-based facial feature extraction and emotion prediction using classical ML methods (GMM, FFNN). It is optimized for real-time webcam inference and includes test suites for PCA, GMM, and FFNN.

---

## 🚀 Features

- Real-time face detection via Haar Cascade
- PCA for dimensionality reduction of facial features
- GMM (Gaussian Mixture Models) for probabilistic modeling
- Feedforward Neural Network (FFNN) for emotion classification
- Modular design: clean separation between feature extraction and classification
- GoogleTest-based unit testing for all core modules
- Lightweight and portable Docker build

---

## 🛠️ Tech Stack

- **Language**: C++17
- **Computer Vision**: OpenCV (`core`, `imgproc`, `objdetect`, `highgui`)
- **Build System**: CMake
- **ML Components**: PCA, GMM, FFNN (handwritten)
- **Testing**: GoogleTest
- **Packaging**: Docker multi-stage build

---

## 🗂️ Project Structure

```

face\_emotion\_cpp/
├── include/               # Public headers
├── src/                   # Core implementations
│   ├── main.cpp           # Entry point
│   ├── Matrix.cpp         # Matrix math utils
│   ├── PCA.cpp            # Principal Component Analysis
│   ├── GMM.cpp            # Gaussian Mixture Model
│   └── FFNN.cpp           # Simple Feedforward NN
├── tests/                 # GoogleTest unit tests
├── data/                  # Model weights, mean\_face.bin, sample.pgm, etc.
├── cmake/                 # Package config template
├── third\_party/           # External dependencies
│   └── googletest/        # Cloned at build time
├── CMakeLists.txt         # Build script
├── Dockerfile             # Docker build instructions
└── generate\_dummy\_data.py # Generates mock model weights for testing

````

---

## 🧪 Setup & Build (Local)

> Requires: Ubuntu 22.04+, OpenCV (`libopencv-dev`), CMake, g++, Python 3.8+

```bash
# Install dependencies
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev python3 python3-numpy

# Clone the repo
git clone https://github.com/yourname/face_emotion_cpp.git
cd face_emotion_cpp

# Generate dummy data
python3 generate_dummy_data.py

# Build the project
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
````

---

## 📷 Run the App

```bash
./build/face_emotion \
  --mode inference \
  --data-dir data \
  --camera 0
```

> You can also test on a static image:

```bash
./build/face_emotion \
  --mode image \
  --test-image data/sample.pgm \
  --data-dir data
```

---

## ✅ Run Unit Tests

```bash
cd build
ctest
```

> Or run individual test binaries:

```bash
./test_matrix
./test_pca
./test_gmm
./test_ffnn
```

---

## 🐳 Build & Run with Docker

> Builds using multi-stage to keep the final image minimal.

```bash
# Build the container
docker build -t face_emotion_cpp .

# Run the application
docker run --rm --device=/dev/video0 \
  face_emotion_cpp --mode inference --data-dir data --camera 0
```

---

## 📦 Installation (CMake Package)

The project supports CMake package exports. After installation:

```bash
cmake -DCMAKE_INSTALL_PREFIX=install -P cmake_install.cmake
```

You can then import `face_core` in other CMake projects via:

```cmake
find_package(face_emotion_cpp REQUIRED)
target_link_libraries(your_target PRIVATE face_emotion_cpp::face_core)
```

---

## 👥 Contributing

Feel free to fork, improve, and open pull requests. Please include unit tests for any new functionality.

