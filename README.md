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
- **ML Components**: PCA, GMM, FFNN (custom implemented)
- **Testing**: GoogleTest
- **Packaging**: Docker multi-stage build

---

## 🗂️ Project Structure
```bash
face_emotion_cpp/
├── include/ # Public headers
├── src/ # Core implementations
│ ├── main.cpp # Entry point
│ ├── Matrix.cpp # Matrix math utils
│ ├── PCA.cpp # Principal Component Analysis
│ ├── GMM.cpp # Gaussian Mixture Model
│ └── FFNN.cpp # Simple Feedforward NN
├── tests/ # GoogleTest unit tests
├── data/ # Model weights, mean_face.bin, sample.pgm, etc.
├── cmake/ # Package config template
├── third_party/ # External dependencies (OpenCV, GTest)
├── CMakeLists.txt # Build script
├── Dockerfile # Docker build instructions
└── generate_dummy_data.py # Generates mock model weights for testing

```

## ⚙️ Local Setup

### Prerequisites

- Ubuntu 22.04+ (or WSL)
- CMake >= 3.15
- g++ with C++17 support
- Python 3.x
- OpenCV (`libopencv-dev`)

### Build Instructions

```bash
# Clone the repo
git clone https://github.com/<your-username>/face_emotion_cpp.git
cd face_emotion_cpp

# Install dependencies
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev python3 python3-numpy

# Generate dummy model files
python3 generate_dummy_data.py

# Configure & Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```


## 📷 Running the App

### With Webcam (Real-Time Inference)

```bash
./build/face_emotion --mode inference --data-dir data --camera 0
```

### With Static Image

```bash
./build/face_emotion --mode image --test-image data/sample.pgm --data-dir data
```

---

## ✅ Running Unit Tests

```bash
cd build
ctest
```

Or run individual test files:

```bash
./test_matrix
./test_pca
./test_gmm
./test_ffnn
```

---

## 🐳 Docker Build & Run

### Build the Docker image

```bash
docker build -t face_emotion_cpp .
```

### Run the container

> Make sure your webcam is exposed to Docker (`/dev/video0`):

```bash
docker run --rm --device=/dev/video0 \
  face_emotion_cpp --mode inference --data-dir data --camera 0
```

---

## 📦 CMake Package Installation

You can install and use this as a CMake package in other projects:

```bash
cmake -B build -DCMAKE_INSTALL_PREFIX=install
cmake --build build --target install
```

In your CMake project:

```cmake
find_package(face_emotion_cpp REQUIRED)
target_link_libraries(my_target PRIVATE face_emotion_cpp::face_core)
```

---

## 🧠 Contributing

1. Fork the repository
2. Create a new feature branch
3. Write clean, modular code with tests
4. Submit a pull request with clear commit messages
