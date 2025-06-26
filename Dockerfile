# ──────────────────────────────────────────────────────────────────────────────
# 1) Build stage
# ──────────────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      ca-certificates \
      libopencv-dev \
      pkg-config \
      python3 python3-numpy \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Clone GoogleTest only
RUN rm -rf third_party/googletest \
 && mkdir -p third_party \
 && git clone --depth 1 https://github.com/google/googletest.git third_party/googletest

# Generate dummy data
RUN python3 generate_dummy_data.py

# Prepare CMake package-config stub
RUN mkdir -p cmake && \
    printf '%s\n' '@PACKAGE_INIT@' '' 'include("${CMAKE_CURRENT_LIST_DIR}/face_emotion_cppTargets.cmake")' \
    > cmake/Config.cmake.in

# Build without tests (using system OpenCV)
RUN rm -rf build && mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF \
    && make -j"$(nproc)"

# ──────────────────────────────────────────────────────────────────────────────
# 2) Runtime stage
# ──────────────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update \
 && apt-get install -y --no-install-recommends libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

# Copy binary & data
COPY --from=builder /app/build/face_emotion ./face_emotion
COPY --from=builder /app/data       ./data

ENTRYPOINT ["./face_emotion"]
CMD ["--mode", "inference", "--data-dir", "data", "--camera", "0"]
