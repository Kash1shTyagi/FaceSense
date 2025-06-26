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
      python3 \
      python3-numpy \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Generate dummy data (mean_face.bin, eigenvectors.bin, gmm_params.bin, FFNN bins, sample.pgm)
RUN python3 generate_dummy_data.py

# Generate minimal cmake/Config.cmake.in for package config
RUN mkdir -p cmake && \
    printf '%s\n' '@PACKAGE_INIT@' '' 'include("${CMAKE_CURRENT_LIST_DIR}/face_emotion_cppTargets.cmake")' \
    > cmake/Config.cmake.in

# Build without tests
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF && \
    make -j"$(nproc)"

# ──────────────────────────────────────────────────────────────────────────────
# 2) Runtime stage
# ──────────────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update \
 && apt-get install -y --no-install-recommends libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home appuser

USER appuser
WORKDIR /home/appuser

# Copy the built executable
COPY --from=builder /app/build/face_emotion ./face_emotion

# Copy generated data (bins + sample.pgm, cascade XML if present)
COPY --from=builder /app/data ./data

ENTRYPOINT ["./face_emotion"]
CMD ["--mode", "inference", "--data-dir", "data", "--camera", "0"]
