import os
import numpy as np

# Ensure output directory
os.makedirs("data", exist_ok=True)

# 1) mean_face.bin: 4096 doubles
mean = np.zeros((4096,), dtype=np.float64)
mean.tofile("data/mean_face.bin")

# 2) eigenvectors.bin: k × 4096 doubles (e.g. k=50)
k = 50
eig = np.zeros((k, 4096), dtype=np.float64)
eig.tofile("data/eigenvectors.bin")

# 3) gmm_params.bin: write M, k (as int32), then for each component:
#    weight (double), mean (k doubles), diag-covariance (k doubles)
M = 4  # number of GMM components (emotions)
with open("data/gmm_params.bin", "wb") as f:
    # write component count and dimension
    f.write(np.array([M, k], dtype=np.int32).tobytes())
    for _ in range(M):
        # weight
        f.write(np.array([1.0 / M], dtype=np.float64).tobytes())
        # mean vector
        f.write(np.zeros((k,), dtype=np.float64).tobytes())
        # diagonal covariance
        f.write(np.ones((k,), dtype=np.float64).tobytes())

# 4) FFNN weights/biases (all doubles)
H, C = 32, 4
np.zeros((H, k), dtype=np.float64).tofile("data/ffnn_w1.bin")
np.zeros((H,),    dtype=np.float64).tofile("data/ffnn_b1.bin")
np.zeros((C, H), dtype=np.float64).tofile("data/ffnn_w2.bin")
np.zeros((C,),     dtype=np.float64).tofile("data/ffnn_b2.bin")

# 5) Dummy PGM sample (64×64 black image) so test-image mode works
width, height = 64, 64
pgm_path = "data/sample.pgm"
with open(pgm_path, "wb") as pgm:
    # P5 header: magic, dimensions, maxval
    pgm.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
    # write all zeros (black)
    pgm.write(bytearray(width * height))

print("Dummy data generated in ./data/ (all in double precision)")
print("  • mean_face.bin, eigenvectors.bin, gmm_params.bin")
print("  • ffnn_w1.bin, ffnn_b1.bin, ffnn_w2.bin, ffnn_b2.bin")
print("  • sample.pgm ({}×{} black image)".format(width, height))
