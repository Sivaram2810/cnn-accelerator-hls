import numpy as np

# ── What is a convolution? ─────────────────────────────
# A convolution slides a small filter across an image
# At each position it multiplies filter values with image values
# Then adds all those products together → one output number
# That output number represents "how much of this pattern exists here"

# ── Step 1: Create a fake 7x7 image ───────────────────
# In real life this would be pixel values (0-255)
# We use simple numbers so we can verify by hand
image = np.array([
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10,11,12,13,14],
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10,11,12,13,14],
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10,11,12,13,14],
    [1, 2, 3, 4, 5, 6, 7]
], dtype=np.float32)

# ── Step 2: Create a 3x3 filter ───────────────────────
# This specific filter is called an "edge detector"
# It finds vertical edges in the image
# When slid over a flat region → output near 0
# When slid over an edge → output large number
filter_3x3 = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# ── Step 3: Perform convolution manually ──────────────
# Output size = (image_size - filter_size + 1)
# For 7x7 image with 3x3 filter → 5x5 output
image_h, image_w = image.shape
filter_h, filter_w = filter_3x3.shape
output_h = image_h - filter_h + 1
output_w = image_w - filter_w + 1

output = np.zeros((output_h, output_w), dtype=np.float32)

# Slide the filter across the image
for i in range(output_h):
    for j in range(output_w):
        # Extract the patch of image the filter currently covers
        patch = image[i:i+filter_h, j:j+filter_w]
        # Multiply patch with filter elementwise, then sum
        output[i, j] = np.sum(patch * filter_3x3)

# ── Step 4: Print results ──────────────────────────────
print("Image (7x7):")
print(image)
print("\nFilter (3x3) - Vertical Edge Detector:")
print(filter_3x3)
print("\nOutput after convolution (5x5):")
print(output)
print("\nOutput shape:", output.shape)
print("\nMax value in output:", np.max(output))
print("Min value in output:", np.min(output))
# ── Step 5: Now create an image WITH a vertical edge ──
image_with_edge = np.array([
    [0, 0, 0, 255, 255, 255, 255],
    [0, 0, 0, 255, 255, 255, 255],
    [0, 0, 0, 255, 255, 255, 255],
    [0, 0, 0, 255, 255, 255, 255],
    [0, 0, 0, 255, 255, 255, 255],
    [0, 0, 0, 255, 255, 255, 255],
    [0, 0, 0, 255, 255, 255, 255]
], dtype=np.float32)

# Left half is black (0), right half is white (255)
# There is a sharp vertical edge in the middle

output2 = np.zeros((output_h, output_w), dtype=np.float32)

for i in range(output_h):
    for j in range(output_w):
        patch = image_with_edge[i:i+filter_h, j:j+filter_w]
        output2[i, j] = np.sum(patch * filter_3x3)

print("\n── New image with vertical edge ──")
print(image_with_edge)
print("\nOutput (should show large values at the edge):")
print(output2)