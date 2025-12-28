import numpy as np
import matplotlib.pyplot as plt

# 1. Generate a 2D NumPy array (e.g., 64x64 pixels) initialized with zeros.
image_size = 64
synthetic_image = np.zeros((image_size, image_size), dtype=float)

# 2. Add several synthetic 'objects' to this image using NumPy slicing and broadcasting:

# A bright square in the center (e.g., value 200).
center_x, center_y = image_size // 2, image_size // 2
square_half_side = 10 # 20x20 square
synthetic_image[center_y - square_half_side : center_y + square_half_side,
                center_x - square_half_side : center_x + square_half_side] = 200.0

# A horizontal bright line (e.g., value 150).
horizontal_line_row_start = image_size // 4
line_thickness = 3
synthetic_image[horizontal_line_row_start : horizontal_line_row_start + line_thickness, :] = 150.0

# A diagonal bright line (e.g., value 100) from top-left to bottom-right.
# Making it a 3-pixel thick diagonal for better visibility
for i in range(image_size):
    synthetic_image[i, i] = 100.0
    if i + 1 < image_size: # Shift one pixel right
        synthetic_image[i, i+1] = 100.0
    if i - 1 >= 0: # Shift one pixel left
        synthetic_image[i, i-1] = 100.0

# 3. Implement a basic 2D convolution function
def manual_convolve2d(image, kernel):
    """
    Performs 2D convolution on an image with a given kernel.
    Handles 'same' padding to ensure output size matches input size.

    Args:
        image (np.ndarray): The input 2D grayscale image.
        kernel (np.ndarray): The 2D convolution kernel.

    Returns:
        np.ndarray: The convolved image with the same dimensions as the input.
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Calculate padding for 'same' convolution.
    # Assumes odd-sized kernels, so padding is (kernel_size - 1) / 2 on each side.
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # Pad the image with zeros.
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Initialize the output array with zeros, ensuring float type for results.
    output = np.zeros_like(image, dtype=float)

    # Perform convolution using nested loops.
    for i in range(image_h):
        for j in range(image_w):
            # Extract the window (sub-array) from the padded image
            # corresponding to the current position and kernel size.
            window = padded_image[i : i + kernel_h, j : j + kernel_w]
            
            # Perform element-wise multiplication with the kernel and sum the results.
            output[i, j] = np.sum(window * kernel)
            
    return output

# 4. Define a 3x3 'edge detection' kernel (Laplacian approximation).
# A common Laplacian kernel that highlights regions of rapid intensity change (edges).
laplacian_kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=float)

# 5. Apply your manual_convolve2d function with the edge detection kernel to your generated image.
edge_detected_image = manual_convolve2d(synthetic_image, laplacian_kernel)

# 6. Visualize the original synthetic image and the convolved (edge-detected) image side-by-side.
fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # Create a figure with two subplots

# Plot the original synthetic image
axes[0].imshow(synthetic_image, cmap='gray', vmin=0, vmax=255) # Set vmin/vmax for consistent grayscale range
axes[0].set_title('Original Synthetic Image')
axes[0].axis('off') # Hide axes for cleaner image display

# Plot the edge-detected image
# For edge detection, values can be negative or positive, matplotlib's imshow
# with cmap='gray' will auto-scale to the min/max values of the array.
axes[1].imshow(edge_detected_image, cmap='gray')
axes[1].set_title('Edge Detected Image (Laplacian Kernel)')
axes[1].axis('off') # Hide axes

plt.tight_layout() # Adjust subplot params for a tight layout
plt.show()

# Print important results to stdout
print("--- Image Information ---")
print(f"Original Image Shape: {synthetic_image.shape}")
print(f"Original Image Max Value: {np.max(synthetic_image):.2f}")
print(f"Original Image Min Value: {np.min(synthetic_image):.2f}")

print("\n--- Edge Detection Kernel ---")
print(laplacian_kernel)

print("\n--- Convolved Image Information ---")
print(f"Edge Detected Image Shape: {edge_detected_image.shape}")
print(f"Edge Detected Image Max Value: {np.max(edge_detected_image):.2f}")
print(f"Edge Detected Image Min Value: {np.min(edge_detected_image):.2f}")