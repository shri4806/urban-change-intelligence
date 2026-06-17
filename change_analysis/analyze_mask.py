from PIL import Image
import numpy as np

# Change this later if you want another mask
image_path = r"C:\Users\shrit\ChangeFormer\samples_LEVIR\predict_CD_ChangeFormerV6\test_2_0000_0000.png"

# Load image
img = Image.open(image_path)

# Convert image to numpy array
img_array = np.array(img)

# Count changed pixels (white pixels)
changed_pixels = np.sum(img_array == 255)

# Total pixels
total_pixels = img_array.size

# Calculate percentage
change_percentage = (changed_pixels / total_pixels) * 100

print("Image loaded successfully!")
print("Shape:", img_array.shape)
print("Data type:", img_array.dtype)
print("Unique pixel values:", np.unique(img_array))

print("\n----- Change Statistics -----")
print("Changed Pixels:", changed_pixels)
print("Total Pixels:", total_pixels)
print("Change Percentage:", round(change_percentage, 2), "%")