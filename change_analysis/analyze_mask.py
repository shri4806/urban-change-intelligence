from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
import json

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

# Detect separate regions
binary_mask = img_array > 0

labeled_mask = label(binary_mask)
regions = regionprops(labeled_mask)

# Store region information
region_info = []

height, width = img_array.shape

for region in regions:

    centroid_y, centroid_x = region.centroid

    # Horizontal position

    if centroid_x < width / 3:
        horizontal = "left"
    elif centroid_x < 2 * width / 3:
        horizontal = "center"
    else:
        horizontal = "right"

    # Vertical position

    if centroid_y < height / 3:
        vertical = "upper"
    elif centroid_y < 2 * height / 3:
        vertical = "middle"
    else:
        vertical = "bottom"

    location = f"{vertical}-{horizontal}"

    region_info.append({
        "id": region.label,
        "size": region.area,
        "bbox": region.bbox,
        "centroid": region.centroid,
        "location": location
    })

# Sort largest to smallest
region_info.sort(
    key=lambda x: x["size"],
    reverse=True
)

num_regions = labeled_mask.max()

# Calculate size of each region
region_sizes = []

for region_id in range(1, num_regions + 1):
    size = np.sum(labeled_mask == region_id)
    region_sizes.append(size)

largest_region = max(region_sizes)
smallest_region = min(region_sizes)
average_region = sum(region_sizes) / len(region_sizes)

print("Image loaded successfully!")
print("Shape:", img_array.shape)
print("Data type:", img_array.dtype)
print("Unique pixel values:", np.unique(img_array))

print("\n----- Change Statistics -----")
print("Changed Pixels:", changed_pixels)
print("Total Pixels:", total_pixels)
print("Change Percentage:", round(change_percentage, 2), "%")
print("Regions Detected:", num_regions)
print("Largest Region:", largest_region, "pixels")
print("Smallest Region:", smallest_region, "pixels")
print("Average Region Size:", round(average_region, 2), "pixels")

# Create report dictionary
report = {
    "changed_pixels": int(changed_pixels),
    "total_pixels": int(total_pixels),
    "change_percentage": round(change_percentage, 2),
    "regions_detected": int(num_regions),
    "largest_region": int(largest_region),
    "smallest_region": int(smallest_region),
    "average_region": round(average_region, 2),

   "top_regions": [
    {
        "id": region["id"],
        "size": int(region["size"]),
        "location": region["location"]
    }
    for region in region_info[:3]
]
}


# Save JSON report
with open("change_report.json", "w") as file:
    json.dump(report, file, indent=4)

print("\nJSON report saved successfully!")

# Create image for visualization
boxed_image = img.convert("RGB")

draw = ImageDraw.Draw(boxed_image)

for region in regions:

    min_row, min_col, max_row, max_col = region.bbox

    draw.rectangle(
        [(min_col, min_row), (max_col, max_row)],
        outline="red",
        width=2
    )

    draw.text(
    (min_col, max(min_row - 15, 0)),
    f"{region.label}",
    fill="yellow"
)

boxed_image.save("boxed_regions.png")

print("Bounding box image saved successfully!")

print("\n----- Top Development Zones -----")

for region in region_info[:3]:
    print(
    f"Region {region['id']} : {region['size']} pixels ({region['location']})"
)

