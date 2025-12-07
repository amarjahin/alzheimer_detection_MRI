import os
from PIL import Image
import matplotlib.pyplot as plt

path = "/Users/ammarjahin/.cache/kagglehub/datasets/ninadaithal/imagesoasis/versions/1"

print("Path to dataset files:", path)

# Example: View an image from the Non Demented folder
image_folder = os.path.join(path, "Data", "Non Demented")
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

if image_files:
    # Load and display the first image
    image_path = os.path.join(image_folder, image_files[0])
    print(f"\nDisplaying: {image_files[0]}")
    
    # Method 1: Using PIL
    img = Image.open(image_path)
    img.show()  # Opens in default image viewer
    
    # Method 2: Using matplotlib (better for analysis)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title(f"Image: {image_files[0]}")
    # plt.show()
