import os
import random

# Define image extensions to consider
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff')

def delete_100_photos_from_large_folders(root_folder):
    for root, dirs, files in os.walk(root_folder):
        # Filter image files
        image_files = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
        
        if len(image_files) > 100:
            print(f"Deleting 100 images from: {root} ({len(image_files)} images found)")
            to_delete = random.sample(image_files, 100)
            for file_name in to_delete:
                file_path = os.path.join(root, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    # 👇 Replace this path with your actual folder path
    folder_path = "C:\Users\vinay\Downloads\Indian"
    
    if os.path.isdir(folder_path):
        delete_100_photos_from_large_folders(folder_path)
    else:
        print("Invalid folder path.")
