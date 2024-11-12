import os
import pandas as pd
import requests

download_folder = "downloaded_images"
os.makedirs(download_folder, exist_ok=True)

csv_file = "generated_prompts.csv"
df = pd.read_csv(csv_file)

def download_image(image_url, image_id):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image_path = os.path.join(download_folder, f"image{image_id}.jpg")
        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"Image image{image_id}.jpg downloaded successfully.")
        return image_path
    except Exception as e:
        print(f"Error downloading image {image_url}: {e}")
        return None

for idx, row in enumerate(df["Image URL"], start=1):
    download_image(row, idx)

print("All images downloaded.")
