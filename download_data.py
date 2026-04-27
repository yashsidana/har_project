import os
import requests
import zipfile
import io

DATA_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
EXTRACT_DIR = "data"

def download_and_extract():
    print(f"Downloading dataset from {DATA_URL}...")
    response = requests.get(DATA_URL)
    response.raise_for_status()
    print("Download complete. Extracting...")
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(EXTRACT_DIR)
        
    # Check for nested zip files (common in UCI datasets)
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.endswith('.zip'):
                nested_zip_path = os.path.join(root, file)
                print(f"Extracting nested zip: {nested_zip_path}")
                with zipfile.ZipFile(nested_zip_path, 'r') as nested_zip:
                    nested_zip.extractall(root)
                
    print(f"Dataset extracted to {EXTRACT_DIR}/")

if __name__ == "__main__":
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR)
    download_and_extract()
