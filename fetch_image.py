from dotenv import load_dotenv
import requests, os
from tqdm import tqdm

load_dotenv()

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
SEARCH_QUERY = "cars"
OUT_DIR = "unsplash_images"
os.makedirs(OUT_DIR, exist_ok=True)

for page in range(1, 10):
    url = f"https://api.unsplash.com/search/photos?page={page}&query={SEARCH_QUERY}&client_id={UNSPLASH_ACCESS_KEY}"
    data = requests.get(url).json()
    print(f"Downloading page {page} with {len(data['results'])} images...")

    for img in data["results"]:
        img_url = img["urls"]["regular"]
        desc = img.get("alt_description", "")
        img_id = img["id"]

        img_data = requests.get(img_url).content
        with open(f"{OUT_DIR}/{img_id}.jpg", "wb") as f:
            f.write(img_data)

        with open(f"{OUT_DIR}/{img_id}.txt", "w") as f:
            f.write(desc or "No description")
