from dotenv import load_dotenv
import os
import torch
import open_clip
import psycopg2
from PIL import Image
from tqdm import tqdm

load_dotenv()

print(os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_PASSWORD"), os.getenv("DB_HOST"), os.getenv("DB_PORT"))

# --- Database Connection ---
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)
cur = conn.cursor()

# --- Ensure table exists with separate embeddings ---
cur.execute("""
CREATE TABLE IF NOT EXISTS image_embeddings (
    image_id TEXT PRIMARY KEY,
    caption TEXT,
    image_url TEXT,
    img_embedding vector(512),
    txt_embedding vector(512)
);
""")
conn.commit()

# --- Load OpenCLIP Model ---
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# --- Folder containing images ---
folder = "unsplash_images"

# --- Process each image ---
for file in tqdm(os.listdir(folder)):
    if not file.endswith(".jpg"):
        continue

    image_id = file.replace(".jpg", "")
    txt_path = os.path.join(folder, image_id + ".txt")

    if not os.path.exists(txt_path):
        print(f"⚠️ Skipping {file} (no caption file found)")
        continue

    with open(txt_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()

    image_path = os.path.join(folder, file)

    # --- Encode image and text ---
    image_tensor = preprocess(Image.open(image_path)).unsqueeze(0)
    text_tensor = tokenizer([caption])

    with torch.no_grad():
        img_emb = model.encode_image(image_tensor)
        txt_emb = model.encode_text(text_tensor)

        # Normalize both embeddings
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
        txt_emb /= txt_emb.norm(dim=-1, keepdim=True)

    # Convert to lists for Postgres
    img_vec = img_emb.squeeze().tolist()
    txt_vec = txt_emb.squeeze().tolist()

    # --- Insert or Update ---
    cur.execute("""
        INSERT INTO image_embeddings (image_id, caption, image_url, img_embedding, txt_embedding)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (image_id)
        DO UPDATE SET
            caption = EXCLUDED.caption,
            image_url = EXCLUDED.image_url,
            img_embedding = EXCLUDED.img_embedding,
            txt_embedding = EXCLUDED.txt_embedding;
    """, (image_id, caption, image_path, img_vec, txt_vec))

    print(f"✅ Stored/Updated embeddings for image ID: {image_id}")

# --- Finalize ---
conn.commit()
cur.close()
conn.close()
print("🎯 All image & text embeddings inserted or updated successfully!")




