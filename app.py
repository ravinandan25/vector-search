from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import psycopg2
import torch
import open_clip
import os
from dotenv import load_dotenv

# --- Load environment variables from .env ---
load_dotenv()

app = FastAPI(title="Smart Image Search API")

# --- Allow frontend to access API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve local images from "unsplash_images" folder ---
IMAGE_DIR = "unsplash_images"
if os.path.exists(IMAGE_DIR):
    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# --- Load OpenCLIP Model ---
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# --- Database Config ---
DB_CONFIG = dict(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
)


@app.get("/search")
def search_images(
    q: str = Query(..., description="Search query text"),
    text_weight: float = Query(
        0.7, ge=0.0, le=1.0, description="Weight for text embedding (0-1)"
    ),
    image_weight: float = Query(
        0.3, ge=0.0, le=1.0, description="Weight for image embedding (0-1)"
    ),
):
    """
    Search images using a mix of text and image embeddings.
    The `text_weight` and `image_weight` parameters control the influence.
    """

    # --- Encode query text ---
    with torch.no_grad():
        text_emb = model.encode_text(tokenizer([q]))
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
    text_vec = text_emb.squeeze().tolist()

    # --- Connect to DB ---
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # --- Combined similarity search ---
    # Weighted similarity: combine both text and image embeddings
    cur.execute(
        """
        SELECT image_id, caption, image_url,
            (
                (1 - (txt_embedding <=> %s::vector)) * %s +
                (1 - (img_embedding <=> %s::vector)) * %s
            ) AS combined_similarity
        FROM image_embeddings
        ORDER BY combined_similarity DESC
        LIMIT 20;
    """,
        (text_vec, text_weight, text_vec, image_weight),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    # --- Prepare response ---
    results = []
    for image_id, caption, image_url, similarity in rows:
        if image_url and os.path.exists(image_url):
            image_url = f"http://localhost:8000/images/{os.path.basename(image_url)}"
        results.append(
            {
                "image_id": image_id,
                "caption": caption,
                "image_url": image_url,
                "similarity": round(float(similarity), 4),
            }
        )

    return {"results": results}
