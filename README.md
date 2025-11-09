# 🧠 Smart Image Search (Vector Search with OpenCLIP + PostgreSQL)

This project is a **FastAPI-based smart image search system** that uses **OpenCLIP embeddings** to store and search images semantically.  
It supports **text-based queries** and stores **image + text embeddings** in a PostgreSQL database using the `pgvector` extension.

---

## 🚀 Features

- Uses **OpenCLIP (ViT-B-32)** model for generating embeddings  
- Stores embeddings in **PostgreSQL + pgvector**  
- Supports **semantic text-based image search**  
- FastAPI backend with **CORS enabled**  
- Image serving from local folder (`unsplash_images/`)

---

## 📦 Prerequisites

Make sure you have installed:

- Python 3.10 or higher  
- PostgreSQL 15+  
- `pgvector` extension for PostgreSQL  

---

## 🧱 Step 1: Setup PostgreSQL + pgvector

### 1️⃣ Create the database
```bash
sudo -u postgres psql
```
```sql
CREATE DATABASE image_search;
\c image_search;
```

### 2️⃣ Enable pgvector extension
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
### 3️⃣ Create table to store image embeddings
```sql
CREATE TABLE image_embeddings (
    image_id TEXT PRIMARY KEY,
    caption TEXT,
    image_url TEXT,
    img_embedding vector(512),
    txt_embedding vector(512)
);
```
## ⚙️ Step 2: Clone the Repository
```bash
git clone https://github.com/ravinandan25/vector-search.git
cd vector-search
```
## 🧩 Step 3: Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
## 🧰 Step 4: Install Required Dependencies
```bash
pip install -r requirements.txt
```

## 🔑 Step 5: Configure Environment Variables
Create file .env in project
Add the following keys:
```bash
# PostgreSQL Configuration
DB_NAME=image_search
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432

# Unsplash API Key
UNSPLASH_ACCESS_KEY=your_unsplash_access_key
```
## 🧰 Step 6: Load Envirement Variable
```bash
source .env
```

## 🖼️ Step 7: Download Images from Unsplash
Run the script to download sample images and captions.
```bash
python download_unsplash_images.py
```
(This will create a folder unsplash_images/ with .jpg and .txt files)

## 🧠 Step 8: Generate and Store Embeddings in Database
```bash
python generate_and_store_embeddings.py
```
This will:
- Load the OpenCLIP model
- Generate embeddings for each image
- Insert them into PostgreSQL

## ⚡ Step 9: Run FastAPI Server
```bash
uvicorn app:app --reload --port 8000
```
Then open index.htl file in browser and start using.

## 🧑‍💻 Author
Ravinandan 


## 📜 License

MIT License © 2025 Ravinandan Bhardwaj


