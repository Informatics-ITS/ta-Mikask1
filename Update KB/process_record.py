import pandas as pd
import requests
import json
import os
from cleanup import clean_content
import pymupdf4llm
import re
from tqdm import tqdm
import shutil
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

added_records = pd.read_csv("ADDED.csv")
updated_records = pd.read_csv("UPDATED.csv")

# Create directories
temp_pdf_dir = os.path.join('temp_pdfs')
chunks_dir = os.path.join('chunks')
os.makedirs(temp_pdf_dir, exist_ok=True)
os.makedirs(chunks_dir, exist_ok=True)

model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

pc = Pinecone(
    api_key="pcsk_5QDQPf_Q7jQsWbeArLGK9Ksk5QvdQ7cuyyVYHiJxsuaZYAZ7yJG7yxX7VYb6LXViDWQjaJ")
pinecone_index = pc.Index("uu-index")


def chunk_markdown_content(content, actual_filename, record):
    chunks = re.split(r'^\s*---\s*$', content, flags=re.MULTILINE)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Remove .pdf extension if present
    base_name = actual_filename.replace('.pdf', '')

    for i, chunk in enumerate(chunks):
        # Encode the chunk using SentenceTransformer
        chunk_embedding = model.encode(chunk)

        # Save as numpy array instead of markdown
        chunk_filename = f"{base_name}_chunk_{i+1}.npy"
        chunk_path = os.path.join(chunks_dir, chunk_filename)

        try:
            # Save the encoded chunk as .npy file
            np.save(chunk_path, chunk_embedding)

            item = {
                "id": chunk_filename,
                "values": chunk_embedding,
                "metadata": {
                    "nama": record['name'],
                    "file_name": record['file_name'],
                    "description": record['description'],
                },
            }

            pinecone_index.upsert(vectors=[item])
            print(f"Upserted chunk {i+1} of {actual_filename}")
        except Exception as e:
            print(
                f"Error saving encoded chunk {i+1} of {actual_filename}: {str(e)}")

    return len(chunks)

def process_record(records):
    for index, (_, record) in tqdm(enumerate(records.iterrows()), desc="Processing records"):
        real_file_name = record['real_file_name']
        actual_file_name = record.get('unseen_file_name', record['file_name'])

        # Parse all URLs from the JSON array
        file_urls = json.loads(real_file_name.replace("'", '"'))

        # Parse actual filenames as array
        actual_filenames = json.loads(actual_file_name.replace("'", '"'))

        for url_index, file_url in enumerate(file_urls):
            url = "https://peraturan.go.id" + file_url
            print(f"Downloading from: {url}")

            response = requests.get(url)

            if response.status_code == 200:
                try:
                    # Get the actual filename as string
                    actual_filename = actual_filenames[url_index]

                    # Save PDF to temporary file
                    pdf_path = os.path.join(temp_pdf_dir, actual_filename)
                    with open(pdf_path, 'wb') as pdf_file:
                        pdf_file.write(response.content)
                    print(f"Saved PDF to {pdf_path}")

                    # Convert PDF file to markdown
                    markdown_text = clean_content(
                        pymupdf4llm.to_markdown(pdf_path))

                    # Process the markdown text with the actual filename
                    chunk_markdown_content(
                        markdown_text, actual_filename, record)

                except IndexError:
                    print(f"Error: No actual filename found for index {url_index}")
                except Exception as e:
                    print(f"Error processing file: {str(e)}")
            else:
                print(f"Failed to download file: HTTP {response.status_code}")

    print(f"All files processed and chunks saved to {chunks_dir} directory")

if __name__ == "__main__":
    process_record(added_records)
