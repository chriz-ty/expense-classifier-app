from sentence_transformers import SentenceTransformer

# Load the model from HuggingFace or wherever you used originally
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Save it properly with all configs
embedder.save("embedder")  # This creates a folder, not a file
