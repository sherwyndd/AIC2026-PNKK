"""Check Qdrant collection metadata and encoder compatibility."""

from qdrant_client import QdrantClient
from models.siglip_encoder import load_siglip, encode_text_siglip

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Load current SigLIP model
print("Loading current SigLIP model...")
load_siglip(device="cuda:0")

# Check collection info
collection_name = "image_siglip"
try:
    collection_info = client.get_collection(collection_name)
    print(f"\n=== Collection: {collection_name} ===")
    print(f"Vectors count: {collection_info.points_count}")
    print(f"Vector dimension: {collection_info.config.params.vectors.size}")
    print(f"Distance metric: {collection_info.config.params.vectors.distance}")
    
    # Test current model encoding dimension
    test_query = "test"
    test_emb = encode_text_siglip(test_query)
    print(f"\n=== Current model encoding ===")
    print(f"Model: google/siglip2-base-patch16-224")
    print(f"Embedding dimension: {test_emb.shape[0]}")
    
    # Check compatibility
    collection_dim = collection_info.config.params.vectors.size
    model_dim = test_emb.shape[0]
    
    print(f"\n=== Compatibility check ===")
    if collection_dim == model_dim:
        print(f"✓ Compatible: Collection dim ({collection_dim}) == Model dim ({model_dim})")
    else:
        print(f"✗ INCOMPATIBLE: Collection dim ({collection_dim}) != Model dim ({model_dim})")
        print("  Need to re-index collection with new encoder!")
        
except Exception as e:
    print(f"Error: {e}")
