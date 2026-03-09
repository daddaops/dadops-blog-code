"""CLIP-based text-to-image and image-to-image search.

From: https://dadops.dev/blog/clip-from-scratch/
"""
import torch
from zero_shot import tokenize


def text_to_image_search(model, query, image_embeds, images, top_k=5):
    """Search images using a text query."""
    query_embed = model.encode_text(tokenize([query]))    # [1, D]
    scores = (query_embed @ image_embeds.T).squeeze(0)    # [num_images]
    top_indices = scores.topk(top_k).indices
    return [(images[i], scores[i].item()) for i in top_indices]

# Usage: text_to_image_search(model, "sunset over the ocean", db_embeds, db_images)


def image_to_image_search(model, query_image, image_embeds, images, top_k=5):
    """Find visually similar images using the shared embedding space."""
    query_embed = model.encode_image(query_image.unsqueeze(0))
    scores = (query_embed @ image_embeds.T).squeeze(0)
    top_indices = scores.topk(top_k).indices
    return [(images[i], scores[i].item()) for i in top_indices]


if __name__ == "__main__":
    from clip_model import CLIP

    torch.manual_seed(42)

    # Create a tiny model (untrained — results will be random)
    model = CLIP(embed_dim=64, vision_dim=96, text_dim=64)
    model.eval()

    # Simulate a database of images
    num_db_images = 20
    db_images = [torch.randn(3, 224, 224) for _ in range(num_db_images)]
    db_labels = [f"image_{i}" for i in range(num_db_images)]

    # Pre-compute image embeddings for the database
    with torch.no_grad():
        db_stack = torch.stack(db_images)
        image_embeds = model.encode_image(db_stack)  # [20, D]

    # Text-to-image search
    print("=== Text-to-Image Search ===")
    query = "sunset over the ocean"
    results = text_to_image_search(model, query, image_embeds, db_labels, top_k=5)
    print(f"Query: '{query}'")
    print(f"Top {len(results)} results:")
    for label, score in results:
        print(f"  {label}: score={score:.4f}")

    # Image-to-image search
    print("\n=== Image-to-Image Search ===")
    query_image = torch.randn(3, 224, 224)
    results = image_to_image_search(model, query_image, image_embeds, db_labels, top_k=5)
    print(f"Top {len(results)} similar images:")
    for label, score in results:
        print(f"  {label}: score={score:.4f}")

    print("PASS")
