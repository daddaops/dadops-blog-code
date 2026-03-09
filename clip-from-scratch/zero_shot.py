"""CLIP zero-shot classification and prompt ensembling.

From: https://dadops.dev/blog/clip-from-scratch/
"""
import torch
import torch.nn.functional as F


def tokenize(texts):
    """Placeholder tokenizer — maps text to random token IDs.

    In a real system, this wraps a BPE tokenizer (see the tokenization post).
    """
    max_len = 77
    token_ids = []
    for text in texts:
        # Deterministic hash-based fake tokenization for reproducibility
        ids = [hash(text + str(i)) % 49152 for i in range(min(len(text.split()) + 2, max_len))]
        # Pad to max_len
        ids = ids + [0] * (max_len - len(ids))
        token_ids.append(ids[:max_len])
    return torch.tensor(token_ids)


def zero_shot_classify(model, image, class_names, prompt="a photo of a {}"):
    """Classify an image into one of the given classes, zero-shot.

    No training on these classes required — just their names.
    """
    # Create text prompts for each class
    prompts = [prompt.format(name) for name in class_names]

    # Encode everything (tokenize() wraps your BPE tokenizer — see our tokenization post)
    with torch.no_grad():
        image_embed = model.encode_image(image.unsqueeze(0))    # [1, D]
        text_embeds = model.encode_text(tokenize(prompts))      # [C, D]

    # Cosine similarity (both are L2-normalized)
    similarities = (image_embed @ text_embeds.T).squeeze(0)     # [C]

    # Return the class with highest similarity
    best_idx = similarities.argmax().item()
    return class_names[best_idx], similarities


# A sampling of CLIP's 80 prompt templates for ImageNet
CLIP_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "a drawing of a {}.",
    "a painting of a {}.",
    "a sculpture of a {}.",
    "a rendering of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a pixelated photo of a {}.",
    "a photo of the dirty {}.",
    "a photo of the clean {}.",
    "a tattoo of a {}.",
    "the origami {}.",
    "a plushie {}.",
    "a toy {}.",
    "itap of a {}.",        # "I took a picture of a..."
    "a {} in a video game.",
    "graffiti of a {}.",
]


def build_ensemble_classifier(model, class_names, templates):
    """Build zero-shot classifier with prompt ensembling.

    Average embeddings across multiple prompt templates per class.
    Gains ~3.5% accuracy on ImageNet vs. a single template.
    """
    ensemble_weights = []

    with torch.no_grad():
        for class_name in class_names:
            # Embed all templates for this class
            prompts = [t.format(class_name) for t in templates]
            embeddings = model.encode_text(tokenize(prompts))   # [T, D]

            # Average and re-normalize
            class_embedding = embeddings.mean(dim=0)
            class_embedding = F.normalize(class_embedding, dim=0)
            ensemble_weights.append(class_embedding)

    # Stack into a [num_classes, D] matrix for fast classification
    return torch.stack(ensemble_weights)


def classify_with_ensemble(model, image, ensemble_weights, class_names):
    """Classify using pre-computed ensemble weights."""
    with torch.no_grad():
        image_embed = model.encode_image(image.unsqueeze(0))    # [1, D]

    similarities = (image_embed @ ensemble_weights.T).squeeze(0) # [C]
    best_idx = similarities.argmax().item()
    return class_names[best_idx], similarities[best_idx].item()


if __name__ == "__main__":
    from clip_model import CLIP

    torch.manual_seed(42)

    # Create a tiny model (untrained — results will be random)
    model = CLIP(embed_dim=64, vision_dim=96, text_dim=64)
    model.eval()

    # Dummy image
    image = torch.randn(3, 224, 224)
    class_names = ["dog", "cat", "car", "airplane", "pizza"]

    # Zero-shot classification
    print("=== Zero-Shot Classification ===")
    predicted, sims = zero_shot_classify(model, image, class_names)
    print(f"Predicted class: {predicted}")
    print("Similarities:")
    for name, sim in zip(class_names, sims.tolist()):
        print(f"  {name:>10}: {sim:+.4f}")

    # Prompt ensembling
    print("\n=== Prompt Ensembling ===")
    print(f"Using {len(CLIP_TEMPLATES)} prompt templates")
    weights = build_ensemble_classifier(model, class_names, CLIP_TEMPLATES)
    print(f"Ensemble weights shape: {weights.shape}")

    predicted_ens, score = classify_with_ensemble(model, image, weights, class_names)
    print(f"Predicted class (ensemble): {predicted_ens} (score={score:.4f})")
    print("PASS")
