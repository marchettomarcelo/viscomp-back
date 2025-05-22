import numpy as np
from transformers import ViTImageProcessor, ViTModel

def get_vit_model_and_processor():
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    return processor, model

# Cosine similarity
def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    return float(dot_product / (magnitude1 * magnitude2))
