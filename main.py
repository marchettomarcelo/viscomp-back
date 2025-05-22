from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from enum import Enum
from PIL import Image
import torch
import os
from io import BytesIO
from utils.model_utils import get_vit_model_and_processor, calculate_cosine_similarity

class AllowedText(str, Enum):
    CAVALO = "cavalo"
    ESTRELA = "estrela"
    GATO = "gato"
    LINUS = "linus"
    LUMINARIA = "luminaria"
    MACK = "mack"
    NIKE = "nike"
    RAPOSA = "raposa"

app = FastAPI(
    title="VisComp API",
    description="Backend API for VisComp application",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and processor once at startup


vit_processor, vit_model = get_vit_model_and_processor()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/process-image")
async def process_image(
    image: UploadFile = File(...),
    text: str = Form(...),
):
    """
    Process an image with associated text and compare it to the respective Canny image.
    """
    # Validate text input
    try:
        validated_text = AllowedText(text.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid text value. Must be one of: {', '.join([e.value for e in AllowedText])}"
        )
    # Read the uploaded image
    image_content = await image.read()
    try:
        uploaded_image = Image.open(BytesIO(image_content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Find the corresponding Canny image
    canny_filename = f"canny_{validated_text.value}.png"
    canny_path = os.path.join("fotos_canny", canny_filename)
    if not os.path.exists(canny_path):
        raise HTTPException(status_code=404, detail=f"Canny image not found for {validated_text.value}.")
    canny_image = Image.open(canny_path).convert("RGB")

    # Process both images with ViT
    inputs1 = vit_processor(images=uploaded_image, return_tensors="pt")
    inputs2 = vit_processor(images=canny_image, return_tensors="pt")

    with torch.no_grad():
        outputs1 = vit_model(**inputs1)
        outputs2 = vit_model(**inputs2)

    vec1 = outputs1.last_hidden_state[:, 0, :].squeeze(0).numpy()
    vec2 = outputs2.last_hidden_state[:, 0, :].squeeze(0).numpy()

    
    similarity = calculate_cosine_similarity(vec1, vec2)

    return {
        "filename": image.filename,
        "text": validated_text.value,
        "cosine_similarity": similarity
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
