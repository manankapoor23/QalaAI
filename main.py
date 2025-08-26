from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from PIL import Image
import io
import torch
from torchvision import models, transforms
import torch.nn as nn

# ===== 1. SETTINGS =====
MODEL_PATH = "/Users/manankapoor/Desktop/untitled folder 4/art_style_model.pth"
CLASSES = ['Baroque', 'Cubism', 'Expressionism', 'Impressionism', 'Pop Art', 'Surrealism']
device = torch.device("cpu")

# ===== 2. IMAGE TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===== 3. LOAD MODEL =====
def load_model():
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Initialize model
model = load_model()

# ===== 4. FASTAPI SETUP =====
app = FastAPI(title="Art Style Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates setup
templates = Jinja2Templates(directory="templates")

# ===== 5. ROUTES =====

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Serve the homepage"""
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/classifier", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def classifier_page(request: Request):
    """Serve the art style classifier"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/home.html", response_class=HTMLResponse)
async def home_redirect(request: Request):
    """Redirect to homepage"""
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predict")
async def predict_art_style(file: UploadFile = File(...)):
    """Predict art style from uploaded image"""
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Transform image for model
        img_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
            top3_prob, top3_idx = torch.topk(probs, 3)
            
        # Format response
        prediction = {
            "success": True,
            "filename": file.filename,
            "top3": [
                {
                    "style": CLASSES[top3_idx[i].item()], 
                    "confidence": round(float(top3_prob[i] * 100), 2)
                }
                for i in range(3)
            ],
            "best_guess": {
                "style": CLASSES[top3_idx[0].item()],
                "confidence": round(float(top3_prob[0] * 100), 2)
            }
        }
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/styles")
async def get_styles():
    """Get list of supported art styles"""
    return {"styles": CLASSES}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)