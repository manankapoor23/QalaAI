from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
import io
import torch
from torchvision import models, transforms
import torch.nn as nn
import os
import logging
from pathlib import Path

# ===== 1. SETTINGS =====
# Use environment variables for production
MODEL_PATH = os.getenv("MODEL_PATH", "/Users/manankapoor/Desktop/untitled folder 4/art_style_model.pth")
PORT = int(os.getenv("PORT", 8002))
HOST = os.getenv("HOST", "127.0.0.1")
CLASSES = ['Baroque', 'Cubism', 'Expressionism', 'Impressionism', 'Pop Art', 'Surrealism']
device = torch.device("cpu")

# File upload limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== 2. IMAGE TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===== 3. LOAD MODEL =====
def load_model():
    """Load the trained model with error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        logger.info(f"Loading model from {MODEL_PATH}")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        
        # Load model weights
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

# Initialize model
try:
    model = load_model()
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    model = None

# ===== 4. UTILITY FUNCTIONS =====
def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file"""
    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file extension
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

async def process_image(file: UploadFile) -> Image.Image:
    """Process uploaded image file"""
    try:
        contents = await file.read()
        
        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Open and convert image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# ===== 5. FASTAPI SETUP =====
app = FastAPI(
    title="Art Style Classifier",
    description="AI-powered art style classification for paintings",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# CORS middleware - adjust for production
allowed_origins = ["*"] if os.getenv("ENVIRONMENT") != "production" else [
    "https://yourdomain.com",
    "https://www.yourdomain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Setup templates and static files
template_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=template_dir)

# Mount static files if directory exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ===== 6. ROUTES =====

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Serve the homepage"""
    try:
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        raise HTTPException(status_code=500, detail="Error loading home page")

@app.get("/classifier", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def classifier_page(request: Request):
    """Serve the art style classifier"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving classifier page: {e}")
        raise HTTPException(status_code=500, detail="Error loading classifier page")

@app.get("/home.html", response_class=HTMLResponse)
async def home_redirect(request: Request):
    """Redirect to homepage"""
    try:
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        raise HTTPException(status_code=500, detail="Error loading home page")

@app.post("/predict")
async def predict_art_style(file: UploadFile = File(...)):
    """Predict art style from uploaded image"""
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Validate file
    validate_image_file(file)
    
    try:
        # Process image
        image = await process_image(file)
        logger.info(f"Processing image: {file.filename}")
        
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
        
        logger.info(f"Prediction completed for {file.filename}: {prediction['best_guess']}")
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "supported_styles": len(CLASSES),
        "version": "1.0.0"
    }

@app.get("/styles")
async def get_styles():
    """Get list of supported art styles"""
    return {
        "styles": CLASSES,
        "count": len(CLASSES)
    }

@app.get("/info")
async def get_info():
    """Get application information"""
    return {
        "title": "Art Style Classifier",
        "version": "1.0.0",
        "supported_styles": CLASSES,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "allowed_extensions": list(ALLOWED_EXTENSIONS)
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error"}

# ===== 7. RUN SERVER =====
if __name__ == "__main__":
    # For production, use a proper ASGI server like gunicorn
    if os.getenv("ENVIRONMENT") == "production":
        # Production settings
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=PORT,
            workers=1,  # Adjust based on your server
            access_log=True
        )
    else:
        # Development settings
        uvicorn.run(
            "main:app",
            host=HOST,
            port=PORT,
            reload=True,
            debug=True
        )