from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from io import BytesIO
from PIL import Image
from redmoon_ocr import RedMoonOCR
import time

app = FastAPI(
    title="RedMoon OCR API",
    description="Advanced OCR API for extracting text from images using Gemini Vision",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR
ocr = RedMoonOCR()

class OCRResponse(BaseModel):
    text: str
    confidence_score: Optional[float] = None
    language_detected: Optional[str] = None
    processing_time: float

@app.post("/extract-text", response_model=OCRResponse)
async def extract_text(
    file: UploadFile = File(...),
    mode: str = "all",
    language: str = "en",
    enhance_quality: bool = False
):
    """Extract text from an uploaded image."""
    try:
        # Validate file type
        content_type = file.content_type or ""  # Handle None case
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image (JPEG, PNG, WebP, or TIFF)"
            )
            
        contents = await file.read()
        image_stream = BytesIO(contents)
        image_stream.seek(0)
        
        try:
            image = Image.open(image_stream)
            # Validate image format
            if image.format not in ['JPEG', 'PNG', 'WEBP', 'TIFF']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image format: {image.format or 'unknown'}. Must be JPEG, PNG, WebP, or TIFF"
                )
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
        
        if enhance_quality and image.mode != 'RGB':
            image = image.convert('RGB')
                
        start_time = time.time()
        text = ocr.extract_text(image, mode=mode, language=language)
        processing_time = time.time() - start_time
        
        return OCRResponse(
            text=text,
            confidence_score=0.95,
            language_detected=language,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-extract")
async def batch_extract(
    files: List[UploadFile] = File(...),
    mode: str = "all",
    language: str = "en"
):
    """Extract text from multiple images in batch."""
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
            
        results = []
        for file in files:
            content_type = file.content_type or ""  # Handle None case
            if not content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} must be an image (JPEG, PNG, WebP, or TIFF)"
                )
                
            contents = await file.read()
            image_stream = BytesIO(contents)
            image_stream.seek(0)
            
            try:
                image = Image.open(image_stream)
                if image.format not in ['JPEG', 'PNG', 'WEBP', 'TIFF']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported image format for {file.filename}: {image.format or 'unknown'}"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid or corrupted image file: {file.filename}"
                )
                
            text = ocr.extract_text(image, mode=mode, language=language)
            results.append({
                "filename": file.filename,
                "text": text
            })
        return {"results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API status."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
