import os
from pathlib import Path
from typing import Optional, List, Union
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

class RedMoonOCR:
    """Advanced OCR system using Gemini Vision for text and handwriting extraction."""
    
    def __init__(self, temperature: float = 0.7):
        """Initialize the OCR system.
        
        Args:
            temperature: Controls randomness in model responses (0.0 to 1.0)
        """
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        
        # Configure the model
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

    def extract_text(self, image: Union[str, Path, Image.Image], 
                    mode: str = "all",
                    language: str = "en") -> str:
        """Extract text from an image.
        
        Args:
            image: Path to image file or PIL Image object
            mode: Extraction mode - 'all' or 'handwritten'
            language: Expected language of the text (default: English)
            
        Returns:
            Extracted text as a string
        """
        # Handle different input types
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path)
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path or PIL Image object")
            
        # Simplified prompts for two modes
        mode_prompts = {
            "all": """Analyze this image and extract ALL text content, both printed and handwritten.
                     Format the output in a clear, structured way.
                     Include any special characters, numbers, and punctuation marks accurately.""",
            
            "handwritten": """Extract ONLY handwritten content from this image.
                            Ignore all printed text.
                            Try to interpret the handwriting as accurately as possible."""
        }
        
        if mode not in mode_prompts:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: {list(mode_prompts.keys())}")
            
        base_prompt = f"""Language: {language}
                         Task: {mode_prompts[mode]}
                         Additional Instructions:
                         - Preserve case sensitivity
                         - Maintain original punctuation
                         - Keep line breaks and spacing where significant
                         Please provide the extracted text without any additional commentary."""
        
        # Generate content using the image
        response = self.model.generate_content([base_prompt, image])
        return response.text.strip()
    
    def batch_extract(self, images: List[Union[str, Path, Image.Image]], 
                     mode: str = "all",
                     language: str = "en") -> List[str]:
        """Extract text from multiple images.
        
        Args:
            images: List of image paths or PIL Image objects
            mode: Extraction mode
            language: Expected language of the text
            
        Returns:
            List of extracted texts
        """
        return [self.extract_text(img, mode, language) for img in images]
