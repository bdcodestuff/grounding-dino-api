import jwt
import os
from fastapi import FastAPI, Header, File, UploadFile, HTTPException
import logging.config
import json
from .model import load_model, generate_bboxes
from .utils import load_image_from_file
from .config import settings

JWT_KEY = os.getenv("JWT_KEY")
JWT_ALGORITHM = "HS256"

app = FastAPI()

# Load the CLIPSEG model once when the app starts
model, processor = load_model(settings.dino_model)

# Configure logging
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Example root path handler
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Image Detection API!"}

@app.post("/detect")
async def segment(image: UploadFile = File(...), authorization: str = Header(None), text: str = None):

    try:
        decoded = secure(authorization)
    except:
        return {"error":"unauthorized"}
    try:
        print(image.content_type)
        if image.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image format")

        loaded_image = load_image_from_file(await image.read())
        
        objects_and_boxes = generate_bboxes(model, processor, loaded_image, text)
        

        return json.dumps(objects_and_boxes)

    except Exception as e:
        logger.exception("Error in /detect endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await image.close()

def secure(token):
    decoded_token = jwt.decode(token, JWT_KEY, algorithms=JWT_ALGORITHM)
    return decoded_token
