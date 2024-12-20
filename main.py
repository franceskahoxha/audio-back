from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import numpy as np
from scipy.io.wavfile import write
from transformers import EncodecModel
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DECODED_DIR = "decoded"
os.makedirs(DECODED_DIR, exist_ok=True)

# Serve decoded files
app.mount("/decoded", StaticFiles(directory=DECODED_DIR), name="decoded")

# Load the pre-trained EnCodec model
logger.info("Loading EnCodec model...")
encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
logger.info("EnCodec model loaded successfully.")

class DecodeRequest(BaseModel):
    encoded_data: list
    audio_scales: list

def validate_and_convert_input(data, scales):
    try:
        encoded_matrix = np.array(data).reshape(1, 8, -1).astype(np.int64)
        scales_matrix = np.array(scales).reshape(1, 8, -1).astype(np.float32)
        return torch.LongTensor(encoded_matrix), torch.FloatTensor(scales_matrix)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid input data format")

def save_decoded_audio(audio_tensor, output_path, sample_rate=24000):
    try:
        audio_np = (audio_tensor / torch.max(torch.abs(audio_tensor))).detach().cpu().numpy()
        audio_np = (audio_np * 32767).astype(np.int16)
        write(output_path, sample_rate, audio_np)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save decoded audio")

@app.post("/encode")
async def decode_audio(request: DecodeRequest):
    try:
        indices_tensor, scales_tensor = validate_and_convert_input(request.encoded_data, request.audio_scales)
        decoded_audio = encodec_model.decode([indices_tensor], audio_scales=scales_tensor)[0]

        # Save decoded audio
        output_file = os.path.join(DECODED_DIR, "decoded_audio.wav")
        save_decoded_audio(decoded_audio, output_file)

        # Return the relative file path
        return JSONResponse({"message": "File decoded and saved.", "file_path": f"decoded/decoded_audio.wav"})
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


#.\env\Scripts\activate
#uvicorn main:app --reload
