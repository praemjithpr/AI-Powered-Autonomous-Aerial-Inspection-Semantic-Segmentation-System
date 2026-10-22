"""
FastAPI Microservice for Live Video Stream Inspection & Defect Alerts
Author: Praemjith P R (IIT Hyderabad - TIHAN Intern)
"""

from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
from PIL import Image
import io
import time
from inference import RealTimeInspectionEngine

app = FastAPI(
    title="Industrial Visual Defect Inspection Microservice",
    description="Real-time video telemetry and defect segmentation service developed with TIHAN IIT Hyderabad.",
    version="1.0.0"
)

engine = RealTimeInspectionEngine()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(engine.device),
        "target_classes": engine.class_names
    }

@app.post("/inspect/frame")
async def inspect_frame(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((256, 256))
    
    # Preprocess
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)

    result = engine.process_frame(img_tensor)
    
    return {
        "filename": file.filename,
        "latency_ms": result["latency_ms"],
        "fps": result["fps"],
        "defect_summary": result["defects_detected"],
        "hazard_alert": result["alert"],
        "timestamp": time.time()
    }
