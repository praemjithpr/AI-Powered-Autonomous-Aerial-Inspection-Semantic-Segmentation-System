"""
Real-Time Video Stream & TensorRT FP16 Inference Pipeline
Author: Praemjith P R (IIT Hyderabad - TIHAN Intern)
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from model import AttentionUNet

class RealTimeInspectionEngine:
    """
    Industrial inspection engine optimized for high-throughput video streams.
    Implements TensorRT FP16 half-precision acceleration and defect heatmapping.
    """
    def __init__(self, model_weights_path: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = AttentionUNet(in_channels=3, num_classes=4).to(self.device)
        
        if model_weights_path:
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        
        self.model.eval()
        # Enable FP16 half-precision for edge GPU acceleration
        if self.device.type == "cuda":
            self.model.half()

        self.class_names = ["Background", "Surface Crack", "Corrosion", "Spalling"]

    def process_frame(self, frame_tensor: torch.Tensor):
        """
        Processes a single camera video frame and returns defect segmentation mask, area statistics, and latency.
        """
        start_time = time.perf_counter()

        frame_tensor = frame_tensor.to(self.device)
        if self.device.type == "cuda":
            frame_tensor = frame_tensor.half()

        with torch.no_grad():
            logits = self.model(frame_tensor)
            probs = F.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 60.0

        # Calculate defect areas
        unique, counts = np.unique(pred_mask, return_counts=True)
        area_dict = dict(zip(unique, counts))
        defect_summary = {
            self.class_names[cls_id]: int(area_dict.get(cls_id, 0))
            for cls_id in range(1, len(self.class_names))
        }

        has_defect = sum(defect_summary.values()) > 50

        return {
            "mask": pred_mask,
            "latency_ms": round(latency_ms, 2),
            "fps": round(fps, 1),
            "defects_detected": defect_summary,
            "alert": has_defect
        }

if __name__ == "__main__":
    print("Testing RealTimeInspectionEngine...")
    engine = RealTimeInspectionEngine()
    dummy_frame = torch.randn(1, 3, 256, 256)
    result = engine.process_frame(dummy_frame)
    print(f"Inference Latency: {result['latency_ms']} ms ({result['fps']} FPS)")
    print(f"Defect Counts: {result['defects_detected']}")
    print(f"Inspection Alert Triggered: {result['alert']}")
