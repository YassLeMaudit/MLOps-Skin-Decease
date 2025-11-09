import os, io, json
from typing import Dict
from PIL import Image
import torch
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "runs/skin5-current/model/model.keras")
CLASS_INDEX_PATH = os.getenv("CLASS_INDEX_PATH", "runs/skin5-current/data/class_index.json")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

app = FastAPI(title="Skin Disease Classifier (PyTorch)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
    class_index: Dict[str, int] = json.load(f)
idx_to_class = {v: k for k, v in class_index.items()}
num_classes = len(idx_to_class)

# --- Charge le modèle ---
has_full_model = False
try:
    model = torch.load(MODEL_PATH, map_location=device)
    has_full_model = True
except Exception:
    # Si tu as sauvegardé un state_dict, reconstruis l’archi ici (ex EfficientNet B0)
    import torch.nn as nn
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    backbone = efficientnet_b0(weights=None)  # pas de weights Internet
    in_features = backbone.classifier[1].in_features
    backbone.classifier[1] = nn.Linear(in_features, num_classes)
    model = backbone
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=False)

model.to(device).eval()

# --- Préproc ImageNet ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "model_path": MODEL_PATH,
        "classes": [idx_to_class[i] for i in range(num_classes)],
        "full_model_file": has_full_model,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            pred_idx = int(torch.argmax(logits, dim=1).item())
        return {
            "label": idx_to_class.get(pred_idx, str(pred_idx)),
            "probs": probs,
            "classes": [idx_to_class[i] for i in range(num_classes)]
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
