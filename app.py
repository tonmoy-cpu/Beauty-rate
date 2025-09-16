from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define MobileNetV2 with regression head
model = models.mobilenet_v2(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 1)

# Load trained weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

#transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#routes
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Beauty Rate API is live!"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).cpu().item()

    pred = max(1.0, min(5.0, pred))

    return jsonify({"score": round(pred, 3)})

#run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
