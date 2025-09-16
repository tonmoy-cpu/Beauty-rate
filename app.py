from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = torch.jit.load("best_model.pt", map_location=device)
    model.eval()
    print("Loaded TorchScript model: best_model.pt")
except Exception as e:
    print("Failed to load TorchScript model:", e)
    raise e


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


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

    try:
        file = request.files["image"]
        img = Image.open(file).convert("RGB")

        # Preprocess
        x = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred = model(x).cpu().item()

        # Clamp strictly between 1 and 5
        pred = max(1.0, min(5.0, pred))

        return jsonify({"score": round(pred, 3)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
