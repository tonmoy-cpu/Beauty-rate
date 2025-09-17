import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load TorchScript model
try:
    model = torch.jit.load("best_model.pt", map_location=device)
    model.eval()
    print("✅ Loaded TorchScript model: best_model.pt")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load TorchScript model: {e}")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict(image):
    try:
        img = Image.fromarray(image).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x).cpu().item()

        # Clamp strictly between 1 and 5
        pred = max(1.0, min(5.0, pred))
        return {"Beauty Score": round(pred, 3)}
    except Exception as e:
        return {"error": str(e)}

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a face image"),
    outputs=gr.Label(num_top_classes=1, label="Prediction"),
    title="Beauty Score Predictor",
    description="Upload an image and get a beauty score between 1 and 5."
)

if __name__ == "__main__":
    iface.launch()
