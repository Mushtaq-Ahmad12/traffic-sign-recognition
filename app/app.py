import torch
import gradio as gr
from PIL import Image
import os
import sys

# Add project root to path so 'src' can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TrafficSignCNN
from src.preprocessing import get_transforms

# Class mapping for GTSRB
CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons', 
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield', 
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited', 
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left', 
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road', 
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work', 
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End of all speed and passing limits', 33: 'Turn right ahead', 
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right', 
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left', 
    40: 'Roundabout mandatory', 41: 'End of no passing', 
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Configuration
MODEL_PATH = 'models/best_traffic_sign_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'

# Load model
model = TrafficSignCNN(num_classes=43).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Preprocessing transforms
transform = get_transforms(train=False)

def predict(image):
    if image is None:
        return "Please upload an image."
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        
    results = {}
    for i in range(3):
        label = CLASSES[top_indices[i].item()]
        results[label] = float(top_probs[i])
        
    return results, f"Inference Running on: {DEVICE_NAME}"

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top Predictions"),
        gr.Textbox(label="Hardware Device")
    ],
    title="Traffic Sign Recognition",
    description=f"Upload a traffic sign image to classify it. [Running on: {DEVICE_NAME}]",
    examples=[],
    theme="soft"
)

if __name__ == "__main__":
    interface.launch()
