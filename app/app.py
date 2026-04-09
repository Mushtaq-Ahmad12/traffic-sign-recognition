import torch
import gradio as gr
from PIL import Image
import os
import sys
import numpy as np

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
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_traffic_sign_model.pth')
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
        return None, "Please capture or upload an image."
    
    # Ensure image is PIL
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
        
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
        
    return results, f"Inference engine: {DEVICE_NAME}"

# Custom CSS for Premium Look
custom_css = """
#project-container {
    max-width: 1000px;
    margin: auto;
}
.header-box {
    text-align: center;
    padding: 20px;
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.status-badge {
    padding: 5px 15px;
    background: #065f46;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: bold;
}
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="project-container"):
        # Header
        with gr.Group(elem_classes="header-box"):
            gr.Markdown("# 🚦 Traffic Sign Recognition System")
            gr.Markdown(f"### Internship Project Dashboard | Status: <span class='status-badge'>Online</span>")
            gr.Markdown(f"Current Deployment: **{DEVICE_NAME} Acceleration**")

        with gr.Row():
            # Left Column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input Source")
                input_mode = gr.Tabs()
                with input_mode:
                    with gr.TabItem("Upload Image"):
                        upload_input = gr.Image(type="pil", label="Pick a sign image", sources=["upload"])
                    with gr.TabItem("Webcam Mode"):
                        webcam_input = gr.Image(type="pil", label="Capture real-time", sources=["webcam"])
                
                predict_btn = gr.Button("🚀 Analyze Sign", variant="primary")

            # Right Column: Results
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Prediction Results")
                output_labels = gr.Label(num_top_classes=3, label="Top Classifications")
                device_info = gr.Textbox(label="System Metadata", interactive=False)
                
                with gr.Accordion("How it works", open=False):
                    gr.Markdown("""
                        This system uses a Deep Convolutional Neural Network (CNN) trained on the GTSRB dataset.
                        It analyzes 43 different categories of signs across the European standard.
                    """)

        # Gallery / Reference (Optional - can be expanded)
        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ℹ️ Dataset Overview")
                gr.Markdown("The model is trained on **GTSRB (German Traffic Sign Recognition Benchmark)**, containing over 50,000 images across 43 classes.")

    # Event handlers (Updated for robustness)
    predict_btn.click(
        fn=predict,
        inputs=upload_input, # Default to upload for button
        outputs=[output_labels, device_info]
    )
    
    # Auto-predict on change
    upload_input.change(fn=predict, inputs=upload_input, outputs=[output_labels, device_info])
    webcam_input.change(fn=predict, inputs=webcam_input, outputs=[output_labels, device_info])

if __name__ == "__main__":
    # In Gradio 6+, css and theme should ideally be in launch or constructor depending on specifics,
    # but the warning said move them to launch.
    demo.launch(css=custom_css)
