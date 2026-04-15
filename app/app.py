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
    print(f"Successfully loaded model from: {MODEL_PATH}")
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}. Using random weights!")
model.eval()

# Preprocessing transforms
transform = get_transforms(train=False)

def predict(image):
    if image is None:
        return None, None, "Please capture or upload an image."
    
    # Ensure image is PIL
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
        
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Prepare visualization for UI
    vis_img = img_tensor[0].cpu().permute(1, 2, 0).numpy()
    # Partial denormalization for display
    vis_img = (vis_img * np.array([0.2672, 0.2564, 0.2629])) + np.array([0.3337, 0.3064, 0.3171])
    vis_img = np.clip(vis_img, 0, 1)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        
    results = {}
    for i in range(3):
        label = CLASSES[top_indices[i].item()]
        results[label] = float(top_probs[i])
        
    return results, vis_img, f"System Engine: {DEVICE_NAME} | Status: Optimized"

# Custom CSS for Premium Look
custom_css = """
#project-container {
    max-width: 1100px;
    margin: auto;
}
.header-box {
    text-align: center;
    padding: 30px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}
.status-badge {
    padding: 5px 15px;
    background: #10b981;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: bold;
    color: white;
}
.card {
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    padding: 15px;
    background: #f8fafc;
}
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="slate")) as demo:
    with gr.Column(elem_id="project-container"):
        # Header
        with gr.Group(elem_classes="header-box"):
            gr.Markdown("# 🚦 Smart Traffic Sign Analyzer")
            gr.Markdown(f"### Computer Vision Internship Project v2.0 | Status: <span class='status-badge'>Neural Engine Active</span>")
            gr.Markdown(f"Hardware Acceleration: **{DEVICE_NAME} Enabled**")

        with gr.Row():
            # Left Column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Image Acquisition")
                with gr.Tabs():
                    with gr.TabItem("File Upload"):
                        upload_input = gr.Image(type="pil", label="Drop or click to upload", sources=["upload"])
                    with gr.TabItem("Live Capture"):
                        webcam_input = gr.Image(type="pil", label="Webcam snapshot", sources=["webcam"])
                
                predict_btn = gr.Button("🚀 Execute Neural Analysis", variant="primary", size="lg")

            # Middle Column: Analysis Insight
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Model Vision")
                with gr.Group(elem_classes="card"):
                    vision_output = gr.Image(label="Neural Preprocessing View", interactive=False)
                    gr.Markdown("*This view shows how the AI 'sees' the image after grayscale and normalization.*")

            # Right Column: Decision
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Decision Output")
                output_labels = gr.Label(num_top_classes=3, label="Class Probabilities")
                device_info = gr.Textbox(label="Backend Metadata", interactive=False)
                
                with gr.Accordion("Model Architecture Details", open=False):
                    gr.Markdown("""
                        **Backbone**: 6-Layer Deep CNN
                        **Optimizer**: Adam with ReduceLROnPlateau
                        **Preprocessing**: Grayscale (3-channel) + GTSRB Normalization
                        **Target**: 43 European Traffic Sign Categories
                    """)

        # Footer
        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ℹ️ Project Context")
                gr.Markdown("Developing robust inference for autonomous vehicle navigation using the German Traffic Sign Recognition Benchmark (GTSRB).")

    # Event handlers
    predict_btn.click(
        fn=predict,
        inputs=upload_input,
        outputs=[output_labels, vision_output, device_info]
    )
    
    # Auto-predict triggers
    upload_input.change(fn=predict, inputs=upload_input, outputs=[output_labels, vision_output, device_info])
    webcam_input.change(fn=predict, inputs=webcam_input, outputs=[output_labels, vision_output, device_info])

if __name__ == "__main__":
    # In Gradio 6+, css and theme should ideally be in launch or constructor depending on specifics,
    # but the warning said move them to launch.
    demo.launch(css=custom_css)
