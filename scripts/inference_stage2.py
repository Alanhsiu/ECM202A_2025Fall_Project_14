import torch
from PIL import Image
from torchvision import transforms
import os
import sys
import numpy as np

# --- 1. System Setup and Model Import ---
# Add the project root to the path to allow model imports
sys.path.append('.')

# Import the Stage 2 Model (AdaptiveGestureClassifier)
try:
    from models.adaptive_controller import AdaptiveGestureClassifier
except ImportError:
    print("Error: Could not import AdaptiveGestureClassifier. Ensure file structure is correct.")
    sys.exit(1)


# --- 2. Define Preprocessing Transforms (Reused from Stage 1) ---
def get_transforms():
    """Defines the necessary image transformation pipelines for ViT inputs."""
    RGB_NORM_MEAN = [0.485, 0.456, 0.406]
    RGB_NORM_STD = [0.229, 0.224, 0.225]
    
    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_NORM_MEAN, std=RGB_NORM_STD)
    ])

    depth_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Repeat to 3 channels as required by ViT
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) 
    ])
    
    return rgb_transform, depth_transform

def load_and_preprocess_image(rgb_path, depth_path):
    """Loads and applies transformations to RGB and Depth images."""
    
    rgb_image = Image.open(rgb_path).convert('RGB')
    depth_image = Image.open(depth_path).convert('L')
    
    rgb_transform, depth_transform = get_transforms()
    # Add batch dimension (unsqueeze(0))
    rgb_tensor = rgb_transform(rgb_image).unsqueeze(0)
    depth_tensor = depth_transform(depth_image).unsqueeze(0)
    
    return rgb_tensor, depth_tensor


# --- 3. Inference Function ---

def run_stage2_inference(rgb_tensor, depth_tensor, model, device):
    """Runs inference for the Stage 2 Adaptive Controller model."""
    
    # Set model to evaluation mode (crucial for deterministic layer selection)
    model.eval()
    
    # Gesture classes
    gesture_classes = ['standing', 'left_hand', 'right_hand', 'both_hands']
    
    # Prepare data
    rgb_tensor = rgb_tensor.to(device)
    depth_tensor = depth_tensor.to(device)
    
    # --- The Core Adaptive Call ---
    # We set temperature low (or 1.0) during eval; the hard selection is deterministic.
    with torch.no_grad():
        # model() returns (logits, layer_allocation)
        logits, allocation = model(
            rgb_tensor, 
            depth_tensor, 
            temperature=1.0, 
            return_allocation=True
        )
        
        # Get prediction
        _, predicted_index = torch.max(logits, 1)
        
    predicted_gesture = gesture_classes[predicted_index.item()]

    # Process allocation: [batch, 2 (RGB/Depth), 12 (Layers)]
    # Squeeze to remove batch dimension (since batch_size=1)
    allocation = allocation.squeeze(0).cpu().numpy()
    
    # Allocation mask is the row of 1s and 0s
    rgb_mask = allocation[0].round().astype(int).tolist()
    depth_mask = allocation[1].round().astype(int).tolist()

    return predicted_gesture, rgb_mask, depth_mask


# --- 4. Main Execution Block ---

if __name__ == "__main__":
    
    # --- Configuration ---
    # NOTE: 1. You MUST replace these paths with your trained model checkpoints.
    # NOTE: 2. total_layers must match the budget the controller was trained for (e.g., 12 or 8).
    
    TOTAL_LAYER_BUDGET = 12 
    CHECKPOINT_PATH_STAGE1 = 'checkpoints/stage1/best_model.pth' 
    CHECKPOINT_PATH_STAGE2 = 'checkpoints/stage2/best_controller_12layers.pth' 
    
    DUMMY_RGB_PATH = '/home/b09901066/ADMN-RealWorld/data/low_light/left_hand/color_image_0001.png'
    DUMMY_DEPTH_PATH = '/home/b09901066/ADMN-RealWorld/data/low_light/left_hand/depth_image_0001.png'

    if not os.path.exists(DUMMY_RGB_PATH):
        Image.new('RGB', (224, 224), color = 'red').save(DUMMY_RGB_PATH)
    if not os.path.exists(DUMMY_DEPTH_PATH):
        Image.new('L', (224, 224), color = 'gray').save(DUMMY_DEPTH_PATH)
    
    print(f"Using dummy input files: {DUMMY_RGB_PATH} and {DUMMY_DEPTH_PATH}")
    
    # --- Model Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Adaptive Controller Model Initialization
    # We pass STAGE 1 path for the model structure/initialization strategy
    model = AdaptiveGestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        total_layers=TOTAL_LAYER_BUDGET,
        qoi_dim=128,
        stage1_checkpoint=CHECKPOINT_PATH_STAGE1 # Used to set model structure/freeze layers
    ).to(device)

    # Load Stage 2 Controller weights (overwrites initial weights)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH_STAGE2, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Successfully loaded Stage 2 model weights from {CHECKPOINT_PATH_STAGE2}.")
    except FileNotFoundError:
        print(f"WARNING: Controller weights not found at {CHECKPOINT_PATH_STAGE2}.")
        print("Model uses frozen Stage 1 weights and an untrained controller.")
    except Exception as e:
        print(f"Error loading Stage 2 weights: {e}. Check if the checkpoint structure is correct.")


    # --- Preprocessing and Inference ---
    rgb_tensor, depth_tensor = load_and_preprocess_image(DUMMY_RGB_PATH, DUMMY_DEPTH_PATH)
    
    predicted_gesture, rgb_mask, depth_mask = run_stage2_inference(
        rgb_tensor, depth_tensor, model, device
    )

    # --- Print Results ---
    rgb_layers_used = sum(rgb_mask)
    depth_layers_used = sum(depth_mask)
    
    print("\n--- STAGE 2 ADAPTIVE INFERENCE RESULTS ---")
    print(f"Total Budget (Fixed): {TOTAL_LAYER_BUDGET} Layers")
    print(f"Predicted Gesture: {predicted_gesture}")
    print("-" * 35)
    print(f"RGB Layers Allocated: {rgb_layers_used} / 12")
    print(f"Depth Layers Allocated: {depth_layers_used} / 12")
    print(f"Combined Total Layers: {rgb_layers_used + depth_layers_used} / 24")
    print(f"Target Total Executed: {rgb_layers_used + depth_layers_used} / {TOTAL_LAYER_BUDGET} (Should match budget)")
    print("-" * 35)
    print(f"RGB Allocation Mask: {rgb_mask}")
    print(f"Depth Allocation Mask: {depth_mask}")