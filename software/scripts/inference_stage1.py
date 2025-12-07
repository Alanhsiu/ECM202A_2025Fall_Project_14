import torch
from PIL import Image
from torchvision import transforms
import os
import sys

# --- 1. System Setup and Model Import ---
# Add the project root to the path to allow model imports
# This ensures that 'models.gesture_classifier' can be found.
sys.path.append('.')

# Import the Stage 1 Model (GestureClassifier)
try:
    from models.gesture_classifier import GestureClassifier
except ImportError:
    print("Error: Could not import GestureClassifier. Ensure file structure is correct.")
    sys.exit(1)


# --- 2. Define Preprocessing Transforms ---
def get_transforms():
    """Defines the necessary image transformation pipelines for ViT inputs."""
    # Standard normalization values for pre-trained ViT models
    RGB_NORM_MEAN = [0.485, 0.456, 0.406]
    RGB_NORM_STD = [0.229, 0.224, 0.225]
    
    # 1. RGB Transform: Resize, Convert to Tensor, Normalize
    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_NORM_MEAN, std=RGB_NORM_STD)
    ])

    # 2. Depth Transform: Resize, Convert to Tensor, Repeat to 3 Channels
    # ViT expects 3 channels. Depth is single-channel grayscale (L), so we repeat it.
    depth_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) 
    ])
    
    return rgb_transform, depth_transform

def load_and_preprocess_image(rgb_path, depth_path):
    """Loads and applies transformations to RGB and Depth images."""
    
    # Load images
    rgb_image = Image.open(rgb_path).convert('RGB')
    # Depth is converted to single channel grayscale ('L')
    depth_image = Image.open(depth_path).convert('L')
    
    # Apply transforms and add batch dimension (unsqueeze(0))
    rgb_transform, depth_transform = get_transforms()
    rgb_tensor = rgb_transform(rgb_image).unsqueeze(0)
    depth_tensor = depth_transform(depth_image).unsqueeze(0)
    
    return rgb_tensor, depth_tensor


# --- 3. Inference Function ---

def run_stage1_inference(rgb_tensor, depth_tensor, model, device):
    """Runs inference for the Stage 1 model."""
    
    # Set model to evaluation mode (crucial: turns off dropout, fixes layer allocation)
    model.eval()
    
    # In Stage 1 eval mode, all 12 layers of both backbones are actively used.
    # The allocation mask is fixed to all ones (keep all).
    FIXED_ALLOCATION = [1] * 12 # 12 layers
    
    # Prepare data
    rgb_tensor = rgb_tensor.to(device)
    depth_tensor = depth_tensor.to(device)
    
    # Gesture classes (based on scripts/train_stage1.py)
    gesture_classes = ['standing', 'left_hand', 'right_hand', 'both_hands']
    
    with torch.no_grad():
        # The GestureClassifier forward() method executes all 12 layers in eval mode.
        logits = model(rgb_tensor, depth_tensor)
        
        # Get prediction index
        _, predicted_index = torch.max(logits, 1)
        
    predicted_gesture = gesture_classes[predicted_index.item()]

    return predicted_gesture, FIXED_ALLOCATION, FIXED_ALLOCATION


# --- 4. Main Execution Block ---

if __name__ == "__main__":
    
    # --- Configuration ---
    # NOTE: You MUST replace this path with the location of your trained model checkpoint.
    CHECKPOINT_PATH = 'checkpoints/stage1/best_model.pth' 
    
    # Create DUMMY files if they don't exist for code execution structure
    DUMMY_RGB_PATH = '/home/b09901066/ADMN-RealWorld/data/low_light/left_hand/color_image_0001.png'
    DUMMY_DEPTH_PATH = '/home/b09901066/ADMN-RealWorld/data/low_light/left_hand/depth_image_0001.png'

    if not os.path.exists(DUMMY_RGB_PATH):
        Image.new('RGB', (224, 224), color = 'red').save(DUMMY_RGB_PATH)
    if not os.path.exists(DUMMY_DEPTH_PATH):
        Image.new('L', (224, 224), color = 'gray').save(DUMMY_DEPTH_PATH)
    
    print(f"Using dummy input files: {DUMMY_RGB_PATH} and {DUMMY_DEPTH_PATH}")
    
    # --- Model Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Stage 1 Model Configuration (12 ViT layers for each modality)
    model = GestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        vision_vit_layers=12,
        depth_vit_layers=12,
        pretrained_path=None # Only relevant for initial training setup
    ).to(device)

    # Load trained weights
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Successfully loaded Stage 1 model weights from {CHECKPOINT_PATH}.")
    except FileNotFoundError:
        print(f"WARNING: Model weights not found at {CHECKPOINT_PATH}.")
        print("Results will be random as the model uses untrained weights.")
    except Exception as e:
        print(f"Error loading weights: {e}. Check if the checkpoint structure is correct.")


    # --- Preprocessing and Inference ---
    rgb_tensor, depth_tensor = load_and_preprocess_image(DUMMY_RGB_PATH, DUMMY_DEPTH_PATH)
    
    predicted_gesture, rgb_alloc, depth_alloc = run_stage1_inference(
        rgb_tensor, depth_tensor, model, device
    )

    # --- Print Results ---
    print("\n--- STAGE 1 INFERENCE RESULTS (BASELINE) ---")
    print(f"Input RGB File: {DUMMY_RGB_PATH}")
    print(f"Input Depth File: {DUMMY_DEPTH_PATH}")
    print("-" * 35)
    print(f"Predicted Gesture: {predicted_gesture}")
    print(f"RGB Layers Allocated: {sum(rgb_alloc)} / 12 (Fixed)")
    print(f"Depth Layers Allocated: {sum(depth_alloc)} / 12 (Fixed)")
    print(f"Combined Total Layers: {sum(rgb_alloc) + sum(depth_alloc)} / 24")
    print(f"RGB Allocation Mask: {rgb_alloc}")
    print(f"Depth Allocation Mask: {depth_alloc}")
    print("\nNOTE: In Stage 1, the layer allocation is fixed to 12 layers for both modalities (i.e., [1] * 12).")