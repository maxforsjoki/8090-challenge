import torch
import torch.nn as nn
import joblib
import numpy as np
import sys

# --- Step 1: Re-define the EXACT same model architecture ---
# This class must be identical to the one you used for training.
# The script needs this blueprint to know how to structure the loaded weights.
class SuperComplexModel(nn.Module):
    def __init__(self, input_features):
        super(SuperComplexModel, self).__init__()
        self.layer1 = nn.Linear(input_features, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64,32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.output_layer(x)
        return x

# --- Step 2: Define paths and load the model and scaler ---

# Define the paths to your saved files
MODEL_STATE_PATH = 'src/model_state.pth'
SCALER_PATH = 'src/scaler.joblib'
INPUT_FEATURES = 3 # The number of X values your model expects

# Initialize the model architecture
model = SuperComplexModel(INPUT_FEATURES)

# Load the saved state dictionary into the model
model.load_state_dict(torch.load(MODEL_STATE_PATH))

# Set the model to evaluation mode. This is CRITICAL for inference.
# It disables layers like Dropout and BatchNorm.
model.eval()

# Load the scaler
scaler = joblib.load(SCALER_PATH)


# --- Step 3: Process command-line arguments and make a prediction ---

def predict(input_data):
    """
    Takes a list or numpy array of 3 raw input values, scales them,
    and returns a prediction from the model.
    """
    # The scaler expects a 2D array, so we reshape our 1D input
    raw_input_np = np.array(input_data).reshape(1, -1)
    
    # Scale the input using the loaded scaler
    scaled_input = scaler.transform(raw_input_np)
    
    # Convert the scaled data to a PyTorch tensor
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    
    # Make a prediction
    # We use torch.no_grad() to ensure no gradients are calculated, saving memory and computation
    with torch.no_grad():
        predicted_value = model(input_tensor)
        
    # The model outputs a tensor, so we extract the single value from it
    return predicted_value.item()

# Main execution block
if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <feature1> <feature2> <feature3>")
        sys.exit(1)
        
    try:
        # Convert command-line arguments from strings to floats
        features = [float(arg) for arg in sys.argv[1:]]
        
        # Get the prediction
        reimbursement = predict(features)
        
        # Print the final result
        print(reimbursement)
        
    except ValueError:
        print("Error: All three arguments must be numbers.")
        sys.exit(1)