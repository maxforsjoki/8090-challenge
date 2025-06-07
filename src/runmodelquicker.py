import torch
import torch.nn as nn
import joblib
import numpy as np
import sys
import json # <-- Import the json library

# --- Step 1: Re-define the EXACT same model architecture ---
# This class must be identical to the one you used for training.
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

# Set the model to evaluation mode
model.eval()

# Load the scaler
scaler = joblib.load(SCALER_PATH)


# --- Step 3: Process input data and make a prediction ---
# This function is unchanged.

def predict(input_data):
    """
    Takes a list or numpy array of 3 raw input values, scales them,
    and returns a prediction from the model.
    """
    raw_input_np = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(raw_input_np)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    
    with torch.no_grad():
        predicted_value = model(input_tensor)
        
    return predicted_value.item()

# --- Step 4: Main execution block for JSON processing ---
# This block is NEW. It reads a JSON file and writes to a TXT file.

if __name__ == "__main__":
    # Define the input JSON and output TXT file paths
    INPUT_JSON_PATH = 'private_cases.json'
    OUTPUT_TXT_PATH = 'reimbursement_predictions.txt'

    try:
        # Open and load the JSON data
        with open(INPUT_JSON_PATH, 'r') as f_in:
            data = json.load(f_in)

        # Open the output file for writing
        with open(OUTPUT_TXT_PATH, 'w') as f_out:
            # Loop through each record in the JSON file
            for record in data:
                # IMPORTANT: Ensure the order of features matches the model's training order
                features = [
                    record['trip_duration_days'],
                    record['miles_traveled'],
                    record['total_receipts_amount']
                ]
                
                # Get the prediction for the current record
                reimbursement = predict(features)
                
                # Write the result to the output file, followed by a newline
                f_out.write(f"{reimbursement}\n")
        
        print(f"✅ Success! Predictions have been saved to {OUTPUT_TXT_PATH}")

    except FileNotFoundError:
        print(f"Error: The input file was not found at '{INPUT_JSON_PATH}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{INPUT_JSON_PATH}'. Please check its format.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing expected key in JSON data: {e}")
        sys.exit(1)