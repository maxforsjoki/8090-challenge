import joblib
import numpy as np
import sys

# Load ONLY the model
MODEL_PATH = 'src/random_forest_model_unscaled.joblib'
model = joblib.load(MODEL_PATH)

def predict(input_data):
    # The model expects a 2D array, so reshape
    input_np = np.array(input_data).reshape(1, -1)
    
    # Predict directly on the raw, unscaled input
    prediction = model.predict(input_np)
    return prediction[0]

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python3 {sys.argv[0]} <feature1> <feature2> <feature3>")
        sys.exit(1)
        
    features = [float(arg) for arg in sys.argv[1:]]
    reimbursement = predict(features)
    print(reimbursement)