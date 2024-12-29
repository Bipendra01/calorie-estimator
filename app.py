import os
import base64
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
from json import JSONEncoder

# -------------------------------
# Custom JSON Encoder
# -------------------------------
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# -------------------------------
# Flask App Initialization
# -------------------------------
app = Flask(__name__)
app.json_encoder = NumpyEncoder

# -------------------------------
# Paths and Configuration
# -------------------------------
MODEL_PATH = "food_calorie_model_inceptionv3.h5"
DATASET_PATH = "dummyDataSet/images"
CALORIE_CSV_PATH = "calories.csv"

# Load Food Labels
try:
    FOOD_LABELS = sorted(os.listdir(DATASET_PATH))
except FileNotFoundError:
    raise Exception(f"Dataset path '{DATASET_PATH}' not found.")

# Load Calorie Mapping
try:
    calorie_data = pd.read_csv(CALORIE_CSV_PATH)
    CALORIE_VALUES = [calorie_data.loc[calorie_data['food_label'] == label, 'calories'].values[0] if
                     label in calorie_data['food_label'].values else 200 for label in FOOD_LABELS]
except Exception as e:
    raise Exception(f"Error loading calorie data: {e}")

# Load the Trained Model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise Exception(f"Failed to load model: {e}")

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image, target_size=(299, 299)):
    try:
        img = Image.open(image).convert('RGB').resize(target_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check for Base64 image data
    if request.is_json:
        data = request.get_json()
        if 'image_data' not in data:
            return jsonify({"error": "No image data found in request"}), 400
        try:
            # Decode Base64 image
            image_data = base64.b64decode(data['image_data'])
            image = BytesIO(image_data)
        except Exception as e:
            return jsonify({"error": f"Invalid Base64 data: {e}"}), 400
    
    # Check for file upload
    elif 'image' in request.files:
        file = request.files['image']
        image = file
    else:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Predict using the model
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions)
        
        # Print predictions for debugging
        print(f"Predictions: {predictions}")
        print(f"Predicted Index: {predicted_index}")
        
        food_label = str(FOOD_LABELS[predicted_index])  # Convert to string
        calories = int(CALORIE_VALUES[predicted_index])  # Convert to Python int
        confidence = float(predictions[0][predicted_index])  # Convert to Python float
        
        # Return the prediction result
        return jsonify({
            "food": food_label,
            "calories": calories,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error details: {str(e)}")  # Added for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)