import os
import base64
import gdown
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO
from json import JSONEncoder
from flask_cors import CORS  

# Load Environment Variables
load_dotenv()

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

# Flask App Initialization
app = Flask(__name__)
# Enable CORS for all routes
# Enable CORS with security configurations
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://calorie-estimator-to6k.onrender.com",  # Your frontend domain
            "https://calorie-estimator-to6k.onrender.com/",  # With trailing slash
            "http://localhost:3000",
            "http://127.0.0.1:5000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True,
        "max_age": 600
    }
})

# Additional security headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://calorie-estimator-to6k.onrender.com')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

app.json_encoder = NumpyEncoder

# Environment Switch
ENVIRONMENT = os.getenv("ENV", "local")  # "local" or "production"

# Paths
MODEL_PATH = "food_calorie_model_inceptionv3.h5"
FOOD_LABELS_PATH = "food_labels.txt"
CALORIE_PATH = "calories.csv"

# Step 2: Load Calorie Dataset
try:
    # Read CSV explicitly
    calorie_data = pd.read_csv(
        CALORIE_PATH,
        delimiter=',',
        on_bad_lines='skip',
        encoding='utf-8',
        header=0  # Use the first row as header
    )
    
    # If the columns are incorrect, manually enforce them
    if list(calorie_data.columns) != ['food_label', 'calories']:
        calorie_data.columns = ['food_label', 'calories']
    
    # Create the calorie mapping
    CALORIE_VALUES_DICT = {row['food_label']: row['calories'] for _, row in calorie_data.iterrows()}
    print("Calorie dataset loaded successfully.")
    
except pd.errors.ParserError as e:
    raise Exception(f"Failed to parse calorie dataset: {e}")
except ValueError as e:
    raise Exception(f"ValueError: {e}")
except Exception as e:
    raise Exception(f"Failed to load calorie dataset: {e}")

# Step 3: Load Food Labels
try:
    with open(FOOD_LABELS_PATH, 'r') as f:
        FOOD_LABELS = [line.strip() for line in f.readlines()]
    print("Food labels loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load food labels: {e}")

# Step 4: Load Model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load model: {e}")

# Image Preprocessing
def preprocess_image(image, target_size=(299, 299)):
    try:
        img = Image.open(image).convert('RGB').resize(target_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

# Routes
@app.route('/')
def home():
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
        
        # Get the predicted food label
        food_label = str(FOOD_LABELS[predicted_index])
        
        # Get calories using the food label as the key
        calories = CALORIE_VALUES_DICT.get(food_label, 0)  # Use get() with default value
        confidence = float(predictions[0][predicted_index])
        
        # Print debug information
        print(f"Predicted food label: {food_label}")
        print(f"Calories: {calories}")
        print(f"Confidence: {confidence}")
        print(f"Available food labels in dict: {list(CALORIE_VALUES_DICT.keys())}")
        
        # Return the prediction result
        return jsonify({
            "food": food_label,
            "calories": calories,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error details: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run the App
if __name__ == "__main__":
    app.run(debug=True if ENVIRONMENT == "local" else False)