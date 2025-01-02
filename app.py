import os
import gdown
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Flask App Initialization
app = Flask(__name__)

# Environment Switch
ENVIRONMENT = os.getenv("ENV", "local")  # "local" or "production"

# Paths
os.makedirs('data', exist_ok=True)
MODEL_PATH = "food_calorie_model_inceptionv3.h5"
CALORIE_PATH = "data/calories.csv"
FOOD_LABELS_PATH = "data/food_labels.txt"

# Google Drive URLs from Environment Variables
MODEL_URL = os.getenv("MODEL_URL")
CALORIE_DATASET_URL = os.getenv("CALORIE_DATASET_URL")
FOOD_LABELS_URL = os.getenv("FOOD_LABELS_URL")

# Function to Download Files Dynamically
def download_file_if_missing(url, local_path, file_description):
    if not os.path.exists(local_path):
        print(f"Downloading {file_description} from Google Drive...")
        gdown.download(url, local_path, quiet=False)
        print(f"{file_description} downloaded successfully.")
    else:
        print(f"{file_description} already exists locally.")

# Step 1: Download Files
download_file_if_missing(MODEL_URL, MODEL_PATH, "Model File")
download_file_if_missing(CALORIE_DATASET_URL, CALORIE_PATH, "Calorie Dataset")
download_file_if_missing(FOOD_LABELS_URL, FOOD_LABELS_PATH, "Food Labels File")

# Step 2: Load Calorie Dataset
try:
    # Use 'on_bad_lines' instead of 'error_bad_lines'
    calorie_data = pd.read_csv(CALORIE_PATH, delimiter=',', on_bad_lines='skip', encoding='utf-8')
    CALORIE_VALUES_DICT = {row['food_label']: row['calories'] for _, row in calorie_data.iterrows()}
    print("Calorie dataset loaded successfully.")
except pd.errors.ParserError as e:
    raise Exception(f"Failed to parse calorie dataset: {e}")
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
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)
    
    try:
        processed_image = preprocess_image(file_path)
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions)
        food_label = FOOD_LABELS[predicted_index]
        calories = CALORIE_VALUES_DICT.get(food_label, 200)
        confidence = float(predictions[0][predicted_index])
        
        return jsonify({
            "food": food_label,
            "calories": calories,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# Run the App
if __name__ == "__main__":
    app.run(debug=True if ENVIRONMENT == "local" else False)