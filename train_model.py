import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# -------------------------------
# � Dataset Paths
# -------------------------------
TRAIN_DIR = 'dummyDataSet/images'
VAL_DIR = 'dummyDataSet/images'
CALORIE_CSV = 'calories.csv'
MODEL_SAVE_PATH = 'food_calorie_model_inceptionv3.h5'
# -------------------------------
# � Image Preprocessing
# -------------------------------
IMG_SIZE = (299, 299) # InceptionV3 expects 299x299 images
BATCH_SIZE = 32
datagen = ImageDataGenerator(
 rescale=1.0/255.0,
 rotation_range=20,
 width_shift_range=0.2,
 height_shift_range=0.2,
 horizontal_flip=True,
 validation_split=0.2
)
train_data = datagen.flow_from_directory(
 TRAIN_DIR,
 target_size=IMG_SIZE,
 batch_size=BATCH_SIZE,
 class_mode='categorical',
 subset='training'
)
val_data = datagen.flow_from_directory(
 VAL_DIR,
 target_size=IMG_SIZE,
 batch_size=BATCH_SIZE,
 class_mode='categorical',
 subset='validation'
)
# -------------------------------
# � Map Labels to Calories
# -------------------------------
calories_df = pd.read_csv(CALORIE_CSV)
label_to_calories = dict(zip(calories_df['food_label'], calories_df['calories']))
# -------------------------------
# � Build the Model with InceptionV3
# -------------------------------
# Load Pre-trained InceptionV3 (Exclude Top Layers)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
# Add Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x) # Flatten the output
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_data.class_indices), activation='softmax')(x)
# Combine into Final Model
model = Model(inputs=base_model.input, outputs=predictions)
# Freeze Base Layers (Initial Training Phase)
for layer in base_model.layers:
 layer.trainable = False
# Compile the Model
model.compile(
 optimizer='adam',
 loss='categorical_crossentropy',
 metrics=['accuracy']
)
# Display Model Summary
model.summary()
# -------------------------------
# � Train the Model (Phase 1)
# -------------------------------
history = model.fit(
 train_data,
 validation_data=val_data,
 epochs=10,
 batch_size=BATCH_SIZE
)
# -------------------------------
# � Fine-Tune the Model (Phase 2)
# -------------------------------
# Unfreeze the Top Layers of the Base Model
for layer in base_model.layers[-30:]:
 layer.trainable = True
# Recompile with a Smaller Learning Rate
model.compile(
 optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
 loss='categorical_crossentropy',
 metrics=['accuracy']
)
# Fine-tune the Model
history_fine = model.fit(
 train_data,
 validation_data=val_data,
 epochs=5,
 batch_size=BATCH_SIZE
)
# -------------------------------
# � Save the Model
# -------------------------------
model.save(MODEL_SAVE_PATH)
print(f"✅ Model training complete and saved as '{MODEL_SAVE_PATH}'")