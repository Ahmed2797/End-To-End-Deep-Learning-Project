import tensorflow as tf
import numpy as np
import os
import keras

"""
Module for image-based tumor detection using deep learning models.
"""

class ImagePredictor:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            # compile=False is used for inference to avoid version mismatch errors
            self.model = keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Fallback loading triggered: {e}")
            self.model = tf.keras.models.load_model(model_path, compile=False)

    def preprocess_image(self, img_path: str, target_size=(224, 224)):
        """
        Loads, resizes, and normalizes the image.
        """
        img = keras.utils.load_img(img_path, target_size=target_size)
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # NOTE: Ensure 1/255 matches your training. 
        # If the model still performs poorly, try: tf.keras.applications.vgg16.preprocess_input
        img_array = img_array / 255.0 

        return img_array

    def predict(self, img_path: str):
        """
        Infers the class from the image. 
        """
        img_array = self.preprocess_image(img_path)
        predictions = self.model.predict(img_array, verbose=0)
        
        # --- DEBUGGING: Uncomment the line below to see the raw numbers in your terminal ---
        # print(f"Raw Model Output: {predictions}")

        if predictions.shape[1] == 1:
            # Binary Logic: Usually 0 = Normal, 1 = Tumor
            confidence = float(predictions[0][0])
            if confidence > 0.5:
                label = "Tumor"
                display_confidence = confidence
            else:
                label = "Normal"
                display_confidence = 1.0 - confidence
        else:
            # Categorical Logic: Mapping indices to labels
            idx = np.argmax(predictions[0])
            conf = float(predictions[0][idx])
            
            # CRITICAL CHECK: 
            # In Keras, folder 'Normal' comes before 'Tumor' alphabetically.
            # Index 0 = Normal, Index 1 = Tumor.
            # If your results are swapped, change this to ["Tumor", "Normal"]
            classes = ["Normal", "Tumor"] 
            
            label = classes[idx]
            display_confidence = conf

        return label, display_confidence
