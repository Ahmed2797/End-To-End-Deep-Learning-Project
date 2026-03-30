import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

class Prediction_Pipeline:
    """
    Inference Pipeline for Pneumothorax detection.
    
    This class handles the end-to-end process of loading a trained segmentation model,
    preprocessing input chest X-rays, and generating both binary masks and 
    bounding box visualizations for medical analysis.
    """

    def __init__(self, model_path: str):
        """
        Initializes the prediction engine by loading the Keras model.

        Args:
            model_path (str): Path to the saved .keras or .h5 model file.
        """
        # Load model with custom loss and metric functions
        self.model = tf.keras.models.load_model(model_path, compile=False)


    def preprocess_image(self, image_path: str) -> tuple:
        """
        Reads, color-corrects, and normalizes an image for the model.

        Args:
            image_path (str): Local path to the X-ray image file.

        Returns:
            tuple: (Original RGB image, Normalized 4D tensor, Original image dimensions)
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2] # (Height, Width)
        
        # Resize to 256x256 as per model training requirements
        img_resized = cv2.resize(img, (224, 224))
        # Normalize pixel values to [0, 1] range
        img_normalized = img_resized / 255.0
        # Expand dimensions to create a batch (1, 224, 224, 3)
        img_final = np.expand_dims(img_normalized, axis=0)
        return img, img_final, original_shape

    def predict(self, image_path: str, threshold: float = 0.60):
        """
        Performs model inference and generates detection results.
        """
        original_img, processed_img, (h, w) = self.preprocess_image(image_path)
        
        # 1. Run model prediction
        prediction = self.model.predict(processed_img)[0]
        prediction = np.squeeze(prediction) # Crucial: Convert (224, 224, 1) to (224, 224)
        
        print(f"Max Confidence: {np.max(prediction):.4f}")
        print(f"Min Confidence: {np.min(prediction):.4f}")
        
        # 2. Generate binary pred and resize to ORIGINAL resolution FIRST
        prediction_image = (prediction > threshold).astype(np.uint8)
        prediction_image = cv2.resize(prediction_image, (w, h), interpolation=cv2.INTER_NEAREST)

        # 3. Detect contours on the RESIZED pred (matching original image dimensions)
        contours, _ = cv2.findContours(prediction_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output_img = original_img.copy()
        found_count = 0
        
        for cnt in contours:
            # Scale the area filter relative to image size
            if cv2.contourArea(cnt) > (w * h * 0.01): # Filter objects smaller than 0.1% of image
                found_count += 1
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Draw on the original sized image
                cv2.rectangle(output_img, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
                # cv2.drawContours(output_img, [cnt], -1, (255, 0, 0), 2)
                label = f"Tumar: {np.max(prediction)*100:.1f}%"
                cv2.putText(output_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        print(f"Detected {found_count} regions.")
        return original_img, prediction_image, output_img

if __name__ == "__main__":
    # Demonstration of the prediction pipeline
    try:
        pipeline = Prediction_Pipeline(model_path="final_model/model.keras")
        test_img_path = "artifacts/data_ingestion/brain_tumor_dataset/yes/Y254.jpg" 
        
        orig, pred, result = pipeline.predict(test_img_path)

        # Visualization setup
        plt.figure(figsize=(16, 10))

        plt.subplot(2, 2, 1)
        plt.title("Original Tumar X-ray", fontsize=12)
        plt.imshow(orig)
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title("Detection Result (Bounding Box)", fontsize=12)
        plt.imshow(result)
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Prediction failed with error: {e}")
