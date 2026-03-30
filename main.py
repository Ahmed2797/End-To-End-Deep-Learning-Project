import matplotlib.pyplot as plt
from src.components.inferance import Prediction_Pipeline



if __name__ == "__main__":
    # Demonstration of the prediction pipeline
    try:
        pipeline = Prediction_Pipeline(model_path="final_model/model.keras")
        test_img_path = "artifacts/data_ingestion/brain_tumor_dataset/yes/Y1.jpg" 
        
        orig, _, result = pipeline.predict(test_img_path)
        # print(f"Result:{result}")

        # Visualization setup
        plt.figure(figsize=(16, 8))

        plt.subplot(2, 2, 1)
        plt.title("Original Tumar X-ray", fontsize=8)
        plt.imshow(orig)
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title("Detection Result (Bounding Box)", fontsize=8)
        plt.imshow(result)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Prediction failed with error: {e}")