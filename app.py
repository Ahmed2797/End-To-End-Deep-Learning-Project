from src.pipeline import TrainingPipeline 
from src.exception import CustomException
import sys


if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run()
    except Exception as e:
        raise CustomException(e, sys)
