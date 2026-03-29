import sys
import tensorflow as tf
from pathlib import Path
from src.exception import CustomException
from src.entity.config import TrainingConfig
from src.utils import create_directories
from src.logger.logging import logging
from src.configeration import ConfigerationManager 
from src.components.callbacks import CallBacks





class Training:
    def __init__(self, config:TrainingConfig):
        """
        Args:
            config: Instance of TrainingConfig containing all training parameters.
        """
        self.config = config
        

    def get_base_model(self):
        """Load and compile the base model."""
        try:
            logging.info("Loading and compiling the base model...")
            self.model = tf.keras.models.load_model(self.config.update_base_model)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.param_learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]
            )
        except Exception as e:
            raise CustomException(e, sys)

    def train_valid_generator(self):
        """Create training and validation datasets using tf.data and modern augmentation."""
        try:
            # Image size: (H, W)
            img_size = tuple(self.config.param_image_size[:-1])
            batch_size = self.config.param_batch_size

            
            train_ds = tf.keras.utils.image_dataset_from_directory(
                self.config.training_data,
                validation_split=0.2,
                subset="training",
                seed=42,
                image_size=img_size,
                batch_size=batch_size,
                shuffle=True,
                label_mode='categorical'
            )

        
            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.config.training_data,
                validation_split=0.2,
                subset="validation",
                seed=42,
                image_size=img_size,
                batch_size=batch_size,
                shuffle=False,
                label_mode='categorical'
            )

            
            # Normalization Layer (replaces rescale=1./255)
            normalization_layer = tf.keras.layers.Rescaling(1./255)

            train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
            val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


            # Data Augmentation (modern version)       
            if self.config.params_augmentation:
                data_augmentation = tf.keras.Sequential([
                    tf.keras.layers.RandomRotation(0.1),
                    tf.keras.layers.RandomTranslation(0.2, 0.2),
                    tf.keras.layers.RandomZoom(0.2),
                    tf.keras.layers.RandomFlip("horizontal")
                ])

                train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

            # ------------------------------
            # Enable Prefetching for Performance
            # ------------------------------
            self.train_data = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            self.val_data = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the trained model to the given path."""
        model.save(str(path))

    def train(self, callbacks: list = None):
        """
        Train the model using the training and validation generators.

        Args:
            callbacks (list, optional): Keras callbacks for training.
        """
        # Calculate steps to cover all samples
        # steps_per_epoch = math.ceil(self.train_data.samples / self.train_data.batch_size)
        # validation_steps = math.ceil(self.val_data.samples / self.val_data.batch_size)
        try:
            steps_per_epoch = self.train_data.cardinality().numpy()
            validation_steps = self.val_data.cardinality().numpy()


            # Train the model
            logging.info("Starting model training...")
            history = self.model.fit(
                self.train_data,
                epochs=self.config.param_epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.val_data,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )

            # Save the trained model to configured path
            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )

            # Optional: Save another copy
            create_directories(["final_model"])
            final_model_dir = Path("final_model")
            self.save_model(path=final_model_dir / "model.keras", model=self.model)

            logging.info("Training completed successfully.")
            return history

        except Exception as e:
            raise CustomException(e, sys)

        
if __name__=="__main__":
        try:
            # Initialize configuration manager
            config = ConfigerationManager()  

            # Prepare callbacks (optional, uncomment if needed)
            callbacks_config = config.get_prepare_callback_config()
            callback_list = CallBacks(config=callbacks_config).get_callbacks()

            # Training setup
            training_config = config.get_training_config()
            trainer = Training(config=training_config)

            # Prepare model and data
            trainer.get_base_model()
            trainer.train_valid_generator()
            trainer.train(callbacks=callback_list)

        except Exception as e:
            raise CustomException (e,sys)