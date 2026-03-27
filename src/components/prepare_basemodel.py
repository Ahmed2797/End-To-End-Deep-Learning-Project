import os,sys
import tensorflow as tf
from pathlib import Path

from src.entity.config import PrepareBasemodelConfig
from src.exception import CustomException
from src.logger.logging import logging
from src.configeration import ConfigerationManager


class PrepareBaseModel:
    def __init__(self, config: PrepareBasemodelConfig):
        try:
            self.config = config
        except Exception as e:
            raise CustomException(e,sys)

    def get_base_model(self):
        try:
            self.model = tf.keras.applications.VGG16(
                include_top=self.config.param_include_top,
                weights=self.config.param_weight,
                input_shape=self.config.param_image_size,
                classes=self.config.param_classes,
                classifier_activation='softmax'
            )

            logging.info("Base model (VGG16) loaded successfully")

            self.save_model(
                file_path=self.config.base_model,
                model=self.model
            )

        except Exception as e:
            raise CustomException(e,sys)

    def update_base_model(self):
        try:
            self.full_model = self.prepare_model_layers(
                model=self.model,
                num_classes=self.config.param_classes,
                freeze_all=True,
                freeze_till=None,
                learning_rate=self.config.param_learning_rate
            )

            self.save_model(
                file_path=self.config.update_base_model,
                model=self.full_model
            )

            logging.info("Updated model created and saved")

        except Exception as e:
            raise CustomException(e,sys)

    @staticmethod
    def save_model(file_path: Path, model: tf.keras.Model):
        model.save(file_path)
        logging.info(f"Model saved at: {file_path}")

    @staticmethod
    def prepare_model_layers(
        model: tf.keras.Model,
        num_classes: int,
        learning_rate: float,
        freeze_all: bool = False,
        freeze_till: int = None
    ) -> tf.keras.Model:
        """
        Freeze layers and add custom classifier
        """

        # Freeze all layers
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False

        # Freeze partial layers
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Custom Head
        x = model.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        # Final model
        full_model = tf.keras.Model(inputs=model.input, outputs=output)

        # Compile
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()

        return full_model
    



if __name__=="__main__":
    try:
        config = ConfigerationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
    except Exception as e:
        raise CustomException(e, sys)
        
