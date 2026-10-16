
import sys

import pandas as pd
import tensorflow as tf

from brain_tumor_detection.entity.config_entity import ModelTrainingConfig
from brain_tumor_detection.exception import CustomException
from brain_tumor_detection.logger import logger
from brain_tumor_detection.utils.common import create_directory


class ModelTrainer:

    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def display_config(self):

        print("Model Training Configuration")
        print("-" * 40)

        print(f"Model Name      : {self.config.model_name}")
        print(f"Image Size      : {self.config.image_size}")
        print(f"Batch Size      : {self.config.batch_size}")
        print(f"Epochs          : {self.config.epochs}")
        print(f"Learning Rate   : {self.config.learning_rate}")
        print(f"Include Top     : {self.config.include_top}")
        print(f"Weights         : {self.config.weights}")
        print(f"Classes         : {self.config.classes}")
        print(f"Color Mode      : {self.config.color_mode}")
        print(f"Normalize       : {self.config.normalize}")

    def train(self):

        try:
            create_directory(self.config.root_dir)

            train_ds = tf.keras.utils.image_dataset_from_directory(
                self.config.train_dir,
                validation_split=0.2,
                subset="training",
                seed=42,
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                color_mode=self.config.color_mode
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.config.train_dir,
                validation_split=0.2,
                subset="validation",
                seed=42,
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                color_mode=self.config.color_mode
            )

            logger.info(f"Dataset classes: {train_ds.class_names}")

            if train_ds.class_names != self.config.class_names:
                raise ValueError(
                    f"Expected class order {self.config.class_names}, "
                    f"found {train_ds.class_names}"
                )

            if len(train_ds.class_names) != self.config.classes:
                raise ValueError(
                    f"Expected {self.config.classes} classes, "
                    f"found {len(train_ds.class_names)}"
                )

            base_model = tf.keras.applications.ResNet50(
                include_top=self.config.include_top,
                weights=self.config.weights,
                input_shape=(*self.config.image_size, 3)
            )

            base_model.trainable = True

            for layer in base_model.layers[:-30]:
                layer.trainable = False

            inputs = tf.keras.Input(
                shape=(*self.config.image_size, 3)
            )

            x = tf.keras.layers.Lambda(
                tf.keras.applications.resnet50.preprocess_input,
                name="resnet_preprocessing"
            )(inputs)

            x = base_model(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            )(x)
            x = tf.keras.layers.Dropout(0.3)(x)

            outputs = tf.keras.layers.Dense(
                self.config.classes,
                activation="softmax"
            )(x)

            model = tf.keras.Model(inputs, outputs)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.config.learning_rate
                ),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=2
                )
            ]

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.epochs,
                callbacks=callbacks
            )

            model.save(self.config.trained_model_path)

            history_path = (
                self.config.root_dir / "finetune_candidate_history.csv"
            )

            pd.DataFrame(history.history).to_csv(
                history_path,
                index=False
            )

            logger.info(
                f"Candidate model saved to {self.config.trained_model_path}"
            )
            logger.info(f"Training history saved to {history_path}")

            return model

        except Exception as e:
            raise CustomException(e, sys)
