#!/usr/bin/python3
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Rescaling,
                                     Flatten, Dense, Dropout, Input)
import os
import sys
from Transformation import Transformation
from helpers.directories import iter_dir_image
import random
import shutil
from tensorflow.keras.optimizers import Adam
import json


class Train:
    def __init__(self, path):
        random.seed(42)
        self.training_path = 'learnings/dataset/train'
        self.validation_path = 'learnings/dataset/validation'
        self.save_dest = 'learnings/model.keras'
        self.validation_pct = 0.05
        self.source_dir = path
        self.model = Sequential([
            Input(shape=(255, 255, 3)),
            Rescaling(1./255),
            Conv2D(filters=16, kernel_size=4, activation='relu'),
            MaxPooling2D(),
            Conv2D(filters=32, kernel_size=4, activation='relu'),
            MaxPooling2D(),
            Dropout(0.1),
            Conv2D(filters=64, kernel_size=4, activation='relu'),
            MaxPooling2D(),
            Dropout(0.1),
            Conv2D(filters=128, kernel_size=4, activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=8, activation='softmax')
            ])
        iter_dir_image(path, self.copy_image)
        self.dataset = keras.utils.image_dataset_from_directory(
            self.training_path,
            labels='inferred',
            label_mode="categorical",
            color_mode="rgb",
            shuffle=True,
            seed=42,
            validation_split=.2,
            image_size=(255, 255),
            verbose=True,
            subset='both'
        )
        self.classes = self.dataset[0].class_names

    def copy_image(self, filename):
        """
        perform the transformation in the image.
        infers save or show from the content of dst_dir
        """
        is_validation = random.random() < self.validation_pct
        dst_dir = self.validation_path if is_validation else self.training_path
        partial_path = os.path.relpath(filename, self.source_dir)
        partial_path = os.path.dirname(partial_path)
        partial_path = os.path.normpath(partial_path)
        partial_path = list(os.path.split(partial_path))
        if len(partial_path) > 1:
            del partial_path[-2]
        partial_path = os.sep.join(partial_path)
        dst_dir = os.path.join(dst_dir, partial_path)
        os.makedirs(dst_dir, exist_ok=True)
        destination = dst_dir
        if is_validation:
            shutil.copyfile(
                filename,
                os.path.join(destination, os.path.basename(filename)))
        else:
            trans = Transformation(filename)
            trans.transform(options=['mask'], dst_dir=destination)

    def compile_model(self, optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      ):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, epochs=30):
        self.model.fit(self.dataset[0], validation_data=self.dataset[1],
                       epochs=epochs)

    def summary(self):
        self.model.summary()

    def save(self):
        with open('learnings/classes.json', 'w') as f:
            json.dump(self.classes, f)
        self.model.save(self.save_dest)

    def load(self):
        return tf.keras.models.load_model(self.save_dest)

    def evaluate(self):
        model = self.load()
        loss, acc = model.evaluate(self.dataset[1], verbose=2)
        print("model, accuracy: {:5.2f}%".format(100 * acc))

    def run(self, optimizer=Adam(), loss='categorical_crossentropy',
            metrics=['accuracy'], epochs=30):

        self.summary()
        self.compile_model(optimizer=optimizer, loss=loss, metrics=metrics)
        self.history = self.fit(epochs=epochs)
        self.save()


def main():
    try:
        train = Train('./augmented_directory')
        train.run()
        train.evaluate()
    except Exception as e:
        print(f"{sys.argv[0]}:", e)


if __name__ == '__main__':
    main()
