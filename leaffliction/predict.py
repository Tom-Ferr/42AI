#!/usr/bin/python3
import os
import cv2
import sys
import json
import argparse
import tensorflow as tf
import keras
import shutil
from Transformation import Transformation
from helpers.signal import set_signal_handler
from helpers.directories import iter_dir_image
import matplotlib.pyplot as plt


class Predict:

    def __init__(self):
        self.load_source = 'learnings/model.keras'
        self.classes_source = 'learnings/classes.json'
        self.prediction_source = "learnings/dataset/validation"
        self.prediction_path = 'learnings/predictions'
        self.model = self.load()
        self.classes = self.load_classes()

    def load(self):
        return tf.keras.models.load_model(self.load_source)

    def load_classes(self):
        with open(self.classes_source, 'r') as f:
            return json.load(f)

    def run(self, path, plot=False):
        model = self.load()
        self.target = Transformation(path, (255, 255))
        self.mask = self.target.mask()
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2RGB)
        self.image_batch = tf.expand_dims(self.mask, axis=0)
        prediction = model.predict(self.image_batch)
        predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]

        predicted_class = self.classes[predicted_class_index]
        self.prediction_text = f'Class predicted : {predicted_class}'
        if plot:
            self.plot()
        return predicted_class

    def plot(self):
        imgs = {'Original': cv2.cvtColor(self.target.image, cv2.COLOR_BGR2RGB),
                'Transformed': self.mask}
        columns = len(imgs)

        fig, ax = plt.subplots(1, columns, figsize=(10, 5))
        for i, a in enumerate(ax.flatten()):
            a.axis('off')
            a.imshow(list(imgs.values())[i])

        plt.figtext(0.5, 0.01,
                    self.prediction_text, ha="center",
                    fontsize=12, color='green',
                    weight='bold')
        plt.figtext(0.5, 0.15,
                    "DL classification",
                    ha="center",
                    fontsize=15,
                    color='white',
                    weight='bold',
                    backgroundcolor='black')

        plt.subplots_adjust(left=0.05,
                            right=0.95,
                            top=0.85,
                            bottom=0.3,
                            wspace=0.1,
                            hspace=0.1)

        fig.patch.set_facecolor('black')
        plt.show()

    def copy_image(self, filename):
        """
        perform the transformation in the image.
        infers save or show from the content of dst_dir
        """
        dst_dir = self.prediction_path
        partial_path = os.path.relpath(filename, self.prediction_source)
        partial_path = os.path.normpath(os.path.dirname(partial_path))
        if partial_path == '.':
            partial_path = "images"
        partial_path = list(os.path.split(partial_path))
        if len(partial_path) > 1:
            del partial_path[-2]
        partial_path = os.sep.join(partial_path)
        dst_dir = os.path.join(dst_dir, partial_path)
        os.makedirs(dst_dir, exist_ok=True)
        destination = dst_dir
        Transformation(filename).transform(options=['mask'],
                                           dst_dir=destination)
        return filename

    def run_dir(self):
        path = self.prediction_source
        try:
            shutil.rmtree(self.prediction_path)
        except Exception:
            pass
        dirs = [d for d in os.listdir(path) if not d.startswith('.')]
        dirs_basename = [os.path.basename(d) for d in dirs]
        if not (all(os.path.isdir(os.path.join(path, d)) for d in dirs)
                or sorted(dirs_basename) != sorted(self.classes)):
            raise Exception(
                f"""'{path}' must contain a directory with classes {
                    self.classes}.""")
        iter_dir_image(path, self.copy_image)
        dataset = keras.utils.image_dataset_from_directory(
            self.prediction_path,
            labels='inferred',
            label_mode="categorical",
            color_mode="rgb",
            image_size=(255, 255),
            shuffle=False,
            verbose=0,
            batch_size=1
        )
        class_labels = dataset.class_names
        if class_labels != self.classes:
            raise Exception(f"The classes in the directory '{path}'",
                            "do not match the classes in the model.")

        _, acc = self.model.evaluate(dataset)
        print("model, accuracy: {:5.2f}%".format(100 * acc))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process an image.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('image', type=str,
                       help='Path to the image to be processed',
                       nargs='?')
    group.add_argument('-e', '--evaluate', dest='options',
                       action='append_const', const='evaluate',
                       help='Evaluates the model.')
    args = parser.parse_args()
    if args.image is not None and not os.path.isfile(args.image):
        raise Exception(f"The image '{args.image}' does not exist.")
    return args


def main():
    set_signal_handler()
    try:
        args = parse_arguments()
        prediction = Predict()
        if args.image:
            prediction.run(args.image, plot=True)
        else:
            prediction.run_dir()
    except Exception as e:
        print(f"{sys.argv[0]}:",  e)


if __name__ == '__main__':
    main()
