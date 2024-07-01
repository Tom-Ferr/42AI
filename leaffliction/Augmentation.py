#!/usr/bin/python3

import os
import argparse
import sys
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from random import randint, seed, choices, uniform
from helpers.signal import set_signal_handler
from helpers.directories import is_above_current_dir, iter_dir_image, iter_dir


class Augmentation:
    def __init__(self, path):
        try:
            self.image = cv2.imread(path)
            if self.image is None:
                raise Exception()
        except Exception as e:
            raise Exception(f"Could not Load the image {path}. {e}")
        self.filename = path

    def crop(self):
        cropped = self.image[10:150, 10:150]
        return cropped

    def rotate(self):
        angle = 0
        while angle < 10 and angle > -10:
            angle = randint(-70, 70)
        scale = 1.0
        w = self.image.shape[1]
        h = self.image.shape[0]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        rotated = cv2.warpAffine(self.image,
                                 M, (w, h),
                                 borderValue=(255, 255, 255))
        return rotated

    def flip(self, x_axis=False):
        c = 1
        if x_axis:
            c = 0
        flipped = cv2.flip(self.image, flipCode=c)
        return flipped

    def blur(self, point=(10, 10)):
        blured = cv2.blur(self.image, point)
        return blured

    def contrast(self, brightness=10):
        contrast = uniform(1.2, 2.1)
        contrasted = cv2.addWeighted(self.image, contrast,
                                     np.zeros(self.image.shape,
                                              self.image.dtype),
                                     0, brightness)
        return contrasted

    def noise(self, mean=0, std=25):
        noise = np.random.normal(mean, std, self.image.shape).astype(np.uint8)
        noisy_image = cv2.add(self.image, noise)
        return noisy_image

    def blur2(self):
        """
        apply a blur filter to the image
        """
        min_blur = 1
        max_blur = 15
        amount = (randint(min_blur, max_blur) / 1000)
        ksize = (int(self.image.shape[0] * amount),
                 int(self.image.shape[1] * amount))
        if ksize[0] % 2 == 0:
            ksize = (ksize[0] + 1, ksize[1])
        if ksize[1] % 2 == 0:
            ksize = (ksize[0], ksize[1] + 1)
        return cv2.blur(self.image, ksize)

    def contrast2(self):
        """
        apply a contrast filter to the image
        """
        alpha = randint(5, 15) / 10
        return cv2.convertScaleAbs(self.image, alpha=alpha)

    def illumination2(self):
        """
        apply a illumination filter to the image
        """
        beta = randint(0, 50)
        return cv2.convertScaleAbs(self.image, beta=beta)

    def rotate2(self):
        """
        rotate the image
        """
        rows, cols = self.image.shape[:2]
        angle = randint(-70, 70)
        image_center = (cols / 2, rows / 2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        new_width = int(rows * abs_sin + cols * abs_cos)
        new_height = int(rows * abs_cos + cols * abs_sin)
        rotation_matrix[0, 2] += new_width / 2 - image_center[0]
        rotation_matrix[1, 2] += new_height / 2 - image_center[1]
        image = cv2.warpAffine(self.image, rotation_matrix,
                               (new_width, new_height),
                               borderValue=(255, 255, 255))
        return image

    def perspective2(self):
        """
        Adds a perspective to the image
        """
        max_skew = .15
        rows, cols = self.image.shape[:2]
        starting_points = np.float32([[0, 0], [cols, 0],
                                     [0, rows], [cols, rows]])
        ending_points = np.float32([[randint(0, int(cols * max_skew)),
                                    randint(0, int(rows * max_skew))],
                                    [cols - randint(0, int(cols * max_skew)),
                                    randint(0, int(rows * max_skew))],
                                    [randint(0, int(cols * max_skew)),
                                    rows - randint(0, int(rows * max_skew))],
                                    [cols - randint(0, int(cols * max_skew)),
                                    rows - randint(0, int(rows * max_skew))]
                                    ])
        matrix = cv2.getPerspectiveTransform(starting_points, ending_points)
        new_corners = cv2.perspectiveTransform(np.array([starting_points]),
                                               matrix)
        image = cv2.warpPerspective(self.image, matrix,
                                    (cols, rows), borderValue=(255, 255, 255))
        min_x = min(new_corners[0, :, 0])
        max_x = max(new_corners[0, :, 0])
        min_y = min(new_corners[0, :, 1])
        max_y = max(new_corners[0, :, 1])
        image = image[int(min_y):int(max_y), int(min_x):int(max_x)]
        image = cv2.resize(image, (cols, rows))
        return image

    def scaling2(self):
        """
        scales the image
        """
        rows, cols = self.image.shape[:2]
        scale = 1.0 + (randint(20, 45) / 100)
        image = cv2.resize(self.image, (int(cols * scale), int(rows * scale)))
        new_rows, new_cols = image.shape[:2]
        start_row = int((new_rows - rows) / 2)
        end_row = new_rows - start_row
        start_col = int((new_cols - cols) / 2)
        end_col = new_cols - start_col
        image = image[start_row:end_row, start_col:end_col]
        return image

    def _save_image(self, image, filename, append):
        """
        save the image to the disk
        returns the image itself.
        """
        if filename.upper().endswith('.JPG'):
            extension = filename[-3:]
        elif filename.upper().endswith('.JPEG'):
            extension = filename[-4:]
        else:
            raise Exception(f"Invalid extension: {filename}")
        image_name = filename[:-(len(extension) + 1)]
        image_name = f"{image_name}_{append}.{extension}"
        try:
            cv2.imwrite(image_name, image)
        except Exception as e:
            print(f"Could not save the image {image_name}: {Exception}, {e}")
            sys.exit(1)
        return image

    def _save_images(self, augmentations):
        """
        for every augmentation create a new image and save it

        arguments:
        augmentations -- a dictionary with the augmentations as keys
                         and the images as values
        """
        for key, value in augmentations.items():
            if key != 'Original':
                self._save_image(value, self.filename, key)

    def _plot(self, augs):
        """
        plot the images in a grid

        arguments:
        augs -- a dictionary with the augmentations as keys
                and the images as values
        """
        columns = len(augs)
        fig, ax = plt.subplots(2, math.ceil(columns / 2), figsize=(10, 5))
        for i, a in enumerate(ax.flatten()):
            a.axis('off')
            if i >= columns:
                break
            a.set_title(list(augs.keys())[i])
            img = list(augs.values())[i]
            out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            a.imshow(out)
        plt.tight_layout()
        plt.show()

    def augmentate(self, show=False, save=True, random=False):
        """
        apply the augmentations to the image
        if random is True, apply only one random augmentation
        if save is True, save the images to the disk
        if show is True, plots the images
        """
        augs = {
            'Original': self.image,
            'Rotate': self.rotate,
            'Flip': self.flip,
            'Blur': self.blur,
            'Contrast': self.contrast,
            'Scaling': self.scaling2,
            'Perspective': self.perspective2,
        }
        if random:
            rnd = randint(1, len(augs) - 1)
            rnd_item = list(augs.keys())[rnd]
            augs = {
                'Original': self.image,
                rnd_item: augs[rnd_item]
            }
        for idx, key in enumerate(augs.keys()):
            if idx > 0:
                augs[key] = augs[key]()
        if save:
            self._save_images(augs)
        if show:
            self._plot(augs)


def process_image(filename, show=False, save=True, random=False):
    aug = Augmentation(filename)
    aug.augmentate(show, save, random)


def parse_arguments():
    parser = argparse.ArgumentParser(
                            description="Process a file or directory.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('filename', type=str,
                       help='Path to the filename to be processed',
                       nargs='?')
    group.add_argument('-r', '--recursive', type=str,
                       help='Process all files within a directory',
                       dest='directory', metavar='directory')
    parser.add_argument('-n', '--nosave', action='store_true', default=False,
                        help='Do not save files to disk')
    parser.add_argument('-b', '--balance',  action='store_true', default=False,
                        help='Do not save files to disk')
    args = parser.parse_args()
    if args.directory is not None and args.nosave is True:
        raise Exception("The option '-n' cannot be used with a directory")
    if args.filename is not None and not os.path.isfile(args.filename):
        raise Exception(f"'{args.filename}' is not a file.")
    if args.directory is not None and not os.path.isdir(args.directory):
        raise Exception(f"'{args.directory}' is not a directory.")
    if args.directory is None and args.balance:
        raise Exception("The option '-b' must be used with a directory")
    return args


def balance(dirname):
    def get_files(directory):
        files = [name for name in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, name))
                 and name.lower().endswith(('.jpg', '.jpeg'))]
        return files

    def count(dirname, directories):
        ret = {}
        for directory in directories:
            files = get_files(os.path.join(dirname, directory))
            ret[os.path.join(dirname, directory)] = len(files)
        return ret

    print("Recursively balancing the dataset on", dirname)
    while True:
        # will loop until all directories have the same number of files
        directories = iter_dir(dirname, count)
        combined = {}
        for d in directories:
            for key, value in d.items():
                combined[key] = value
        if len(set(combined.values())) == 1:
            break
        print('.', end='', flush=True)
        max_value = max(combined.values())
        max_value = max(max_value, 3600)
        for key, value in combined.items():
            if value < max_value:
                files = get_files(key)
                files = choices(files, k=max_value - value)
                for file in files:
                    process_image(os.path.join(key, file),
                                  show=False,
                                  save=True,
                                  random=True)
    print()


def main():
    """
    main function to run the augmentation
    """
    set_signal_handler()
    seed()
    try:
        args = parse_arguments()
        if args.filename:
            process_image(args.filename, show=True, save=not args.nosave)
        else:
            if not is_above_current_dir(os.path.abspath(args.directory)):
                raise Exception(
                        "Dataset must be inside the current directory.")
            elif args.balance:
                balance(args.directory)
            else:
                iter_dir_image(args.directory, process_image)
    except Exception as e:
        print(f"{sys.argv[0]}:", e)


if __name__ == "__main__":
    """
    entry point of the script
    """
    main()
