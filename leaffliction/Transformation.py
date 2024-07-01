#!/usr/bin/python3

import os
import argparse
import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from helpers.signal import set_signal_handler
from helpers.directories import (is_above_current_dir, iter_dir_image,
                                 is_last_level, iter_images)
import matplotlib.gridspec as gridspec


class Transformation:
    """
    Transformation class to perform the image transformations
    """

    def __init__(self, path=None, resize=None, rgb_img=None):
        """
        initializes the transformation class
        if resize is set, the image is resized to the new dimensions
        (width, height) tuple.
        If no path is given, rgb_img is considered, which is expected to be
        a numpy array with the image in RGB format.
        """
        pcv.params.line_thickness = 2
        if path:
            self.image = cv2.imread(path)
        else:
            self.image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        self.filename = path
        if resize:
            self.image = cv2.resize(self.image, resize)
        self.roi = pcv.roi.rectangle(img=self.image, x=0, y=0,
                                     h=self.image.shape[0],
                                     w=self.image.shape[1])
        self.thresholded, self.filtered_mask = self.threshold()

    def white_balance(self, image=None):
        """
        white balance
        """
        if image is None:
            image = self.image
        white_balanced = pcv.white_balance(image)
        return white_balanced

    def gaussian_blur(self):
        gaussian_img = pcv.gaussian_blur(img=self.thresholded, ksize=(15, 15),
                                         sigma_x=0, sigma_y=None)
        return gaussian_img

    def threshold(self, image=None):
        """
        threshold the image

        Methods applied:
        - white balance
        - threshold in a and/or b channels
        - roi filter
        - threshold dilation
        - fill (to reduce noise)
        - fill holes
        """
        if not image:
            image = self.image
        image = self.white_balance(image)
        b_channel = self._channel_info('b', image)
        a_channel = self._channel_info('a', image)
        if ((a_channel['useful'] and b_channel['useful']) or
                (not a_channel['useful'] and not b_channel['useful'])):
            threshold = pcv.logical_or(a_channel['threshold'],
                                       b_channel['threshold'])
        elif (a_channel['useful']):
            threshold = a_channel['threshold']
        else:
            threshold = b_channel['threshold']
        filtered_mask = self.roi_filter(threshold, self.roi)
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=2)
        fill_threshold = pcv.fill(bin_img=dilated_mask, size=10)

        if len(np.unique(fill_threshold)) == 2:
            fill_threshold = pcv.fill_holes(fill_threshold)

        return fill_threshold, filtered_mask

    def _get_channel(self, channel='R', image=None):
        """
        retrieves the channel from an image, by label
        """
        if image is None:
            image = self.image
        if channel in "lab":
            return pcv.rgb2gray_lab(image, channel=channel)
        elif channel in "hsv":
            return pcv.rgb2gray_hsv(image, channel=channel)
        elif channel in "RGB":
            return image[:, :, 'RGB'.index(channel)]
        return None

    def _channel_info(self, channel, image=None):
        """
        Normalize a channel and apply a threshold to isolate
        light/darker regions
        """
        if image is None:
            image = self.image
        channel = self._get_channel(channel, image)
        channel_norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        height, width = channel_norm.shape
        h_size = height // 4
        w_size = width // 4
        center_region = channel_norm[h_size:3 * h_size, w_size:3 * w_size]
        corner_top_left = channel_norm[0:h_size, 0:w_size]
        corner_top_right = channel_norm[0:h_size, 3 * w_size:width]
        corner_bottom_left = channel_norm[3 * h_size:height:, 0:w_size]
        corner_bottom_right = channel_norm[3 * h_size:height, 3 * w_size:width]
        border_region = np.concatenate((corner_top_left.flatten(),
                                        corner_top_right.flatten(),
                                        corner_bottom_left.flatten(),
                                        corner_bottom_right.flatten()), axis=0)
        avg_center_intensity = np.mean(center_region)
        avg_border_intensity = np.mean(border_region)
        is_lighter = avg_center_intensity > avg_border_intensity
        channel_threshold = pcv.threshold.otsu(
                            gray_img=channel_norm,
                            object_type='light' if is_lighter else 'dark')
        intensity_difference = abs(avg_center_intensity - avg_border_intensity)
        return {
            'channel': channel,
            'channel_norm': channel_norm,
            'threshold': channel_threshold,
            'intensity_difference': intensity_difference,
            'useful': intensity_difference > 30
        }

    def mask(self, image=None):
        if image is None:
            image = self.image
        masked_image = pcv.apply_mask(img=image,
                                      mask=self.thresholded,
                                      mask_color='white')
        return masked_image

    def shape(self, image=None):
        if image is None:
            image = self.image
        shape_img = pcv.analyze.size(img=image,
                                     labeled_mask=self.thresholded)
        return shape_img

    def roi_filter(self, mask, roi):
        """
        filters the mask by the region of interest,
        returning only the largest object
        stderr redirection is necessary to avoid the
        warnings from plantcv that says that smaller objects
        will be dropped
        """
        stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, 'w')
        except Exception:
            sys.stderr = stderr
        filtered_mask = pcv.roi.filter(mask=mask,
                                       roi=roi,
                                       roi_type='largest')
        sys.stderr = stderr
        return filtered_mask

    def roi_obj(self, image=None):
        if image is None:
            image = self.image
        filtered_mask = self.roi_filter(self.thresholded, self.roi)
        boundary_image = pcv.analyze.bound_vertical(
                                    img=image.copy(),
                                    labeled_mask=filtered_mask,
                                    line_position=0,
                                    label="default")
        cv2.rectangle(boundary_image, (0, 0),
                      (image.shape[0], image.shape[1]), (0, 0, 255), 3)
        return boundary_image

    def land_marks(self, image=None):
        if image is None:
            image = self.image
        homolog_pts, start_pts, stop_pts = pcv.homology.y_axis_pseudolandmarks(
                                            img=image, mask=self.thresholded)
        img_plms = image.copy()
        homolog_pts = homolog_pts.reshape(-1, 2).astype(int)
        start_pts = start_pts.reshape(-1, 2).astype(int)
        stop_pts = stop_pts.reshape(-1, 2).astype(int)
        for i in homolog_pts:
            cv2.circle(img_plms, i, 3, (255, 0, 0), -1)

        for i in start_pts:
            cv2.circle(img_plms, i, 3, (0, 255, 0), -1)

        for i in stop_pts:
            cv2.circle(img_plms, i, 3, (0, 0, 255), -1)
        return img_plms

    def heatmap(self, image=None, channel='a'):
        if image is None:
            image = self.image
        channel = self._channel_info(channel)
        image = cv2.applyColorMap(channel['channel_norm'], cv2.COLORMAP_JET)
        masked_image = pcv.apply_mask(image, self.thresholded, 'white')
        return masked_image

    def merge(self, options: [str] = []):
        image = self.image
        if 'mask' in options:
            image = self.mask(image)
        if 'analize' in options:
            image = self.shape(image)
        if 'landmarks' in options:
            image = self.land_marks(image)
        return image

    def concat(self, options: [str] = []):
        """
        concatenates all images returning a single image
        """
        def list_to_tile(lst):
            """
            converts a list of images into a grid list.

            example:
            [img1, img2, img3, img4] -> [[img1, img2], [img3, img4]
            """
            blank_image = np.full(self.image.shape, 255, dtype=np.uint8)
            n = math.ceil(math.sqrt(len(lst)))
            if n == 0:
                return [[blank_image]]
            grid = [[blank_image] * n for _ in range(n)]
            for i in range(len(lst)):
                row = i // n
                col = i % n
                grid[row][col] = lst[i]
            if all([np.all(grid[-1][i] == blank_image) for i in range(n)]):
                grid.pop()
            return grid
        concat = []
        if 'original' in options:
            concat.append(self.image)
        if 'mask' in options:
            concat.append(self.mask())
        if 'threshold' in options:
            concat.append(
                cv2.cvtColor(self.thresholded, cv2.COLOR_GRAY2RGB))
        if 'blur' in options:
            concat.append(
                cv2.cvtColor(self.gaussian_blur(), cv2.COLOR_GRAY2RGB))
        if 'analize' in options:
            concat.append(self.shape())
        if 'roi' in options:
            concat.append(self.roi_obj())
        if 'landmarks' in options:
            concat.append(self.land_marks())
        image = cv2.vconcat([cv2.hconcat(lst) for lst in list_to_tile(concat)])
        return image

    def _save_image(self, image, filename, dst_dir, append=None):
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
        image_name = os.path.split(filename)[-1]
        if append:
            image_name = image_name[:-(len(extension) + 1)]
            image_name = f"{image_name}_{append}.{extension}"
        try:
            full_path = os.path.join(dst_dir, image_name)
            cv2.imwrite(full_path, image)
        except Exception as e:
            print(f"Could not save the image {image_name}: {Exception}, {e}")
            sys.exit(1)
        return image

    def _save_images(self, augmentations, dst_dir):
        """
        for every augmentation create a new image and save it

        arguments:
        augmentations -- a dictionary with the augmentations as keys
                         and the images as values
        """
        items = list(augmentations.items())
        if len(items) == 2:
            self._save_image(items[-1][1], self.filename, dst_dir, append=None)
            return
        for key, value in augmentations.items():
            if key != 'Original':
                self._save_image(value, self.filename, dst_dir, key)

    def histogram(self, ax=None):
        """
        if ax is None, creates and returns a figure
        """
        def calculate_proportion(image_channel):
            flattened = image_channel.flatten()
            total_pixels = flattened.size
            pixel_counts = np.bincount(flattened, minlength=255)
            pixel_proportions = pixel_counts / total_pixels
            return pixel_proportions
        fig = None
        if not ax:
            fig, ax = plt.subplots()
        channels = {
            'B': {'t': 'Blue', 'c': 'blue'},
            'b': {'t': 'Blue-Yellow', 'c': 'yellow'},
            'G': {'t': 'Green', 'c': 'green'},
            'a': {'t': 'Green-Magenta', 'c': 'magenta'},
            'h': {'t': 'Hue', 'c': 'purple'},
            'l': {'t': 'Luminosity', 'c': 'black'},
            'R': {'t': 'Red', 'c': 'red'},
            's': {'t': 'Saturation', 'c': 'cyan'},
            'v': {'t': 'Value', 'c': 'orange'},
        }
        for channel in channels:
            ax.plot(
                calculate_proportion(self._get_channel(channel)),
                color=channels[channel]['c'],
                label=channels[channel]['t'])
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_title('Histogram')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Proportion of pixels (%)')
        return fig

    def pixel_scatter(self, ax=None, x_channel='a', y_channel='b'):
        fig = None
        if not ax:
            fig, ax = plt.subplots()
        h, w, c = self.image.shape
        colors = self.image.reshape(-1, c) / 255.0
        xchannel = self._get_channel(x_channel)
        ychannel = self._get_channel(y_channel)
        ax.scatter(xchannel.flatten(),
                   ychannel.flatten(),
                   facecolors=colors,
                   alpha=0.05)

        ax.set_xlabel(f'{x_channel} Channel')
        ax.set_ylabel(f'{y_channel} Channel')
        ax.set_title(f'Pixel Scatter {x_channel} vs {y_channel}')
        return fig

    def _plot(self, augs, graphs=None):
        """
        plot the images in a grid

        arguments:
        augs -- a dictionary with the augmentations as keys
                and the images as values
        """
        if graphs:
            outer = gridspec.GridSpec(2, 1, hspace=0.1, height_ratios=[.7, .3])
        else:
            outer = gridspec.GridSpec(1, 1)
        rows = 2
        columns = math.ceil(len(augs) / rows)
        plt.figure(figsize=(12, 10))
        top = gridspec.GridSpecFromSubplotSpec(rows,
                                               columns,
                                               subplot_spec=outer[0],
                                               wspace=0.2,
                                               hspace=0.2)

        # since top is a grid, build the axes from there
        axes = [plt.subplot(top[i, j])
                for j in range(columns) for i in range(rows)]

        len_augs = len(augs)
        items = list(augs.items())
        for i, _ in enumerate(axes):
            axes[i].axis('off')
            if i < len_augs:
                out = cv2.cvtColor(items[i][1], cv2.COLOR_BGR2RGB)
                axes[i].set_title(items[i][0])
                axes[i].imshow(out,
                               cmap='gray'
                               if len(items[i][1].shape) == 2 else None)
        if graphs:
            len_graphs = len(graphs)
            bottom = gridspec.GridSpecFromSubplotSpec(1, len_graphs,
                                                      subplot_spec=outer[1],
                                                      wspace=0.2)
            axes = [plt.subplot(bottom[i]) for i in range(len_graphs)]
            for idx, graph in enumerate(graphs):
                graphs[graph](axes[idx])
        # plt.tight_layout()
        plt.show()

    def transform(self, show=False, save=True,
                  options: [str] = [], dst_dir=None):
        """
        Returns a dictionary with the transformations performed.

        available options:
        ['mask', 'threshold', 'blur', 'analize',
         'roi', 'histogram', 'landmarks', 'merge', 'concat']

        returns a dict:
        {
            'Original': np.array,
            ('Merged' or 'Concatenated': np.array,
            or one or many from
            ('Mask', Threshold, 'GaussianBlur',
             'AnalyzeObject', 'RoiObject', 'Histogram',
             'Pseudolandmarks'): np.array
        }
        """
        default_options = ['mask', 'blur',
                           'analize', 'roi',
                           'landmarks', 'heatmap', 'histogram']
        augs = {'Original': self.image}
        graphs = {}
        if options is None:
            options = default_options
        elif not set(options) - set(['merge', 'concat']):
            options.extend(default_options)
        if 'merge' in options:
            augs['Merged'] = self.merge(options)
        elif 'concat' in options:
            augs['Concatenated'] = self.concat(options)
        else:
            if 'mask' in options:
                augs['Mask'] = self.mask()
            if 'whitebalance' in options:
                augs['WhiteBalance'] = self.white_balance()
            if 'threshold' in options:
                augs['Threshold'] = self.thresholded
            if 'blur' in options:
                augs['GaussianBlur'] = self.gaussian_blur()
            if 'analize' in options:
                augs['AnalyzeObject'] = self.shape()
            if 'roi' in options:
                augs['RoiObject'] = self.roi_obj()
            if 'landmarks' in options:
                augs['Pseudolandmarks'] = self.land_marks()
            if 'channels' in options:
                augs['Channel-l'] = self._get_channel('l')
                augs['Channel-a'] = self._get_channel('a')
                augs['Channel-b'] = self._get_channel('b')
                augs['Channel-h'] = self._get_channel('h')
                augs['Channel-s'] = self._get_channel('s')
                augs['Channel-v'] = self._get_channel('v')
                augs['Channel-R'] = self._get_channel('R')
                augs['Channel-G'] = self._get_channel('G')
                augs['Channel-B'] = self._get_channel('B')
            if 'histogram' in options:
                graphs['Histogram'] = self.histogram
            if 'scatter' in options:
                graphs['PixelScatter'] = self.pixel_scatter
            if 'heatmap' in options:
                augs['Heatmap a'] = self.heatmap(channel='a')
        if save:
            self._save_images(augs, dst_dir)
        if show:
            self._plot(augs, graphs)
        return augs


def process_image(filename, src_dir=None, dst_dir=None, options: [str] = []):
    """
    perform the transformation in the image.
    infers save or show from the content of dst_dir
    """
    trans = Transformation(filename)
    save = True if dst_dir else False
    show = False if dst_dir else True
    if dst_dir:
        partial_path = os.path.relpath(filename, src_dir)
        partial_path = os.path.dirname(partial_path)
        dst_dir = os.path.join(dst_dir, partial_path) if dst_dir else None
        os.makedirs(dst_dir, exist_ok=True)
    trans.transform(show=show, save=save, options=options, dst_dir=dst_dir)


def process_directory(src_dir, dst_dir, options: [str] = []):
    """
    perform the transformation in the directories,
    copying the resulting files into a new directory
    """
    def wrapper(filename):
        """
        since iter_dir_image receives a function with only one
        argument (the filename), this wrapper function is necessary
        to call process_image with the correct arguments
        """
        return process_image(filename, src_dir, dst_dir, options)

    if is_last_level(src_dir):
        return iter_images('', [src_dir], wrapper)
    return iter_dir_image(src_dir, wrapper)


def parse_arguments():
    parser = argparse.ArgumentParser(
                description="Creates transformation of a file or directory.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('filename', type=str,
                       help='Path to the filename to be processed',
                       nargs='?')
    group.add_argument('-s', '--src', type=str,
                       help='Process all files within a directory',
                       dest='src', metavar='source', nargs='?')
    parser.add_argument('-d', '--dst', type=str,
                        help='Saves the processed file into a directory',
                        dest='dst',
                        metavar='destination', nargs='?')
    parser.add_argument('-m', '--mask', dest='options', action='append_const',
                        const='mask', help='Add mask option.')
    parser.add_argument('-t', '--threshold', dest='options',
                        action='append_const', const='threshold',
                        help='Add threshold option.')
    parser.add_argument('-b', '--blur', dest='options', action='append_const',
                        const='blur', help='Add blur option.')
    parser.add_argument('-a', '--analize', dest='options',
                        action='append_const', const='analize',
                        help='Add analize option.')
    parser.add_argument('-r', '--roi', dest='options',
                        action='append_const', const='roi',
                        help='Add ROI option.')
    parser.add_argument('-i', '--histogram', dest='options',
                        action='append_const', const='histogram',
                        help='Add histogram option.')
    parser.add_argument('-x', '--scatter', dest='options',
                        action='append_const', const='scatter',
                        help='Add Pixel Scatter option.')
    parser.add_argument('-w', '--whitebalance', dest='options',
                        action='append_const', const='whitebalance',
                        help='Add White Balance option.')
    parser.add_argument('-l', '--landmarks', dest='options',
                        action='append_const', const='landmarks',
                        help='Add landmarks option.')
    parser.add_argument('-c', '--channels', dest='options',
                        action='append_const', const='channels',
                        help='Add channels option.')
    parser.add_argument('-e', '--heatmap', dest='options',
                        action='append_const', const='heatmap',
                        help='Add heatmap option.')
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--merge', dest='options',
                              action='append_const', const='merge',
                              help='Merges the Image(s).')
    action_group.add_argument('--concat', dest='options',
                              action='append_const', const='concat',
                              help='Concatenates the image(s).')
    args = parser.parse_args()
    if args.src and not args.dst:
        parser.error('--src requires --dst to be set.')
    if args.dst and not args.src:
        parser.error('--dst requires --src to be set.')
    if args.filename is not None and not os.path.isfile(args.filename):
        raise Exception(f"'{args.filename}' is not a file.")
    if args.src is not None and not os.path.isdir(args.src):
        raise Exception(f"--src '{args.src}' is not a directory.")
    if args.dst is not None and os.path.isfile(args.dst):
        raise Exception(f"--dst '{args.dst}' is not a directory.")
    return args


def main():
    """
    main function to run the transformation
    """
    set_signal_handler()
    try:
        args = parse_arguments()
        if args.filename:
            process_image(args.filename, options=args.options)
        else:
            if not is_above_current_dir(os.path.abspath(args.src)):
                raise Exception(
                        "Dataset must be inside the current directory.")
            else:
                process_directory(args.src, args.dst, options=args.options)
    except Exception as e:
        print(f"{sys.argv[0]}:", e)


if __name__ == "__main__":
    main()
