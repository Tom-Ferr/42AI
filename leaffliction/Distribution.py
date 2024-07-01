#!/usr/bin/python3

import os
import sys
import argparse
from matplotlib import pyplot as plt
from helpers.directories import (list_dir, is_file,
                                 is_above_current_dir,
                                 iter_dir)
from helpers.signal import set_signal_handler


def distribution(dirname, directories):
    """
    perform the distribution of the files in the directories
    """
    dirs = {}
    for directory in directories:
        files = [name for name in list_dir(os.path.join(dirname, directory))
                 if is_file(os.path.join(dirname, directory, name))
                 and name.lower().endswith(('.jpg', '.jpeg'))]
        dir_len = len(files)
        if (dir_len):
            dirs[directory] = dir_len
    if len(dirs) == 0:
        print("No classes found in", dirname)
        return
    print("Displaying classes for", dirname)
    dirs = dict(sorted(dirs.items(), key=lambda item: item[1], reverse=True))
    keys = list(dirs.keys())
    values = list(dirs.values())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    wedges, texts, autotexts = ax1.pie(values, labels=keys, autopct='%1.0f%%')
    colors = [w.get_facecolor() for w in wedges]
    ax2.grid(True, which='both', linestyle='--',
             linewidth=0.5, color='black', zorder=1)
    ax2.bar(range(len(dirs)), values, tick_label=keys, color=colors, zorder=2)
    fig.suptitle(f"{dirname.split('/')[-1].lower()} class distribution",
                 ha='left', x=0.05)
    plt.tight_layout()
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a directory.")
    parser.add_argument('directory', type=str,
                        help="Path to the directory to be processed")
    args = parser.parse_args()
    if not os.path.isdir(args.directory):
        print(f"Error: The directory '{args.directory}' does not exist.")
        sys.exit(1)
    return args


def main():
    """
    main function to run the distribution
    """
    set_signal_handler()
    args = parse_arguments()
    try:
        if not os.path.isdir(args.directory):
            print(sys.argv[1] + " is not a directory.")
        elif not is_above_current_dir(os.path.abspath(args.directory)):
            print("Dataset must be inside the current directory.")
        else:
            iter_dir(args.directory, distribution)
    except Exception as e:
        print("Cound not run the distribution:", e)


if __name__ == "__main__":
    """
    entry point of the script
    """
    main()
