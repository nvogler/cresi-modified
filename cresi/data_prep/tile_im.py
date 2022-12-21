#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:16:58 2018

@author: avanetten
"""

from __future__ import print_function

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import json

# cv2 can't load large files, so need to import skimage too
import skimage.io

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "configs"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.config import Config


###############################################################################
def slice_ims(
    im_dir,
    out_dir,
    slice_x,
    slice_y,
    stride_x,
    stride_y,
    pos_columns=[
        "idx",
        "name",
        "name_full",
        "xmin",
        "ymin",
        "slice_x",
        "slice_y",
        "im_x",
        "im_y",
    ],
    sep="__",
    verbose=True,
):
    """Slice images into patches, assume ground truth masks
        are present
    Adapted from basiss.py"""

    print("slicing", im_dir)
    if verbose:
        print("Slicing images in:", im_dir)

    t0 = time.time()
    count = 0
    pos_list, name_list = [], []
    tot_pixels = 0
    # nims,h,w,nbands = im_arr.shape

    im_roots = np.sort([z for z in os.listdir(im_dir) if z.endswith(".tif")])
    for i, im_root in enumerate(im_roots):

        im_path = os.path.join(im_dir, im_root)

        print("im_path", im_path)

        if verbose:
            print("im_path:", im_path)
        name = im_root.split(".")[0]

        # load with skimage, and reverse order of bands
        im = skimage.io.imread(im_path)[::-1]
        # cv2 can't load large files
        # im = cv2.imread(im_path)
        h, w, nbands = im.shape
        n_pix = h * w
        print("im.shape:", im.shape)
        print("n pixels:", n_pix)
        tot_pixels += n_pix

        seen_coords = set()

        # if verbose and (i % 10) == 0:
        #    print(i, "im_root:", im_root)

        # dice it up
        # after resize, iterate through image
        #     and bin it up appropriately
        for x in range(0, w - 1, stride_x):
            # y axis is inverted
            for y in range(0, h - 1, stride_y):
                ymax = h - y

                xmin = min(x, w - slice_x)
                # ymax = m(ymax, 0)
                coords = (xmin, ymax)

                # check if we've already seen these coords
                if coords in seen_coords:
                    continue
                else:
                    seen_coords.add(coords)

                # check if we screwed up binning
                if (xmin + slice_x > w) or (ymax - slice_y < 0):
                    print("Improperly binned image,")
                    return

                # handle y axis inversion
                print("old", ymax)
                print("new", ymax - slice_y)

                # get satellite image cutout
                im_cutout = im[ymax - slice_y : ymax, xmin : xmin + slice_x]

                ##############
                # skip if the whole thing is black
                if np.max(im_cutout) < 1.0:
                    continue
                else:
                    count += 1

                if verbose and (count % 50) == 0:
                    print("count:", count, "x:", x, "y:", y)
                ###############

                # set slice name
                name_full = (
                    str(i)
                    + sep
                    + name
                    + sep
                    + str(xmin)
                    + sep
                    + str(ymin)
                    + sep
                    + str(slice_x)
                    + sep
                    + str(slice_y)
                    + sep
                    + str(w)
                    + sep
                    + str(h)
                    + ".tif"
                )

                pos = [i, name, name_full, xmin, ymin, slice_x, slice_y, w, h]
                # add to arrays
                # idx_list.append(idx_full)
                name_list.append(name_full)
                # im_list.append(im_cutout)
                # mask_list.append(mask_cutout)
                pos_list.append(pos)

                name_out = os.path.join(out_dir, name_full)

                # if we read in with skimage, need to reverse colors
                cv2.imwrite(name_out, cv2.cvtColor(im_cutout, cv2.COLOR_RGB2BGR))

                # cv2.imwrite(name_out, im_cutout)

    # create position datataframe
    df_pos = pd.DataFrame(pos_list, columns=pos_columns)
    df_pos.index = np.arange(len(df_pos))

    # if True:  # verbose:
    print("  len df;", len(df_pos))
    print("  Time to slice arrays:", time.time() - t0, "seconds")
    print("  Total pixels in test image(s):", tot_pixels)

    return df_pos


##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    # get config
    with open(args.config_path, "r") as f:
        cfg = json.load(f)
        config = Config(**cfg)

    # get input dir
    path_images_8bit = os.path.join(config.eight_bit_dir)

    # make output dirs
    res_dir = os.path.join(config.path_results_root, config.results_dir)
    os.makedirs(res_dir, exist_ok=True)

    path_tile_df_csv = os.path.join(
        config.path_results_root, config.results_dir, config.tile_df_csv
    )

    # path for sliced data
    path_sliced = config.sliced_dir

    print("Output path for sliced images:", path_sliced)

    # only run if nonzero tile and sliced_dir
    if (len(config.sliced_dir) > 0) and (config.slice_x > 0):
        print("processing starting")

        os.makedirs(path_sliced, exist_ok=True)

        df_pos = slice_ims(
            path_images_8bit,
            path_sliced,
            config.slice_x,
            config.slice_y,
            config.stride_x,
            config.stride_y,
            pos_columns=[
                "idx",
                "name",
                "name_full",
                "xmin",
                "ymin",
                "slice_x",
                "slice_y",
                "im_x",
                "im_y",
            ],
            verbose=False,
        )
        # save to file
        df_pos.to_csv(path_tile_df_csv)
        print("df saved to file:", path_tile_df_csv)

    else:
        print("Not creating tile_df.csv file")


###############################################################################
if __name__ == "__main__":
    main()
