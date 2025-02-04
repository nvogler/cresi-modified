import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import json

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
):
    """Slice images into patches, assume ground truth masks
        are present
    Adapted from basiss.py"""

    print("Slicing", im_dir)

    t0 = time.time()
    count = 0
    pos_list, name_list = [], []
    tot_pixels = 0
    # nims,h,w,nbands = im_arr.shape

    im_roots = np.sort([z for z in os.listdir(im_dir) if z.endswith(".tif")])
    for i, im_root in enumerate(im_roots):

        im_path = os.path.join(im_dir, im_root)

        print("im_path", im_path)

        name = im_root.split(".")[0]

        # load with skimage, and reverse order of bands
        im = cv2.imread(im_path)

        h, w, nbands = im.shape

        n_pix = h * w
        print("im.shape:", im.shape)
        print("n pixels:", n_pix)
        print("nbands ", nbands)
        tot_pixels += n_pix

        seen_coords = set()

        # dice it up
        # after resize, iterate through image
        #     and bin it up appropriately
        for x in range(0, w - 1, stride_x):
            for y in range(0, h - 1, stride_y):
                xmin = min(x, w - slice_x)
                ymin = min(y, h - slice_y)
                coords = (xmin, ymin)

                # check if we've already seen these coords
                if coords in seen_coords:
                    continue
                else:
                    seen_coords.add(coords)

                # check if we screwed up binning
                if (xmin + slice_x > w) or (ymin + slice_y > h):
                    print("Improperly binned image,")
                    return

                # get satellite image cutout
                im_cutout = im[ymin : ymin + slice_y, xmin : xmin + slice_x]

                # skip if the whole thing is black
                if np.max(im_cutout) < 1.0:
                    continue
                else:
                    count += 1

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
                name_list.append(name_full)
                pos_list.append(pos)

                name_out = os.path.join(out_dir, name_full)

                cv2.imwrite(name_out, im_cutout)

    # create position datataframe
    df_pos = pd.DataFrame(pos_list, columns=pos_columns)
    df_pos.index = np.arange(len(df_pos))

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
    path_tile_df_csv = os.path.join(config.path_results_root, config.tile_df_csv)
    eight_bit_dir = os.path.join(config.path_data_root, config.eight_bit_dir)
    sliced_dir = os.path.join(config.path_data_root, config.sliced_dir)

    # make output dirs
    os.makedirs(config.path_results_root, exist_ok=True)
    os.makedirs(sliced_dir, exist_ok=True)
    os.makedirs(eight_bit_dir, exist_ok=True)

    print("Output path for sliced images:", config.sliced_dir)

    # only run if nonzero tile and sliced_dir
    if (len(sliced_dir) > 0) and (config.slice_x > 0):
        print("processing starting")

        df_pos = slice_ims(
            eight_bit_dir,
            sliced_dir,
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
        )
        # save to file
        df_pos.to_csv(path_tile_df_csv)
        print("df saved to file:", path_tile_df_csv)

    else:
        print("Not creating tile_df.csv file")


###############################################################################
if __name__ == "__main__":
    main()
