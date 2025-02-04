from skimage.morphology import (
    skeletonize,
    remove_small_objects,
    remove_small_holes,
)
import numpy as np
from matplotlib.pylab import plt
from utils import sknw, sknw_int64
import os
import pandas as pd
from itertools import tee
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict
import json
import time
import argparse
import networkx as nx
from multiprocessing.pool import Pool
import skimage
import skimage.draw
import skimage.io
import cv2

from configs.config import Config

linestring = "LINESTRING {}"


###############################################################################
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


###############################################################################
def remove_sequential_duplicates(seq):
    # todo
    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res


###############################################################################
def remove_duplicate_segments(seq):
    seq = remove_sequential_duplicates(seq)
    segments = set()
    split_seg = []
    res = []
    for idx, (s, e) in enumerate(pairwise(seq)):
        if (s, e) not in segments and (e, s) not in segments:
            segments.add((s, e))
            segments.add((e, s))
        else:
            split_seg.append(idx + 1)
    for idx, v in enumerate(split_seg):
        if idx == 0:
            res.append(seq[:v])
        if idx == len(split_seg) - 1:
            res.append(seq[v:])
        else:
            s = seq[split_seg[idx - 1] : v]
            if len(s) > 1:
                res.append(s)
    if not len(split_seg):
        res.append(seq)
    return res


###############################################################################
def flatten(l):
    return [item for sublist in l for item in sublist]


###############################################################################
def get_angle(p0, p1=np.array([0, 0]), p2=None):
    """ compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    """
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


###############################################################################
def preprocess(
    img, thresh, img_mult=255, hole_size=300, cv2_kernel_close=7, cv2_kernel_open=7,
):
    """
    http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_holes
    hole_size in remove_small_objects is the maximum area, in pixels of the
    hole
    """

    # sometimes get a memory error with this approach
    if img.size < 10000000000:
        img = (img > (img_mult * thresh)).astype(np.bool)
        remove_small_objects(img, hole_size, in_place=True)
        remove_small_holes(img, hole_size, in_place=True)
        # img = cv2.dilate(img.astype(np.uint8), np.ones((7, 7)))

    # cv2 is generally far faster and more memory efficient (though less
    #  effective)
    else:
        # from road_raster.py, dl_post_process_pred() function
        kernel_close = np.ones((cv2_kernel_close, cv2_kernel_close), np.uint8)
        kernel_open = np.ones((cv2_kernel_open, cv2_kernel_open), np.uint8)
        kernel_blur = cv2_kernel_close

        # global thresh
        # mask_thresh = (img > (img_mult * thresh))#.astype(np.bool)
        blur = cv2.medianBlur((img * img_mult).astype(np.uint8), kernel_blur)
        glob_thresh_arr = cv2.threshold(blur, thresh, 1, cv2.THRESH_BINARY)[1]
        glob_thresh_arr_smooth = cv2.medianBlur(glob_thresh_arr, kernel_blur)
        mask_thresh = glob_thresh_arr_smooth

        # opening and closing
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        # gradient = cv2.morphologyEx(mask_thresh, cv2.MORPH_GRADIENT, kernel)
        closing_t = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
        opening_t = cv2.morphologyEx(closing_t, cv2.MORPH_OPEN, kernel_open)
        img = opening_t.astype(np.bool)
        # img = opening

    return img


###############################################################################
def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines


###############################################################################
def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(
        line1[1] - line1[0]
    )


###############################################################################
def remove_small_terminal(
    G, weight="weight", min_weight_val=30, pix_extent=1300, edge_buffer=4
):
    """Remove small terminals, if a node in the terminal is within edge_buffer
    of the the graph edge, keep it"""
    deg = dict(G.degree())
    terminal_points = [i for i, d in deg.items() if d == 1]

    edges = list(G.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = flatten([[v] for v in G[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val["pts"])
            if sum_len < 3:
                G.remove_edge(s, e)
                continue

        # check if at edge
        sx, sy = G.nodes[s]["o"]
        ex, ey = G.nodes[e]["o"]
        edge_point = False
        for ptmp in [sx, sy, ex, ey]:
            if (ptmp < (0 + edge_buffer)) or (ptmp > (pix_extent - edge_buffer)):
                edge_point = True
            else:
                continue
        # don't remove edges near the edge of the image
        if edge_point:
            continue

        vals = flatten([[v] for v in G[s][e].values()])
        for ix, val in enumerate(vals):
            if s in terminal_points and val.get(weight, 0) < min_weight_val:
                G.remove_node(s)
            if e in terminal_points and val.get(weight, 0) < min_weight_val:
                G.remove_node(e)
    return


###############################################################################
def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps


###############################################################################
def add_small_segments(
    G, terminal_points, terminal_lines, dist1=24, dist2=80, angle1=30, angle2=150,
):
    """Connect small, missing segments
    terminal points are the end of edges.  This function tries to pair small
    gaps in roads.  It will not try to connect a missed T-junction, as the 
    crossroad will not have a terminal point"""

    print("Running add_small_segments()")
    try:
        node = G.node
    except:
        node = G.nodes

    term = [node[t]["o"] for t in terminal_points]
    # print("term:", term)
    dists = squareform(pdist(term))
    possible = np.argwhere((dists > 0) & (dists < dist1))
    good_pairs = []
    for s, e in possible:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]

        if G.has_edge(s, e):
            continue
        good_pairs.append((s, e))

    possible2 = np.argwhere((dists > dist1) & (dists < dist2))
    for s, e in possible2:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if G.has_edge(s, e):
            continue
        l1 = terminal_lines[s]
        l2 = terminal_lines[e]
        d = line_points_dist(l1, l2[0])

        if abs(d) > dist1:
            continue
        angle = get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])
        if (-1 * angle1 < angle < angle1) or (angle < -1 * angle2) or (angle > angle2):
            good_pairs.append((s, e))

    dists = {}
    for s, e in good_pairs:
        s_d, e_d = [G.nodes[s]["o"], G.nodes[e]["o"]]

        dists[(s, e)] = np.linalg.norm(s_d - e_d)

    dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))

    wkt = []
    added = set()
    good_coords = []
    for s, e in dists.keys():
        if s not in added and e not in added:
            added.add(s)
            added.add(e)
            s_d, e_d = (
                G.nodes[s]["o"].astype(np.int32),
                G.nodes[e]["o"].astype(np.int32),
            )
            line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
            line = "(" + ", ".join(line_strings) + ")"
            wkt.append(linestring.format(line))
            good_coords.append((tuple(s_d), tuple(e_d)))
    return wkt, good_pairs, good_coords


###############################################################################
def make_skeleton(
    img_loc,
    thresh,
    fix_borders,
    replicate=5,
    clip=2,
    img_mult=255,
    hole_size=300,
    cv2_kernel_close=7,
    cv2_kernel_open=7,
    num_classes=1,
    skeleton_band="all",
):
    """
    Extract a skeleton from a mask.
    skeleton_band is the index of the band of the mask to use for 
        skeleton extraction, set to string 'all' to use all bands
    """
    t0 = time.time()
    # replicate = 5
    # clip = 2
    rec = replicate + clip

    # read in data
    if num_classes == 1:
        try:
            img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
        except:
            img = skimage.io.imread(img_loc, as_gray=True).astype(np.uint8)  # [::-1]

    else:
        # ensure 8bit?
        img_tmp = skimage.io.imread(img_loc).astype(np.uint8)
        # we want skimage to read in (channels, h, w) for multi-channel
        #   assume less than 20 channels
        if img_tmp.shape[0] > 20:
            img_full = np.moveaxis(img_tmp, 0, -1)
        else:
            img_full = img_tmp
        # select the desired band for skeleton extraction
        # if < 0, sum all bands
        if type(skeleton_band) == str:  # skeleton_band < 0:
            img = np.sum(img_full, axis=0).astype(np.int8)
        else:
            img = img_full[skeleton_band, :, :]

    if fix_borders:
        img = cv2.copyMakeBorder(
            img, replicate, replicate, replicate, replicate, cv2.BORDER_REPLICATE
        )

    t1 = time.time()
    img = preprocess(
        img,
        thresh,
        img_mult=img_mult,
        hole_size=hole_size,
        cv2_kernel_close=cv2_kernel_close,
        cv2_kernel_open=cv2_kernel_open,
    )

    t2 = time.time()

    if not np.any(img):
        return None, None

    ske = skeletonize(img).astype(np.uint16)
    # ske = skimage.morphology.medial_axis(img).astype(np.uint16)
    t3 = time.time()

    if fix_borders:
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(
            ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0
        )
        # ske = ske[replicate:-replicate,replicate:-replicate]
        img = img[replicate:-replicate, replicate:-replicate]
        t4 = time.time()

    t1 = time.time()

    return img, ske


###############################################################################
def img_to_ske_G(params):

    (
        img_loc,
        out_ske_file,
        out_gpickle,
        thresh,
        fix_borders,
        skel_replicate,
        skel_clip,
        img_mult,
        hole_size,
        cv2_kernel_close,
        cv2_kernel_open,
        min_spur_length_pix,
        num_classes,
        skeleton_band,
    ) = params

    # create skeleton
    img_refine, ske = make_skeleton(
        img_loc,
        thresh,
        fix_borders,
        replicate=skel_replicate,
        clip=skel_clip,
        img_mult=img_mult,
        hole_size=hole_size,
        cv2_kernel_close=cv2_kernel_close,
        cv2_kernel_open=cv2_kernel_open,
        skeleton_band=skeleton_band,
        num_classes=num_classes,
    )

    if ske is None:
        return [linestring.format("EMPTY"), [], []]

    # save to file
    if out_ske_file:
        cv2.imwrite(out_ske_file, ske.astype(np.uint8) * 255)

    # if the file is too large, use sknw_int64 to accomodate high numbers
    #   for coordinates
    if np.max(ske.shape) > 32767:
        G = sknw_int64.build_sknw(ske, multi=True)
    else:
        G = sknw.build_sknw(ske, multi=True)

    # iteratively clean out small terminals
    for itmp in range(8):
        ntmp0 = len(G.nodes())

        # sknw attaches a 'weight' property that is the length in pixels
        pix_extent = np.max(ske.shape)
        remove_small_terminal(
            G,
            weight="weight",
            min_weight_val=min_spur_length_pix,
            pix_extent=pix_extent,
        )

        # kill the loop if we stopped removing nodes
        ntmp1 = len(G.nodes())
        if ntmp0 == ntmp1:
            break
        else:
            continue

    if len(G.edges()) == 0:
        return [linestring.format("EMPTY"), [], []]

    # remove self loops
    ebunch = nx.selfloop_edges(G)
    G.remove_edges_from(list(ebunch))

    # save G
    if len(out_gpickle) > 0:
        nx.write_gpickle(G, out_gpickle)

    return G, ske, img_refine


###############################################################################
def G_to_wkt(
    G, add_small=True, img_copy=None, debug=False,
):
    """Transform G to wkt"""
    if G == [linestring.format("EMPTY")] or type(G) == str:
        return [linestring.format("EMPTY")]

    node_lines = graph2lines(G)

    if not node_lines:
        return [linestring.format("EMPTY")]
    try:
        node = G.node
    except:
        node = G.nodes

    deg = dict(G.degree())
    wkt = []
    terminal_points = [i for i, d in deg.items() if d == 1]

    # refine wkt
    terminal_lines = {}
    vertices = []
    for i, w in enumerate(node_lines):
        coord_list = []
        additional_paths = []
        for s, e in pairwise(w):
            vals = flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):

                s_coord, e_coord = node[s]["o"], node[e]["o"]

                pts = val.get("pts", [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = add_direction_change_nodes(pts, s, e, s_coord, e_coord)

                if len(ps.shape) < 2 or len(ps) < 2:
                    continue

                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue

                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)

                vertices.append(ps)

        if not len(coord_list):
            continue
        segments = remove_duplicate_segments(coord_list)

        for coord_list in segments:
            if len(coord_list) > 1:
                line = "(" + ", ".join(coord_list) + ")"
                wkt.append(linestring.format(line))
        for line_strings in additional_paths:
            line = ", ".join(line_strings)
            line_rev = ", ".join(reversed(line_strings))
            for s in wkt:
                if line in s or line_rev in s:
                    break
            else:
                wkt.append(linestring.format("(" + line + ")"))

    if add_small and len(terminal_points) > 1:
        small_segs, good_pairs, good_coords = add_small_segments(
            G, terminal_points, terminal_lines
        )
        print("small_segs", small_segs)
        wkt.extend(small_segs)

    if not wkt:
        return [linestring.format("EMPTY")]

    # return cross_segs
    return wkt


###############################################################################
def build_wkt_dir(
    indir,
    outfile,
    out_ske_dir,
    out_gdir="",
    thresh=0.3,
    im_prefix="",
    add_small=True,
    fix_borders=True,
    skel_replicate=5,
    skel_clip=2,
    img_mult=255,
    hole_size=300,
    cv2_kernel_close=7,
    cv2_kernel_open=7,
    min_spur_length_pix=16,
    num_classes=1,
    skeleton_band="all",
    n_threads=12,
):
    """Execute built_graph_wkt for an entire folder
    Split image name on AOI, keep only name after AOI.  This is necessary for 
    scoring"""

    im_files = np.sort([z for z in os.listdir(indir) if z.endswith(".tif")])
    nfiles = len(im_files)
    n_threads = min(n_threads, nfiles)
    params = []
    for i, imfile in enumerate(im_files):

        img_loc = os.path.join(indir, imfile)

        im_root = imfile.split(".")[0]

        if len(im_prefix) > 0:
            im_root = im_root.split(im_prefix)[-1]

        if out_ske_dir:
            out_ske_file = os.path.join(out_ske_dir, imfile)
        else:
            out_ske_file = ""

        if len(out_gdir) > 0:
            out_gpickle = os.path.join(out_gdir, imfile.split(".")[0] + ".gpickle")
        else:
            out_gpickle = ""

        param_row = (
            img_loc,
            out_ske_file,
            out_gpickle,
            thresh,
            fix_borders,
            skel_replicate,
            skel_clip,
            img_mult,
            hole_size,
            cv2_kernel_close,
            cv2_kernel_open,
            min_spur_length_pix,
            num_classes,
            skeleton_band,
        )
        params.append(param_row)

    # execute
    if n_threads > 1:
        pool = Pool(n_threads)
        pool.map(img_to_ske_G, params)
    else:
        img_to_ske_G(params[0])

    # now build wkt_list (single-threaded)
    all_data = []
    for gpickle in os.listdir(out_gdir):
        gpath = os.path.join(out_gdir, gpickle)
        imfile = gpickle.split(".")[0] + ".tif"
        im_root = imfile.split(".")[0]

        if len(im_prefix) > 0:
            im_root = im_root.split(im_prefix)[-1]

        G = nx.read_gpickle(gpath)
        wkt_list = G_to_wkt(G, add_small=add_small)

        # add to all_data
        for v in wkt_list:
            all_data.append((im_root, v))

    # save to csv
    df = pd.DataFrame(all_data, columns=["ImageId", "WKT_Pix"])
    df.to_csv(outfile, index=False)

    return df


###############################################################################
def main():

    add_small = True
    fix_borders = True
    skel_replicate = 5
    skel_clip = 2
    img_mult = 255
    hole_size = 300
    cv2_kernel_close = 7
    cv2_kernel_open = 7
    n_threads = 12
    im_prefix = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        cfg = json.load(f)
        config = Config(**cfg)

    min_spur_length_pix = int(np.rint(config.min_spur_length_m / config.GSD))
    print("min_spur_length_pix:", min_spur_length_pix)

    # check if we are stitching together large images or not
    out_dir_mask_norm = os.path.join(config.path_results_root, config.stitched_dir_norm)
    folds_dir = os.path.join(config.path_results_root, config.folds_save_dir)
    merge_dir = os.path.join(config.path_results_root, config.merged_dir)

    if os.path.exists(out_dir_mask_norm):
        im_dir = out_dir_mask_norm
    else:
        if config.num_folds > 1:
            im_dir = merge_dir
        else:
            im_dir = folds_dir
            im_prefix = "fold0_"

    os.makedirs(im_dir, exist_ok=True)

    # outut files
    res_root_dir = config.path_results_root
    outfile_csv = os.path.join(config.path_results_root, config.wkt_submission)

    out_ske_dir = os.path.join(
        res_root_dir, config.skeleton_dir
    )  # set to '' to not save

    os.makedirs(out_ske_dir, exist_ok=True)

    if len(config.skeleton_pkl_dir) > 0:
        out_gdir = os.path.join(
            res_root_dir, config.graph_dir, config.skeleton_pkl_dir
        )  # set to '' to not save
        os.makedirs(out_gdir, exist_ok=True)
    else:
        out_gdir = ""

    print("im_dir:", im_dir)
    print("out_ske_dir:", out_ske_dir)
    print("out_gdir:", out_gdir)

    thresh = config.skeleton_thresh

    t0 = time.time()
    df = build_wkt_dir(
        im_dir,
        outfile_csv,
        out_ske_dir,
        out_gdir,
        thresh,
        add_small=add_small,
        fix_borders=fix_borders,
        skel_replicate=skel_replicate,
        skel_clip=skel_clip,
        img_mult=img_mult,
        hole_size=hole_size,
        min_spur_length_pix=min_spur_length_pix,
        cv2_kernel_close=cv2_kernel_close,
        cv2_kernel_open=cv2_kernel_open,
        skeleton_band=config.skeleton_band,
        num_classes=config.num_classes,
        im_prefix=im_prefix,
        n_threads=n_threads,
    )

    print("len df:", len(df))
    print("outfile:", outfile_csv)

    t1 = time.time()

    print("Total time to run build_wkt_dir:", t1 - t0, "seconds")


##############################################################################
if __name__ == "__main__":
    main()
