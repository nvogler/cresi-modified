import os
import utm
import shapely.wkt
import shapely.ops
from shapely.geometry import mapping, Point, LineString
import fiona
import networkx as nx
import osmnx as ox
from osgeo import gdal, ogr, osr
import argparse
import json
import pandas as pd
import numpy as np
import time
import simplekml

from multiprocessing.pool import Pool

# import cv2
from utils import rdp, osmnx_funcs
from configs.config import Config


###############################################################################
# from apls.py
###############################################################################
def clean_sub_graphs(
    G_, min_length=300, max_nodes_to_skip=20, weight="length_pix",
):
    """Remove subgraphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length
        (this step great improves processing time)"""

    if len(G_.nodes()) == 0:
        return G_

    try:
        sub_graphs = list(nx.connected_component_subgraphs(G_))
    except:
        sub_graph_nodes = nx.connected_components(G_)
        sub_graphs = [G_.subgraph(c).copy() for c in sub_graph_nodes]

    bad_nodes = []

    for G_sub in sub_graphs:
        # don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue

        else:
            all_lengths = dict(nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))

            # get all lenghts
            lens = []
            # for u,v in all_lengths.iteritems():
            for u in all_lengths.keys():
                v = all_lengths[u]
                # for uprime, vprime in v.iteritems():
                for uprime in v.keys():
                    vprime = v[uprime]
                    lens.append(vprime)

            max_len = np.max(lens)

            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())

    # remove bad_nodes
    G_.remove_nodes_from(bad_nodes)

    return G_


###############################################################################
def remove_short_edges(G_, min_spur_length_m=2, length_key="length"):
    """Remove unconnected edges shorter than the desired length"""

    deg_list = list(G_.degree)

    # iterate through list
    bad_nodes = []
    for i, (n, deg) in enumerate(deg_list):

        # check if node has only one neighbor
        if deg == 1:
            # get edge
            edge = list(G_.edges(n))
            u, v = edge[0]

            # get edge length
            edge_props = G_.get_edge_data(u, v, 0)
            length = edge_props[length_key]

            # edge_props = G_.edges([u, v])
            if length < min_spur_length_m:
                bad_nodes.append(n)

    G_.remove_nodes_from(bad_nodes)

    return G_


###############################################################################
def wkt_list_to_nodes_edges(wkt_list, node_iter=10000, edge_iter=10000):
    """Convert wkt list to nodes and edges
    Make an edge between each node in linestring. Since one linestring
    may contain multiple edges, this is the safest approach"""

    node_loc_set = set()  # set of edge locations
    node_loc_dic = {}  # key = node idx, val = location
    node_loc_dic_rev = {}  # key = location, val = node idx
    edge_loc_set = set()  # set of edge locations
    edge_dic = {}  # edge properties

    for i, lstring in enumerate(wkt_list):
        # get lstring properties
        shape = shapely.wkt.loads(lstring)

        xs, ys = shape.xy

        # iterate through coords in line to create edges between every point
        for j, (x, y) in enumerate(zip(xs, ys)):
            loc = (x, y)
            # for first item just make node, not edge
            if j == 0:
                # if not yet seen, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1

            # if not first node in edge, retrieve previous node and build edge
            else:
                prev_loc = (xs[j - 1], ys[j - 1])
                # print ("prev_loc:", prev_loc)
                prev_node = node_loc_dic_rev[prev_loc]

                # if new, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1

                # if seen before, retrieve node properties
                else:
                    node = node_loc_dic_rev[loc]

                # add edge, which is start_node to end_node
                edge_loc = (loc, prev_loc)
                edge_loc_rev = (prev_loc, loc)

                # shouldn't be duplicate edges, so break if we see one
                if (edge_loc in edge_loc_set) or (edge_loc_rev in edge_loc_set):
                    print("Oops, edge already seen, returning:", edge_loc)
                    return

                # get distance to prev_loc and current loc
                proj_prev = shape.project(Point(prev_loc))
                proj = shape.project(Point(loc))

                # edge length is the diffence of the two projected lengths
                #   along the linestring
                edge_length = abs(proj - proj_prev)

                # make linestring
                line_out = LineString([prev_loc, loc])
                line_out_wkt = line_out.wkt

                edge_props = {
                    "start": prev_node,
                    "start_loc_pix": prev_loc,
                    "end": node,
                    "end_loc_pix": loc,
                    "length_pix": edge_length,
                    "wkt_pix": line_out_wkt,
                    "geometry_pix": line_out,
                    "osmid": i,
                }

                edge_loc_set.add(edge_loc)
                edge_dic[edge_iter] = edge_props
                edge_iter += 1

    return node_loc_dic, edge_dic


###############################################################################
def nodes_edges_to_G(node_loc_dic, edge_dic, name="glurp"):
    """Take output of wkt_list_to_nodes_edges(wkt_list) and create networkx
    graph"""

    G = nx.MultiDiGraph()
    # set graph crs and name
    G.graph = {"name": name, "crs": "EPSG:4326"}

    # add nodes
    for key in node_loc_dic.keys():
        val = node_loc_dic[key]
        attr_dict = {"osmid": key, "x_pix": val[0], "y_pix": val[1]}
        G.add_node(key, **attr_dict)

    # add edges
    for key in edge_dic.keys():
        val = edge_dic[key]
        attr_dict = val
        u = attr_dict["start"]
        v = attr_dict["end"]

        if type(attr_dict["start_loc_pix"]) == list:
            return

        G.add_edge(u, v, **attr_dict)

    G2 = G.to_undirected()

    return G2


###############################################################################
def wkt_to_shp(wkt_list, shp_file):
    """Take output of build_graph_wkt() and render the list of linestrings
    into a shapefile
    # https://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles
    """

    print(wkt_list)
    print("writing shapefile from wkt list")

    # Define a linestring feature geometry with one attribute
    schema = {
        "geometry": "LineString",
        "properties": {"id": "int"},
    }

    # Write a new shapefile
    with fiona.open(shp_file, "w", "ESRI Shapefile", schema) as c:
        for i, line in enumerate(wkt_list):
            shape = shapely.wkt.loads(line)
            c.write(
                {"geometry": mapping(shape), "properties": {"id": i},}
            )
    return


###############################################################################
def shp_to_G(shp_file):
    """Ingest G from shapefile
    DOES NOT APPEAR TO WORK CORRECTLY"""

    G = nx.read_shp(shp_file)

    return G


###############################################################################
def pixelToGeoCoord(params):
    """from spacenet geotools"""

    sourceSR = ""
    geomTransform = ""
    targetSR = osr.SpatialReference()
    targetSR.ImportFromEPSG(4326)

    identifier, xPix, yPix, inputRaster = params

    if targetSR == "":
        performReprojection = False
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
    else:
        performReprojection = True

    if geomTransform == "":
        srcRaster = gdal.Open(inputRaster)
        geomTransform = srcRaster.GetGeoTransform()

        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(srcRaster.GetProjectionRef())

    geom = ogr.Geometry(ogr.wkbPoint)
    xOrigin = geomTransform[0]
    yOrigin = geomTransform[3]
    pixelWidth = geomTransform[1]
    pixelHeight = geomTransform[5]

    xCoord = (xPix * pixelWidth) + xOrigin
    yCoord = (yPix * pixelHeight) + yOrigin
    geom.AddPoint(xCoord, yCoord)

    if performReprojection:
        if sourceSR == "":
            srcRaster = gdal.Open(inputRaster)
            sourceSR = osr.SpatialReference()
            sourceSR.ImportFromWkt(srcRaster.GetProjectionRef())
        coord_trans = osr.CoordinateTransformation(sourceSR, targetSR)
        geom.Transform(coord_trans)

    return {identifier: (geom.GetX(), geom.GetY())}


##############################################################################
def get_node_geo_coords(G, im_file, fix_utm_zone=True, n_threads=12):
    # get pixel params
    params = []
    nn = len(G.nodes())
    for i, (n, attr_dict) in enumerate(G.nodes(data=True)):

        x_pix, y_pix = attr_dict["x_pix"], attr_dict["y_pix"]
        params.append((n, x_pix, y_pix, im_file))

    n_threads = min(n_threads, nn)
    # execute
    print("Computing geo coords for nodes (" + str(n_threads) + " threads)...")
    if n_threads > 1:
        pool = Pool(n_threads)
        coords_dict_list = pool.map(pixelToGeoCoord, params)
    else:
        coords_dict_list = pixelToGeoCoord(params[0])

    # combine the disparate dicts
    coords_dict = {}
    for d in coords_dict_list:
        coords_dict.update(d)

    # update data
    print("Updating data properties")
    utm_letter = "Oooops"
    for i, (n, attr_dict) in enumerate(G.nodes(data=True)):

        lon, lat = coords_dict[n]

        # fix zone
        if i == 0 or fix_utm_zone == False:
            [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)

        else:
            [utm_east, utm_north, _, _] = utm.from_latlon(
                lat, lon, force_zone_number=utm_zone, force_zone_letter=utm_letter
            )

        if lat > 90:
            print("lat > 90, returning:", n, attr_dict)
            return
        attr_dict["lon"] = lon
        attr_dict["lat"] = lat
        attr_dict["utm_east"] = utm_east
        attr_dict["utm_zone"] = utm_zone
        attr_dict["utm_letter"] = utm_letter
        attr_dict["utm_north"] = utm_north
        attr_dict["x"] = lon
        attr_dict["y"] = lat

    return G


##############################################################################
def get_node_geo_coords_single_threaded(G, im_file, fix_utm_zone=True):

    for i, (n, attr_dict) in enumerate(G.nodes(data=True)):

        x_pix, y_pix = attr_dict["x_pix"], attr_dict["y_pix"]

        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
        lon, lat = pixelToGeoCoord(x_pix, y_pix, im_file, targetSR=targetSR)

        # fix zone
        if i == 0 or fix_utm_zone == False:
            [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)
        else:
            [utm_east, utm_north, _, _] = utm.from_latlon(
                lat, lon, force_zone_number=utm_zone, force_zone_letter=utm_letter
            )

        if lat > 90:
            print("lat > 90, returning:", n, attr_dict)
            return
        attr_dict["lon"] = lon
        attr_dict["lat"] = lat
        attr_dict["utm_east"] = utm_east
        attr_dict["utm_zone"] = utm_zone
        attr_dict["utm_letter"] = utm_letter
        attr_dict["utm_north"] = utm_north
        attr_dict["x"] = lon
        attr_dict["y"] = lat

    return G


###############################################################################
def convert_pix_lstring_to_geo(params):

    """Convert linestring in pixel coords to geo coords
    If zone or letter changes inthe middle of line, it's all screwed up, so
    force zone and letter based on first point
    (latitude, longitude, force_zone_number=None, force_zone_letter=None)
    Or just force utm zone and letter explicitly
    """

    identifier, geom_pix_wkt, im_file, utm_zone, utm_letter = params
    shape = shapely.wkt.loads(geom_pix_wkt)
    x_pixs, y_pixs = shape.xy
    coords_latlon = []
    coords_utm = []

    for i, (x, y) in enumerate(zip(x_pixs, y_pixs)):
        params_tmp = ("tmp", x, y, im_file)
        tmp_dict = pixelToGeoCoord(params_tmp)
        (lon, lat) = list(tmp_dict.values())[0]

        if utm_zone and utm_letter:
            [utm_east, utm_north, _, _] = utm.from_latlon(
                lat, lon, force_zone_number=utm_zone, force_zone_letter=utm_letter
            )
        else:
            [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)

        coords_utm.append([utm_east, utm_north])
        coords_latlon.append([lon, lat])

    lstring_latlon = LineString([Point(z) for z in coords_latlon])
    lstring_utm = LineString([Point(z) for z in coords_utm])

    return {identifier: (lstring_latlon, lstring_utm, utm_zone, utm_letter)}


###############################################################################
def get_edge_geo_coords(
    G, im_file, remove_pix_geom=True, fix_utm_zone=True, n_threads=12
):
    """Get geo coords of all edges"""

    # first, get utm letter and zone of first node in graph
    for i, (n, attr_dict) in enumerate(G.nodes(data=True)):
        x_pix, y_pix = attr_dict["x_pix"], attr_dict["y_pix"]
        if i > 0:
            break
    params_tmp = ("tmp", x_pix, y_pix, im_file)
    print("params_tmp", params_tmp)

    tmp_dict = pixelToGeoCoord(params_tmp)
    print("tmp_dict:", tmp_dict)

    (lon, lat) = list(tmp_dict.values())[0]
    [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)

    # now get edge params
    params = []
    ne = len(list(G.edges()))
    for i, (u, v, attr_dict) in enumerate(G.edges(data=True)):

        geom_pix = attr_dict["geometry_pix"]

        if fix_utm_zone == False:
            params.append(((u, v), geom_pix.wkt, im_file, None, None))
        else:
            params.append(((u, v), geom_pix.wkt, im_file, utm_zone, utm_letter))

    n_threads = min(n_threads, ne)

    # execute
    print("Computing geo coords for edges (" + str(n_threads) + " threads)...")
    if n_threads > 1:
        pool = Pool(n_threads)
        coords_dict_list = pool.map(convert_pix_lstring_to_geo, params)
    else:
        coords_dict_list = convert_pix_lstring_to_geo(params[0])

    # combine the disparate dicts
    coords_dict = {}
    for d in coords_dict_list:
        coords_dict.update(d)

    print("Updating edge data properties")
    for i, (u, v, attr_dict) in enumerate(G.edges(data=True)):

        geom_pix = attr_dict["geometry_pix"]

        lstring_latlon, lstring_utm, utm_zone, utm_letter = coords_dict[(u, v)]

        attr_dict["geometry_latlon_wkt"] = lstring_latlon.wkt
        attr_dict["geometry_utm_wkt"] = lstring_utm.wkt
        attr_dict["length_latlon"] = lstring_latlon.length
        attr_dict["length_utm"] = lstring_utm.length
        attr_dict["length"] = lstring_utm.length
        attr_dict["utm_zone"] = utm_zone
        attr_dict["utm_letter"] = utm_letter

        # geometry screws up osmnx.simplify function
        if remove_pix_geom:
            attr_dict["geometry_pix"] = geom_pix.wkt

        # try actual geometry, not just linestring, this seems necessary for
        #  projections
        attr_dict["geometry"] = lstring_latlon

        # ensure utm length isn't excessive
        if lstring_utm.length > 5000:
            print(u, v, "edge length too long:", attr_dict, "returning!")
            return

    return G


###############################################################################
def wkt_to_G(params):
    """Execute all functions"""

    n_threads_max = 12

    (
        wkt_list,
        im_file,
        node_iter,
        edge_iter,
        min_spur_length_m,
        simplify_graph,
        rdp_epsilon,
        manually_reproject_nodes,
        out_file,
        graph_dir,
        geo_dir,
        n_threads,
    ) = params

    print("im_file:", im_file)

    pickle_protocol = 4

    t0 = time.time()

    node_loc_dic, edge_dic = wkt_list_to_nodes_edges(
        wkt_list, node_iter=node_iter, edge_iter=edge_iter
    )
    G0 = nodes_edges_to_G(node_loc_dic, edge_dic)

    # This graph will have a unique edge for each line segment, meaning that
    #  many nodes will have degree 2 and be in the middle of a long edge.

    # run clean_sub_graph() in 04_skeletonize.py?  - Nope, do it here
    # so that adding small terminals works better...

    G1 = clean_sub_graphs(G0, min_length=300, weight="length_pix")

    if len(G1) == 0:
        return G1

    # geo coords
    if im_file:
        # let's not over multi-thread a multi-thread
        if n_threads > 1:
            n_threads_tmp = 1
        else:
            n_threads_tmp = n_threads_max
        G1 = get_node_geo_coords(G1, im_file, n_threads=n_threads_tmp)

        # let's not over multi-thread a multi-thread
        if n_threads > 1:
            n_threads_tmp = 1
        else:
            n_threads_tmp = n_threads_max

        G1 = get_edge_geo_coords(G1, im_file, n_threads=n_threads_tmp)

        G_projected = osmnx_funcs.project_graph(G1)
        # get geom wkt (for printing/viewing purposes)
        for i, (u, v, attr_dict) in enumerate(G_projected.edges(data=True)):
            attr_dict["geometry_wkt"] = attr_dict["geometry"].wkt

        Gout = G_projected  # G_simp

    else:
        Gout = G1

    # ###########################################################################
    # # remove short edges?
    # # this is done in 04_skeletonize.remove_small_terminal()
    # t31 = time.time()
    # Gout = remove_short_edges(Gout, min_spur_length_m=min_spur_length_m)
    # t32 = time.time()
    # print("Time to remove_short_edges():", t32 - t31, "seconds")
    # ###########################################################################

    if simplify_graph:
        t7 = time.time()
        # 'geometry' tag breaks simplify, so maket it a wkt
        for i, (u, v, attr_dict) in enumerate(G_projected.edges(data=True)):
            if "geometry" in attr_dict.keys():
                attr_dict["geometry"] = attr_dict["geometry"].wkt

        G0 = ox.simplify_graph(Gout.to_directed())
        G0 = G0.to_undirected()

        # this break latlon for shape files and kmls, don't really need utm atm
        # Gout = osmnx_funcs.project_graph(G0, to_latlong=True)
        Gout = G0

        # BUG, GOOF, ERROR IN OSMNX PROJECT, SO NEED TO MANUALLY SET X, Y FOR NODES!!??
        if manually_reproject_nodes:
            # make sure geometry is utm for nodes?
            for i, (n, attr_dict) in enumerate(Gout.nodes(data=True)):
                attr_dict["x"] = attr_dict["utm_east"]
                attr_dict["y"] = attr_dict["utm_north"]

        keys_tmp = [
            "geometry_wkt",
            "geometry_pix",
            "geometry_latlon_wkt",
            "geometry_utm_wkt",
        ]
        for key_tmp in keys_tmp:
            for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
                if key_tmp not in attr_dict.keys():
                    continue

                if (i % 10000) == 0:
                    print(i, u, v)
                geom = attr_dict[key_tmp]

                if type(geom) == list:
                    # check if the list items are wkt strings, if so, create
                    #   linestrigs
                    if type(geom[0]) == str:
                        geom = [shapely.wkt.loads(ztmp) for ztmp in geom]
                    # merge geoms
                    geom_out = shapely.ops.linemerge(geom)

                elif type(geom) == str:
                    geom_out = shapely.wkt.loads(geom)

                else:
                    geom_out = geom

                # now straighten edge with rdp
                if rdp_epsilon > 0:
                    coords = list(geom_out.coords)
                    new_coords = rdp.rdp(coords, epsilon=rdp_epsilon)
                    geom_out_rdp = LineString(new_coords)
                    geom_out_final = geom_out_rdp
                else:
                    geom_out_final = geom_out

                len_out = geom_out_final.length

                # updata edge properties
                attr_dict[key_tmp] = geom_out_final

                # update length
                if key_tmp == "geometry_pix":
                    attr_dict["length_pix"] = len_out
                if key_tmp == "geometry_utm_wkt":
                    attr_dict["length_utm"] = len_out

        # assign 'geometry' tag to geometry_wkt
        # !! assign 'geometry' tag to geometry_utm_wkt
        key_tmp = "geometry_wkt"  # 'geometry_utm_wkt'
        for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
            line = attr_dict["geometry_utm_wkt"]
            if type(line) == str:  # or type(line) == unicode:
                attr_dict["geometry"] = shapely.wkt.loads(line)
            else:
                attr_dict["geometry"] = attr_dict[key_tmp]
            attr_dict["geometry_wkt"] = attr_dict["geometry"].wkt

            # set length
            attr_dict["length"] = attr_dict["geometry"].length

            # update wkt_pix?
            attr_dict["wkt_pix"] = attr_dict["geometry_pix"].wkt

            # update 'length_pix'
            attr_dict["length_pix"] = np.sum([attr_dict["length_pix"]])

    # create kml object
    kml = simplekml.Kml()
    for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
        coords = []

        line_str = str(attr_dict["geometry_latlon_wkt"])
        line_str = line_str[12:-1].split(",")
        for x_y_ in line_str:
            x_y_ = x_y_.strip()

            lon, lat = x_y_.split(" ")
            coords.append((lon, lat))

        ln = kml.newlinestring(name="Road", description="Road", coords=coords,)
        ln.extrude = 1
        ln.style.linestyle.color = simplekml.Color.blue
        ln.style.linestyle.width = 5

    kml.save(os.path.join(geo_dir, "roads.kml"))

    kml = simplekml.Kml()

    # record non-intersection nodes
    node_ends = [n for n, x in Gout.degree if x <= 1]

    for i, (n, attr_dict) in enumerate(Gout.nodes(data=True)):
        if n not in node_ends:

            pt = kml.newpoint(
                name="Intersection",
                description="Intersection",
                coords=[(attr_dict["lon"], attr_dict["lat"])],
            )

            ln.style.linestyle.color = simplekml.Color.yellow

    kml.save(os.path.join(geo_dir, "intersections.kml"))

    Gout.graph["N_nodes"] = len(Gout.nodes())
    Gout.graph["N_edges"] = len(Gout.edges())

    # get total length of edges
    tot_meters = 0
    for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
        tot_meters += attr_dict["length"]

    Gout.graph["Tot_edge_km"] = tot_meters / 1000

    # save graph
    nx.write_gpickle(Gout, out_file, protocol=pickle_protocol)

    # # save shapefile
    try:
        for node, data in Gout.nodes(data=True):
            if "osmid" in data:
                data["osmid_original"] = data.pop("osmid")

        ox.save_graph_shapefile(
            Gout, os.path.join(geo_dir, "shapefiles"), encoding="utf-8"
        )

    except Exception as e:
        print(e)
        print("Cannot save shapefile...")

    return  # Gout


################################################################################
def main():
    simplify_graph = True  # True # False
    pickle_protocol = 4  # 4 is most recent, python 2.7 can't read 4
    node_iter = 10000  # start int for node naming
    edge_iter = 10000  # start int for edge naming
    manually_reproject_nodes = False  # True
    n_threads = 12

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        cfg = json.load(f)
        config = Config(**cfg)

    # outut files
    raw_data_dir = os.path.join(config.path_data_root, config.raw_data_dir)

    csv_file = os.path.join(config.path_results_root, config.wkt_submission)
    graph_dir = os.path.join(config.path_results_root, config.graph_dir)
    geo_dir = os.path.join(config.path_results_root, config.geo_output_dir)

    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(geo_dir, exist_ok=True)

    min_spur_length_m = config.min_spur_length_m

    # read in wkt list
    df_wkt = pd.read_csv(csv_file)

    # iterate through image ids and create graphs
    t0 = time.time()
    image_ids = np.sort(np.unique(df_wkt["ImageId"]))
    nfiles = len(image_ids)
    print("image_ids:", image_ids)
    print("len image_ids:", len(image_ids))
    n_threads = min(n_threads, nfiles)

    params = []
    for i, image_id in enumerate(image_ids):
        out_file = os.path.join(graph_dir, image_id.split(".")[0] + ".gpickle")

        # for geo referencing, im_file should be the raw image
        print(image_id)
        if config.num_channels == 3:
            im_file = os.path.join(raw_data_dir, image_id + ".tif")

        # filter
        df_filt = df_wkt["WKT_Pix"][df_wkt["ImageId"] == image_id]
        wkt_list = df_filt.values

        if (len(wkt_list) == 0) or (wkt_list[0] == "LINESTRING EMPTY"):
            G = nx.MultiDiGraph()
            nx.write_gpickle(G, out_file, protocol=pickle_protocol)

        else:
            params.append(
                (
                    wkt_list,
                    im_file,
                    node_iter,
                    edge_iter,
                    min_spur_length_m,
                    simplify_graph,
                    config.rdp_epsilon,
                    manually_reproject_nodes,
                    out_file,
                    graph_dir,
                    geo_dir,
                    n_threads,
                )
            )

    # exectute
    if n_threads > 1:
        pool = Pool(n_threads)
        pool.map(wkt_to_G, params)
    else:

        wkt_to_G(params[0])

    tf = time.time()
    print("Time to run wkt_to_G.py: {} seconds".format(tf - t0))


###############################################################################
if __name__ == "__main__":
    main()
