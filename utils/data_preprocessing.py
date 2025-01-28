import pandas as pd
import geopandas as gpd
import torch
import numpy as np
import os
import random
random.seed(2024)
import sys

sys.path.append("../")
from models.fourier_encoder import GeometryFourierEncoder


def normalize_coordinate(value, min_val, max_val):
    """Normalize a coordinate to the range [0, 2]."""
    """Add -1 to shift the range to [-1, 1]."""
    return 2 * (value - min_val) / (max_val - min_val) - 1


def normalize_geometry_polygon(geometry, min_x, max_x, min_y, max_y):
    if geometry.is_empty:
        return geometry
    normalized_coords = [(normalize_coordinate(x, min_x, max_x), normalize_coordinate(y, min_y, max_y)) for x, y in
                         geometry.exterior.coords]
    return normalized_coords


def normalize_geometry_linestring(geometry, min_x, max_x, min_y, max_y):
    if geometry.is_empty:
        return geometry
    normalized_coords = [(normalize_coordinate(x, min_x, max_x), normalize_coordinate(y, min_y, max_y)) for x, y in
                         geometry.coords]
    return normalized_coords


def read_urban_dataset(root_dir, dataset_name):
    # Load original dataset file
    # TODO: We need to unify the file_name, the type of objects saved in each file,
    #  and the column name of the geometry in the original data file and the gdf objects.
    # region (some of the regions could be 3D)
    if dataset_name == "NewYork":
        regions_file = f"{root_dir}/{dataset_name}/regions.pkl"
        regions_pd = pd.DataFrame(pd.read_pickle(regions_file))
        regions_pd = regions_pd.rename(columns={"shape": "geometry"})
        # TODO: Even if we claim geometry is this region_geometry_marker (e.g., "shape" for NewYork,
        #  the created gdf object still doesn't have "geometry" column but only "shape" column)
        regions_gdf = gpd.GeoDataFrame(regions_pd, geometry=r"geometry")
        regions_gdf["region_id"] = regions_gdf.index
    elif dataset_name == "Singapore":
        regions_file = f"{root_dir}/{dataset_name}/regions_updated.pkl"
        regions_gdf = pd.read_pickle(regions_file)
        regions_gdf["region_id"] = regions_gdf.index
    # buildings
    if dataset_name == "NewYork":
        buildings_file = f"{root_dir}/{dataset_name}/buildings.pkl"
        buildings_gdf = pd.read_pickle(buildings_file)
        buildings_gdf["building_id"] = buildings_gdf.index
    elif dataset_name == "Singapore":
        buildings_file = f"{root_dir}/{dataset_name}/building.pkl"
        buildings_pd = pd.DataFrame(pd.read_pickle(buildings_file))
        buildings_pd = buildings_pd.rename(columns={"shape": "geometry"})
        buildings_gdf = gpd.GeoDataFrame(buildings_pd, geometry="geometry")
        buildings_gdf["building_id"] = buildings_gdf.index
    # poi
    poi_file = f"{root_dir}/{dataset_name}/poi_updated.pkl"
    poi_gdf = pd.read_pickle(poi_file)
    poi_gdf["poi_id"] = poi_gdf.index
    # roads
    roads_file = f"{root_dir}/{dataset_name}/roads.pkl"
    if dataset_name == "NewYork":
        roads_gdf = pd.read_pickle(roads_file)
    elif dataset_name == "Singapore":
        roads_pd = pd.read_pickle(roads_file)
        #roads_pd = roads_pd.rename(columns={"shape": "geometry"})
        #TODO: For Singapore, its road.pkl contains two types of geometries, "geometry" is the GPS coordinates system, while "geometry" is its own coordinates system
        roads_gdf = gpd.GeoDataFrame(roads_pd, geometry="shape")
    roads_gdf["road_id"] = roads_gdf.index

    return regions_gdf, buildings_gdf, poi_gdf, roads_gdf


def urban_dataset_process(root_dir, dataset_name):
    print(f"Processing {dataset_name}...")
    # Load original dataset file
    regions_gdf, buildings_gdf, poi_gdf, roads_gdf = read_urban_dataset(root_dir, dataset_name)

    # Get the regions' bounds
    min_x, min_y, max_x, max_y = regions_gdf["geometry"].total_bounds

    # POIs (points)
    print(f"POI (points)")
    # Normalize the coordinates
    poi_gdf["coordinates"] = poi_gdf["geometry"].apply(lambda geom: list(geom.coords[0]))
    poi_gdf['normalized_coordinates'] = poi_gdf['coordinates'].apply(
        lambda coord: [normalize_coordinate(coord[0], min_x, max_x), normalize_coordinate(coord[1], min_y, max_y)])
    poi_gdf.to_pickle(f"{root_dir}/{dataset_name}/poi_normalized.pkl")
    print("POI normalization finished.")

    pois_to_regions = {}
    # Loop through each region and find the pois that fall within it
    for idx, region in regions_gdf.iterrows():
        # Get the region's geometry
        region_geom = region["geometry"]
        # Use spatial join to find pois within this region
        pois_in_region = poi_gdf[poi_gdf.geometry.within(region_geom)]
        # Assign the poi IDs or geometries to the corresponding region in the dictionary
        pois_to_regions[region['region_id']] = pois_in_region['poi_id'].tolist()
    # add the list of pois to the regions_gdf
    regions_gdf['pois'] = regions_gdf['region_id'].map(pois_to_regions)
    print("Incorporated POI into regions")

    # Roads (polylines)
    print("Roads (polylines)")
    roads_gdf['coordinates'] = roads_gdf['geometry'].apply(lambda geom: list(geom.coords))
    roads_gdf['normalized_coordinates'] = roads_gdf['geometry'].apply(
        lambda geom: normalize_geometry_linestring(geom, min_x, max_x, min_y, max_y))
    print("Roads normalization finished.")
    # pad normalized coordinates to max length
    max_length = max(roads_gdf['normalized_coordinates'].apply(len))
    roads_gdf['padded_coordinates'] = roads_gdf['coordinates'].apply(
        lambda coords: coords + [(0, 0)] * (max_length - len(coords)))
    roads_gdf['padded_norm_coordinates'] = roads_gdf['normalized_coordinates'].apply(
        lambda coords: coords + [(0, 0)] * (max_length - len(coords)))
    roads_gdf['len'] = roads_gdf['normalized_coordinates'].apply(len)
    print(
        f"After padding to max length {max_length}, we add {max_length * roads_gdf.shape[0] - roads_gdf['len'].sum()} points to {roads_gdf.shape[0]} roads. "
        f"Invalid ratio is {(max_length * roads_gdf.shape[0] - roads_gdf['len'].sum()) / roads_gdf.shape[0]}")
    # save roads
    roads_gdf.to_pickle(f"{root_dir}/{dataset_name}/roads_normalized.pkl")

    roads_to_regions = {}
    # Loop through each region and find the roads that fall within it
    for idx, region in regions_gdf.iterrows():
        # Get the region's geometry
        region_geom = region["geometry"]
        # Use spatial join to find roads within this region
        roads_in_region = roads_gdf[roads_gdf.geometry.within(region_geom)]
        # Assign the road IDs or geometries to the corresponding region in the dictionary
        roads_to_regions[region['region_id']] = roads_in_region['road_id'].tolist()
    # add the list of roads to the regions_gdf
    regions_gdf['roads'] = regions_gdf['region_id'].map(roads_to_regions)
    print("Incorporated roads into regions")

    # Buildings (polygons)
    print("Buildings (polygons)")
    # simplify geometries of buildings
    buildings_gdf['simple_shape'] = buildings_gdf['geometry'].simplify(tolerance=0.1)
    buildings_gdf['coordinates'] = buildings_gdf['simple_shape'].apply(lambda geom: list(geom.exterior.coords))
    buildings_gdf['normalized_coordinates'] = buildings_gdf['simple_shape'].apply(
        lambda geom: normalize_geometry_polygon(geom, min_x, max_x, min_y, max_y))
    print("Buildings normalization finished.")
    # pad normalized coordinates to max length
    max_length = max(buildings_gdf['normalized_coordinates'].apply(len))
    buildings_gdf['padded_coordinates'] = buildings_gdf['coordinates'].apply(
        lambda coords: coords + [(0, 0)] * (max_length - len(coords)))
    buildings_gdf['padded_norm_coordinates'] = buildings_gdf['normalized_coordinates'].apply(
        lambda coords: coords + [(0, 0)] * (max_length - len(coords)))
    buildings_gdf['len'] = buildings_gdf['normalized_coordinates'].apply(len)
    # TODO: There are some invalid polygons in NewYork dataset
    buildings_gdf = buildings_gdf.loc[buildings_gdf.geometry.is_valid]
    buildings_gdf = buildings_gdf.reset_index(drop=True)
    # Don't forget to reset ["building_id"] after reset index.
    buildings_gdf["building_id"] = buildings_gdf.index
    print(
        f"After padding to max length {max_length}, we add {max_length * buildings_gdf.shape[0] - buildings_gdf['len'].sum()} points to {buildings_gdf.shape[0]} buildings. "
        f"Invalid ratio is {(max_length * buildings_gdf.shape[0] - buildings_gdf['len'].sum()) / buildings_gdf.shape[0]}")
    # save the updated buildings data with normalized coordinates
    buildings_gdf.to_pickle(f"{root_dir}/{dataset_name}/buildings_normalized.pkl")

    buildings_to_regions = {}
    # Loop through each region and find the buildings that fall within it
    for idx, region in regions_gdf.iterrows():
        # Get the region's geometry
        region_geom = region["geometry"]
        # Use spatial join to find buildings within this region
        buildings_in_region = buildings_gdf[buildings_gdf.geometry.within(region_geom)]
        # Assign the building IDs or geometries to the corresponding region in the dictionary
        buildings_to_regions[region['region_id']] = buildings_in_region['building_id'].tolist()
    regions_gdf['buildings'] = regions_gdf['region_id'].map(buildings_to_regions)
    print("Incorporated buildings into regions")

    # save regions_gdf with all the associated data
    regions_gdf.to_pickle(f"{root_dir}/{dataset_name}/regions_with_pois_buildings_roads.pkl")


def load_features(data_path, args):
    encoder = args.encoder_type
    if encoder == 'poly2vec':
        features_dir = data_path + 'poly2vec_features_backup/'
        # check if directory exists or create it
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
            features = preprocess_dataset(data_path, args)
            # Save features
            np.save(features_dir + 'features.npy', features)
        # Load features
        features = np.load(features_dir + 'features.npy', allow_pickle=True)


def compute_region_features(args):
    """Compute region features based on POIs, roads, and buildings.

    Args:
        args (args): Arguments containing encoder type and other parameters.

    Returns:
        np.ndarray: Region features.
    """
    # Load POI, road, and building features
    #TODO: Based on the previous poly2vec, the geom_type is given by the config.json, so technically we're not allowed to deal with regions with multi-type geometries.
    # poi_features = np.load(args.data_path + 'poly2vec_features_backup/poi_features_normalized.npy', allow_pickle=True)
    # road_features = np.load(args.data_path + 'poly2vec_features_backup/road_features_normalized.npy', allow_pickle=True)
    building_features = np.load(args.data_path + 'poly2vec_features/building_features_normalized.npy',
                                allow_pickle=True)

    regions = np.load(args.data_path + 'regions_with_pois_buildings_roads.pkl', allow_pickle=True)
    poi_idx = regions['pois']
    buildings_idx = regions['buildings']
    roads_idx = regions['roads']
    print(f"We have {building_features.shape} building features and {len(regions)} regions")
    region_features = []
    for i in range(len(regions)):
        region_feature = []
        # Get features by index and concatenate
        # if len(poi_idx[i]) > 0:
        #     region_feature.append(poi_features[poi_idx[i]])
        if len(buildings_idx[i]) > 0:
            region_feature.append(building_features[buildings_idx[i]])
        # if len(roads_idx[i]) > 0:
        #     region_feature.append(road_features[roads_idx[i]])

        # Concatenate all features for the region along the first axis
        concatenated_features = np.concatenate(region_feature, axis=0) if region_feature else np.zeros(
            (0, building_features.shape[1]))
        region_features.append(concatenated_features)
    print(len(region_features))

    # Find maximum length for padding
    max_length = max([feature.shape[0] for feature in region_features])
    print(max_length)

    # Pad region features to the max length
    padded_region_features = np.array([
        np.pad(feature, ((0, max_length - feature.shape[0]), (0, 0)), 'constant') for feature in region_features
    ])

    # Convert to tensor
    print(padded_region_features.shape)
    padded_region_features = torch.tensor(padded_region_features, dtype=torch.complex128)
    print(padded_region_features.shape)

    # Create attention masks (1 for valid tokens, 0 for padding)
    attention_masks = torch.tensor([
        [1] * feature.shape[0] + [0] * (max_length - feature.shape[0]) for feature in region_features
    ], dtype=torch.float32)

    if args.task == 'population':
        y = pd.to_numeric(regions['population'], errors='coerce')
        y = y.fillna(0.0)  # convert nan values to 0
        y = torch.tensor(y, dtype=torch.float32)
    else:
        y = regions['land_use']
        y = torch.tensor(y, dtype=torch.float32)

    return padded_region_features, attention_masks, y


def preprocess_dataset(data_path, args):
    """Preencode POIs, roads, and buildings using the specified encoder.

    Args:
        data_path (string): Path to the dataset.
        args (args): Arguments containing encoder type and other parameters.

    Returns:
        _type_: _description_
    """
    ### Start with region data
    regions = np.load(data_path + 'regions_with_pois_buildings_roads.pkl', allow_pickle=True)
    if args.encoder_type == 'poly2vec':
        encoder = GeometryFourierEncoder(args)
        # check if POIs are already encoded
        if os.path.exists(data_path + 'poly2vec_features/poi_features_normalized.npy'):
            #poi_features = np.load(data_path + 'poly2vec_features_backup/poi_features_normalized.npy', allow_pickle=True)
            pass
        else:
            pois = np.load(data_path + 'poi_normalized.pkl', allow_pickle=True)
            pois_coords = pois['normalized_coordinates']
            pois_tensor = torch.tensor(pois_coords, dtype=torch.complex128)
            print("Encoding POIs...")
            poi_features = encoder.point_encoder(pois_tensor)
            poi_features = poi_features.reshape(len(pois_tensor), -1)
            np.save(data_path + 'poly2vec_features/poi_features_normalized.npy', poi_features)
        if os.path.exists(data_path + 'poly2vec_features/road_features_normalized.npy'):
            #road_features = np.load(data_path + 'poly2vec_features_backup/road_features_normalized.npy', allow_pickle=True)
            pass
        else:
            roads = np.load(data_path + 'roads_normalized.pkl', allow_pickle=True)
            road_coords = roads['padded_norm_coordinates'].to_list()
            road_tensor = torch.tensor(road_coords, dtype=torch.float64)
            road_length_tensor = torch.tensor(roads['len'].to_list())
            print("Encoding Roads...")
            road_features = encoder.polyline_encoder(road_tensor, road_length_tensor)
            road_features = road_features.reshape(len(road_tensor), -1)
            np.save(data_path + 'poly2vec_features/road_features_normalized.npy', road_features)
        if os.path.exists(data_path + 'poly2vec_features/building_features_normalized.npy'):
            #building_features = np.load(data_path + 'poly2vec_features_backup/building_features_normalized.npy', allow_pickle=True)
            pass
        else:
            buildings = np.load(data_path + 'buildings_normalized.pkl', allow_pickle=True)
            building_coords = buildings['padded_norm_coordinates']
            building_tensor = torch.tensor(building_coords, dtype=torch.float64)
            buildings_length_tensor = torch.tensor(buildings['len'])
            print("Encoding Buildings...")
            building_features = encoder.polygon_encoder(building_tensor, buildings_length_tensor)
            building_features = building_features.reshape(len(building_tensor), -1)
            np.save(data_path + 'poly2vec_features/building_features_normalized.npy', building_features)

    return regions


if __name__ == "__main__":
    root_dir = "/home/jiali/Poly2Vec/data"
    for dataset_name in ["NewYork", "Singapore"]:
        print(dataset_name)

