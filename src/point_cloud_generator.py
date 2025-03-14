#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: point_cloud_generator.py
Description: This script reads Sentinel-2 satellite data,
loads corresponding DEM data, generates a point cloud,
down-samples it, and creates a 3D mesh.
Author: Your Name
Date: 2025-03-14
License: Apache 2.0
"""

import os

from util.general import (PcdGenerator, PointCloudHandler, Sentinel2Reader,
                          load_dem_utm)

# Retrieve authentication token
token = os.environ.get('hdb_token')

# Define the Sentinel-2 product path
product_path = "/home/ubuntu/project/destine-godot-mvp/src/sentinel2-data/\
    T32TLR_20241030T103151_TCI_20m.jp2"

# Read Sentinel-2 data
reader = Sentinel2Reader(filepath=product_path, preprocess=True)
bounds = reader.bounds
width = reader.width
height = reader.height

# Load DEM data
parameter = 'dem'
dem_data = load_dem_utm(token, parameter, bounds, width, height)

# Initialize and generate point cloud
pcd_gen = PcdGenerator(reader.data, dem_data)
pcd_gen.generate_point_cloud()
pcd_gen.downsample(sample_fraction=0.20)

# Process and save point cloud and mesh
handler = PointCloudHandler(pcd_gen.df)
handler.to_open3d()
handler.generate_mesh(depth=9)
handler.save_point_cloud("point_cloud.ply")
handler.save_mesh("mesh.glb")
