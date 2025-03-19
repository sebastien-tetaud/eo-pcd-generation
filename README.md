# EO Point Cloud Generation

This repository contains a collection of Jupyter notebooks and utilities for generating 3D point clouds from Earth Observation (EO) data, including Digital Elevation Models (DEM), Sentinel-2 imagery, and LiDAR data.

## Overview

The project demonstrates various techniques for processing and visualizing geospatial data in 3D, with a focus on:

1. Accessing and visualizing DEM data from the DestinE Earth Data Hub (https://earthdatahub.destine.eu/getting-started)
2. Combining DEM data with Sentinel-2 imagery to create colorized 3D point clouds
3. Processing IGN HD LiDAR data for high-resolution terrain modeling
4. Integrating multiple data sources (DEM, Sentinel-2, and LiDAR) for comprehensive 3D models

## Repository Structure

```
.
├── notebooks/
│   ├── basic_part_1.ipynb         # Basic DEM visualization
│   ├── basic_part_2.ipynb         # Point cloud generation from DEM and Sentinel-2
│   └── basic_part_3.ipynb         # IGN HD LiDAR processing
├── src/
│   ├── data/                      # Data storage directory
│   │   ├── sentinel2/             # Sentinel-2 imagery
│   │   └── ign/                   # IGN LiDAR data
│   └── util/                      # Utility functions
│       └── general.py             # Common processing functions
└── README.md                      # This file
```

```bash
├── LICENSE
├── README.md
├── notebooks
│   ├── basic_part_1.ipynb          # Basic DEM visualization
│   ├── basic_part_2.ipynb          # Point cloud generation from DEM and Sentinel-2
│   ├── basic_part_3.py.py          # IGN HD LiDAR processing
│   ├── intermediate_part_1.ipynb   # Integration of multiple data sources
└── src
    ├── data                        # Data storage directory
    │   ├── grille.zip
    │   ├── ign/
    │   └── sentinel2/
    │       └── T32TLR_20241030T103151_TCI_20m.jp2
    ├── point_cloud_generator.py
    └── util/                       # Utility functions
        ├── file_downloader.py      # Utility to download files
        └── general.py              # Common processing functions
```

## Notebooks

### Basic Part 1: DEM Visualization

This notebook demonstrates how to:
- Access Copernicus DEM data from the Earth Data Hub
- Select and load specific geographic regions
- Create basic visualizations using matplotlib and cartopy
- Save the visualizations as images

### Basic Part 2: Point Cloud Generation from DEM and Sentinel-2

This notebook focuses on:
- Loading and preprocessing Sentinel-2 Level 2A imagery
- Combining it with DEM data to create 3D point clouds
- Handling large datasets through downsampling
- Generating and saving point clouds in PLY format
- Creating 3D meshes from point clouds

### Basic Part 3: IGN HD LiDAR Processing

This notebook covers:
- Downloading IGN LiDAR data tiles
- Reading and processing LiDAR data with PDAL
- Projecting data to UTM coordinates
- Assigning custom colors based on classification
- Downsampling and filtering point clouds
- Saving processed point clouds in PLY format

### Intermediate Part 1: Multi-Source Data Integration
This notebook demonstrates advanced techniques for:

- Integrating Sentinel-2, DEM, and LiDAR data into a single point cloud
- Visualizing data coverage using interactive Folium maps
- Processing multiple LiDAR tiles that intersect with a Sentinel-2 image
- Filtering LiDAR data by classification (e.g., ground, vegetation, buildings)
- Generating comprehensive 3D models from multiple data sources

## Utility Functions

The repository includes several utility classes to streamline data processing:

- `Sentinel2Reader`: For reading and preprocessing Sentinel-2 imagery
- `PcdGenerator`: For creating point clouds from DEM and imagery data
- `PointCloudHandler`: For manipulating and saving point clouds
- `IGNLidarProcessor`: For processing IGN LiDAR data

## Requirements

- xarray
- rasterio
- numpy
- pandas
- matplotlib
- cartopy
- open3d
- pdal
- geopandas
- cv2 (OpenCV)

## Getting Started

1. Clone this repository

```bash
git clone git@github.com:sebastien-tetaud/eo-pcd-generation.git
```
2. Install the required dependencies

```bash
conda create --name env python==3.13.1
conda activate env
pip install -r requiremts.txt
```

```bash
python -m pip install "xarray[complete]" -y
```

```bash
conda install -c conda-forge gdal -y
```

```bash
conda install -c conda-forge python-pdal -y
```

```bash
conda install -c conda-forge open3d -y
```

3. Set up your Earth Data Hub token as an environment variable:
```bash
export hdb_token=your_token_here
```
4. Run the notebooks in sequence to understand the workflow

## Usage Examples

### Accessing DEM Data

```python
import os
import xarray as xr

token = os.environ.get('hdb_token')
data = xr.open_dataset(
    f"https://edh:{token}@data.earthdatahub.destine.eu/copernicus-dem/GLO-30-v0.zarr",
    chunks={},
    engine="zarr",
)
```

### Generating a Point Cloud (local Data)

```python
from src.util.general import Sentinel2Reader, load_dem_utm, PcdGenerator

reader = Sentinel2Reader(filepath="path/to/sentinel2/image.jp2", preprocess=True)
dem_data = load_dem_utm(url=dem_url, bounds=reader.bounds, width=reader.width, height=reader.height)
pcd_gen = PcdGenerator(reader.data, dem_data["dem"])
pcd_gen.generate_point_cloud()
pcd_gen.downsample(sample_fraction=0.2)
```

## LiDAR Classification information

| Code | Description |
|------|-------------|
| 1 | Unclassified |
| 2 | Ground |
| 3 | Low vegetation (0-50 cm) |
| 4 | Medium vegetation (50 cm-1.50 m) |
| 5 | High vegetation (>1.50 m) |
| 6 | Building |
| 9 | Water |
| 17 | Bridge deck |
| 64 | Permanent overlay |
| 66 | Virtual points |
| 67 | Miscellaneous - built |

## License

Apache License 2.0

## Acknowledgments

- [Copernicus DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model)
- [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [IGN LiDAR HD](https://geoservices.ign.fr/lidarhd)
- [DestinE Earth Data Hub](https://earthdatahub.destine.eu/)