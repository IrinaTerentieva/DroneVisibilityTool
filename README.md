# Drone Visibility Analysis Tool 🚁

A Python tool for analyzing drone visibility from ground observation points using Digital Surface Models (DSM). The tool casts rays at specified elevation angles to determine line-of-sight visibility areas, helping optimize drone flight paths and observation point placement.

## Features

- **Ray-casting visibility analysis**: Casts rays at specified elevation angles to find terrain obstructions
- **Spatial polygon generation**: Creates visibility polygons showing observable areas
- **Cluster-aware analysis**: Maintains staging point cluster information throughout analysis
- **GeoPackage export**: Outputs results in QGIS-compatible format
- **Configurable parameters**: Uses Hydra for flexible configuration management
- **Depth-checking algorithm**: Validates obstructions to avoid false positives from single pixels

## Installation

```
pip install -r requirements.txt
```

### Prerequisites
- Python 3.8+
- GDAL libraries installed on your system

## Quick Start

Configure your analysis parameters in configs/angle.yaml:

```
paths:
  dsm_path: /path/to/your/dsm.tif
  staging_gpkg: /path/to/staging_points.gpkg
  output_gpkg: /path/to/output.gpkg

params:
  drone_height_agl: 120.0  # Drone height above ground level (meters)
  elevation_angle: 5.0     # Elevation angle for ray casting (degrees)
  max_distance: 3000       # Maximum analysis distance (meters)
  num_rays: 360           # Number of rays (360 = 1° increments)
```

## How It Works
The tool analyzes drone visibility by:

Loading staging points: Reads observation points from a GeoPackage file
Ray casting: For each point, casts rays at the specified elevation angle in all directions
Obstruction detection: Finds where each ray intersects with terrain (considering obstruction depth)
Polygon creation: Connects intersection points to create visibility polygons
Export: Saves results as GeoPackage layers with cluster information preserved

### Algorithm Details

Elevation Angle: Rays are cast upward at the specified angle from observer eye level (1.7m above ground)
Depth Checking: Obstructions must span multiple pixels to be considered valid (avoids noise)
No-data Handling: DSM no-data values are handled gracefully
The tool generates a GeoPackage with two layers:

staging_points: Original observation points with metadata

staging_id: Unique identifier
cluster: Cluster assignment (if available)
staging_elev: Ground elevation at point


visibility_zones_5deg: Visibility polygons

staging_id: Links to staging points
cluster: Cluster assignment
visibility_area_ha: Area in hectares
elevation_angle: Angle used for analysis

## Usage Instructions:

1. Clone and setup:
```
git clone <your-repo>
cd drone-visibility-analysis
pip install -e .
```

2. Configure analysis:
Edit configs/angle/default.yaml with your paths and parameters

3. Run analysis:
```
python src/main.py
```

4. Run with overrides:
```
python src/main.py params.elevation_angle=10 params.max_distance=5000
```

5. Run multiple configurations:
```
python src/main.py --multirun params.elevation_angle=5,10,15
```

## 🤝 Acknowledgments

Developed by:

    Applied Geospatial Research Group, University of Calgary
      https://www.appliedgrg.ca/

    Falcon & Swift Geomatics Ltd
      https://www.falconandswift.ca/

---
