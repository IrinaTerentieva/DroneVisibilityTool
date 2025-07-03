# Drone Visibility Analysis Tools 🚁

A comprehensive Python toolkit for analyzing drone visibility from ground observation points using Digital Surface Models (DSM). The toolkit includes two complementary analysis methods: fixed-angle ray casting and dynamic drone-target visibility calculation.

## 🌟 Features

### Core Capabilities
- **Ray-casting visibility analysis**: Determines line-of-sight visibility areas
- **Spatial polygon generation**: Creates visibility polygons showing observable areas
- **Cluster-aware analysis**: Maintains staging point cluster information throughout analysis
- **GeoPackage export**: Outputs results in QGIS-compatible format
- **Configurable parameters**: Uses Hydra for flexible configuration management
- **Depth-checking algorithm**: Validates obstructions to avoid false positives from single pixels
- **Adaptive positioning**: Automatically adjusts drone distance for realistic visibility (drone-target mode)

### Two Analysis Modes

#### 1. **Fixed-Angle Analysis** (`SpatialDroneVisibilityAnalyzer`)
- Casts rays at a specified elevation angle (e.g., 5°) in all directions
- Identifies terrain obstacles blocking the line of sight
- Creates visibility polygons showing observable areas at fixed angles
- Useful for regulatory angle checks and worst-case envelope analysis

#### 2. **Drone-Target Analysis** (`DroneAngleVisibilityAnalyzer`)
- Calculates the actual angle needed to see a drone at specific positions
- Places drones at target distances and heights above ground level
- Uses minimum elevation in 5m buffer for realistic drone positioning
- **Adaptive positioning**: Automatically moves drone closer if not visible
- Creates realistic visibility polygons based on actual drone positions
- Perfect for flight planning and operational scenarios

## 📏 Key Differences Between Analysis Modes

| Feature                    | **Fixed-Angle Analyzer**                                     | **Drone-Target Analyzer**                                                   |
|---------------------------|---------------------------------------------------------------|------------------------------------------------------------------------------|
| **Elevation Angle**       | Fixed (e.g., 5° from staging point)                          | Computed dynamically using observer → drone geometry                        |
| **Drone Position**        | Not modeled                                                   | Y meters above terrain (at each location)                                   |
| **Drone Elevation Source**| Not applicable                                                | Min elevation in a 5m radius buffer under each candidate drone location     |
| **Visibility Criterion**  | Beam intersects terrain at fixed angle                       | Line-of-sight to drone is blocked by terrain along the path                |
| **Use Case**              | Regulatory angle checks, worst-case envelope                  | Realistic flight line-of-sight modeling for actual drone routes             |
| **Terrain Sampling**      | Single pixel per ray step                                     | Uses a local terrain minimum in 5m radius under drone position              |
| **Beam Type**             | Fixed-angle beam casting                                      | Dynamic beam toward target location                                         |
| **Output**                | Polygon of visible areas from fixed angle                     | Polygon of areas where drone is visible from ground observer                |
| **Adaptive Mode**         | Not available                                                 | Can move drone closer until visible                                         |

## 🔍 Understanding the Difference: Sky Visibility vs. Drone Visibility

![Sky vs Drone Visibility Analysis](examples/drone_sky_visibility.png)

The image above illustrates the fundamental difference between our two analysis methods using real terrain data:

### Terrain Context
- **Eastern area**: Uphill terrain (higher elevation)
- **Western area**: Valley approximately 200m below the hilltop
- **Staging point**: Located on the mid-slope (drone operator position)

### Key Visibility Scenarios

The analysis reveals two critical scenarios that demonstrate why both tools are necessary:

#### 🔴 Point 1: Sky Not Visible, Drone Visible
- **Location**: Uphill (eastern) direction
- **Sky visibility (5° angle)**: ❌ Blocked by rising terrain
- **Drone visibility**: ✅ Visible flying 120m above the slope
- **Why**: The drone flies high enough above the terrain to be seen over the hill, even though the 5° sky view is blocked

#### 🟢 Point 2: Sky Visible, Drone Not Visible  
- **Location**: Downhill (western) direction toward the valley
- **Sky visibility (5° angle)**: ✅ Clear view above horizon
- **Drone visibility**: ❌ Not visible despite clear sky
- **Why**: The drone at 120m AGL is actually below the observer's line of sight due to the steep downward slope

### Why This Matters

This comparison demonstrates that:
- **Fixed-angle analysis** (5° sky visibility) doesn't predict actual drone visibility
- **Drone-target analysis** accounts for terrain elevation changes and actual drone position
- In mountainous terrain, a drone can be visible when flying uphill despite blocked sky views
- Conversely, clear sky visibility doesn't guarantee drone visibility in valleys

**Bottom line**: For safe drone operations, you need to analyze actual drone positions, not just sky visibility angles.

## 📋 Requirements

- Python 3.8+
- GDAL libraries installed on your system
- Required Python packages:
  ```
  numpy>=1.21.0
  rasterio>=1.3.0
  geopandas>=0.13.0
  shapely>=2.0.0
  matplotlib>=3.5.0
  pandas>=1.5.0
  scipy>=1.9.0
  hydra-core>=1.3.0
  omegaconf>=2.3.0
  ```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/drone-visibility-tools.git
   cd drone-visibility-tools
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

## 📖 Quick Start

### Fixed-Angle Analysis

Configure your analysis parameters in `configs/angle.yaml`:
```yaml
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

Run analysis:
```bash
python src/main.py
```

### Drone-Target Analysis

Configure parameters in `configs/drone_angle.yaml`:
```yaml
drone_params:
  drone_distance: 2000.0      # Target distance to place drone
  drone_height_agl: 120.0     # Drone height above ground level
  adaptive_positioning: true  # Enable adaptive positioning
  step_back: 200.0           # Distance to move closer if not visible
  min_distance: 100.0        # Minimum distance to try
```

Run analysis:
```bash
python main_drone_angle.py
```

## 🔍 How It Works

### Fixed-Angle Analysis Algorithm
1. **Loading staging points**: Reads observation points from a GeoPackage file
2. **Ray casting**: For each point, casts rays at the specified elevation angle in all directions
3. **Obstruction detection**: Finds where each ray intersects with terrain (considering obstruction depth)
4. **Polygon creation**: Connects intersection points to create visibility polygons
5. **Export**: Saves results as GeoPackage layers with cluster information preserved

### Drone-Target Analysis Algorithm
1. **Drone placement**: Positions drone at target distance and height above ground level
2. **Terrain sampling**: Uses minimum elevation in 5m buffer for realistic drone positioning
3. **Angle calculation**: Computes angle from observer (1.7m eye height) to drone
4. **Line-of-sight check**: Traces ray from observer to drone checking for obstructions
5. **Adaptive positioning** (if enabled):
   - If blocked, moves drone closer by `step_back` distance
   - Repeats until visible or minimum distance reached
6. **Polygon creation**: Creates polygon showing actual drone visibility

### Algorithm Details
- **Observer height**: 1.7m above ground (eye level)
- **Depth checking**: Obstructions must span multiple pixels to be considered valid (avoids noise)
- **No-data handling**: DSM no-data values are handled gracefully
- **Terrain buffer**: Drone-target mode uses 5m radius minimum elevation for stability

## 📊 Output

Both tools generate GeoPackage files with multiple layers:

### Staging Points Layer
- `staging_id`: Unique identifier
- `cluster`: Cluster assignment (if available)
- `staging_elev`: Ground elevation at point
- Analysis parameters

### Visibility Zones Layer
- Visibility polygons
- `visibility_area_ha`: Area in hectares
- `elevation_angle`: Angle used for analysis (fixed-angle mode)
- `mean_angle_deg`: Average angle to drone (drone-target mode)
- `percent_rays_clear`: Percentage of unobstructed rays (drone-target mode)
- `adaptation_pct`: Percentage of rays requiring closer positioning (adaptive mode)

## 💡 Usage Examples

### Command Line Overrides

```bash
# Fixed angle analysis with 10° angle
python src/main.py params.elevation_angle=10.0

# Drone analysis at 3km distance
python main_drone_angle.py drone_params.drone_distance=3000

# Disable adaptive positioning
python main_drone_angle.py drone_params.adaptive_positioning=false
```

### Batch Analysis

```bash
# Test multiple angles
python src/main.py --multirun params.elevation_angle=5,10,15,20

# Test multiple drone distances
python main_drone_angle.py --multirun drone_params.drone_distance=1000,2000,3000,4000
```

## 🎯 Use Cases

### Fixed-Angle Analysis
- **Regulatory compliance**: Verify minimum elevation angles for drone operations
- **Terrain assessment**: Identify challenging topography
- **Worst-case planning**: Conservative visibility envelopes
- **Infrastructure planning**: Tower and antenna placement

### Drone-Target Analysis
- **Flight planning**: Determine actual visibility for specific drone routes
- **Search and rescue**: Find optimal observation points
- **Wildlife monitoring**: Plan observation posts with maximum coverage
- **Emergency response**: Identify safe operating areas
- **Delivery operations**: Verify line-of-sight for urban deliveries

## 🤝 Acknowledgments

Developed by:

- **Applied Geospatial Research Group (AGRG), University of Calgary**
  - https://www.appliedgrg.ca/
  
- **Falcon & Swift Geomatics Ltd**
  - https://www.falconandswift.ca/

Built with:
- [Hydra](https://hydra.cc/) for configuration management
- [Rasterio](https://rasterio.readthedocs.io/) for efficient DSM processing
- [GeoPandas](https://geopandas.org/) for spatial data handling
- [Matplotlib](https://matplotlib.org/) for visualization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use these tools in your research, please cite:
```bibtex
@software{drone_visibility_tools,
  title = {Drone Visibility Analysis Tools},
  author = {Applied Geospatial Research Group and Falcon & Swift Geomatics},
  year = {2024},
  organization = {University of Calgary},
  url = {https://github.com/yourusername/drone-visibility-tools}
}
```

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact AGRG: https://www.appliedgrg.ca/contact
- Contact Falcon & Swift: https://www.falconandswift.ca

---

**Happy Flying! 🚁✨**

*Advancing drone operations through intelligent visibility analysis*