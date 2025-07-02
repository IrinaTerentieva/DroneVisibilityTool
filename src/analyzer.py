import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from typing import List, Tuple, Dict, Optional
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SpatialDroneVisibilityAnalyzer:
    """
    Analyzes drone visibility from ground observation points using ray casting.
    """

    def __init__(self, dsm_path: str, drone_height_agl: float = 120.0):
        """
        Initialize the spatial drone visibility analyzer.

        Args:
            dsm_path: Path to DSM raster file
            drone_height_agl: Drone height above ground level in meters
        """
        self.dsm_path = dsm_path
        self.drone_height_agl = drone_height_agl

        logger.info(f"Loading DSM from: {dsm_path}")

        # Load DSM metadata
        with rasterio.open(dsm_path) as src:
            self.dsm = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
            self.nodata = src.nodata
            self.pixel_size = abs(self.transform.a)

            logger.info(f"DSM shape: {self.dsm.shape}")
            logger.info(f"DSM resolution: {self.pixel_size:.2f}m")
            logger.info(f"DSM CRS: {self.crs}")

    def _world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        row, col = rowcol(self.transform, x, y)
        return int(row), int(col)

    def _pixel_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates."""
        return xy(self.transform, row, col)

    def _get_elevation_safe(self, row: int, col: int) -> float:
        """Safely get elevation value with bounds checking."""
        if 0 <= row < self.dsm.shape[0] and 0 <= col < self.dsm.shape[1]:
            return float(self.dsm[row, col])
        return np.nan

    def _check_obstruction_depth(self, hit_row: int, hit_col: int,
                                 beam_elevation: float, min_depth_pixels: int = 3) -> bool:
        """
        Check if obstruction has sufficient depth to be considered valid.

        Args:
            hit_row, hit_col: Pixel coordinates of hit point
            beam_elevation: Elevation of beam at hit point
            min_depth_pixels: Minimum pixels the obstruction should span

        Returns:
            True if obstruction is significant enough
        """
        if not (0 <= hit_row < self.dsm.shape[0] and 0 <= hit_col < self.dsm.shape[1]):
            return False

        obstruction_count = 0
        search_radius = min_depth_pixels

        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                check_row = hit_row + dr
                check_col = hit_col + dc

                if 0 <= check_row < self.dsm.shape[0] and 0 <= check_col < self.dsm.shape[1]:
                    terrain_elev = self.dsm[check_row, check_col]
                    if terrain_elev >= beam_elevation:
                        obstruction_count += 1

        return obstruction_count >= min_depth_pixels

    def _cast_elevation_beam(self, start_x: float, start_y: float,
                             start_elevation: float, bearing_deg: float,
                             elevation_angle_deg: float, max_distance: float = 5000.0,
                             min_depth_pixels: int = 3,
                             observer_height: float = 1.7) -> Optional[Tuple[float, float, float]]:
        """
        Cast a beam at specified elevation angle and find obstruction.

        Args:
            start_x, start_y: Starting coordinates
            start_elevation: Ground elevation at start
            bearing_deg: Horizontal bearing in degrees
            elevation_angle_deg: Elevation angle in degrees
            max_distance: Maximum casting distance
            min_depth_pixels: Minimum obstruction depth
            observer_height: Observer eye height above ground

        Returns:
            Tuple of (distance, hit_x, hit_y) or None if no hit
        """
        bearing_rad = np.radians(bearing_deg)
        elevation_rad = np.radians(elevation_angle_deg)

        # Observer eye level
        eye_level = start_elevation + observer_height

        step_size = self.pixel_size
        num_steps = int(max_distance / step_size)

        for i in range(1, num_steps + 1):
            horizontal_distance = i * step_size

            # Calculate position
            x = start_x + horizontal_distance * np.sin(bearing_rad)
            y = start_y + horizontal_distance * np.cos(bearing_rad)

            # Beam elevation increases with distance and angle
            beam_elevation = eye_level + horizontal_distance * np.tan(elevation_rad)

            # Get terrain elevation
            row, col = self._world_to_pixel(x, y)
            terrain_elevation = self._get_elevation_safe(row, col)

            if np.isnan(terrain_elevation):
                continue

            # Check for intersection
            if beam_elevation <= terrain_elevation:
                # Validate obstruction depth
                if self._check_obstruction_depth(row, col, beam_elevation, min_depth_pixels):
                    return horizontal_distance, x, y

        return None

    def _create_visibility_polygon(self, staging_x: float, staging_y: float,
                                   staging_elevation: float, elevation_angle: float,
                                   num_rays: int = 360, max_distance: float = 5000.0,
                                   **kwargs) -> Polygon:
        """
        Create visibility polygon for given elevation angle.

        Args:
            staging_x, staging_y: Staging point coordinates
            staging_elevation: Ground elevation at staging point
            elevation_angle: Elevation angle in degrees
            num_rays: Number of rays to cast
            max_distance: Maximum analysis distance
            **kwargs: Additional parameters passed to ray casting

        Returns:
            Shapely Polygon representing visible area
        """
        visibility_points = []

        for ray_idx in range(num_rays):
            bearing = ray_idx * (360.0 / num_rays)

            hit = self._cast_elevation_beam(
                staging_x, staging_y, staging_elevation,
                bearing, elevation_angle, max_distance, **kwargs
            )

            if hit:
                distance, hit_x, hit_y = hit
                visibility_points.append((hit_x, hit_y))
            else:
                # No obstruction - extend to max distance
                bearing_rad = np.radians(bearing)
                edge_x = staging_x + max_distance * np.sin(bearing_rad)
                edge_y = staging_y + max_distance * np.cos(bearing_rad)
                visibility_points.append((edge_x, edge_y))

        # Close polygon
        if visibility_points and visibility_points[0] != visibility_points[-1]:
            visibility_points.append(visibility_points[0])

        # Create polygon
        if len(visibility_points) >= 3:
            return Polygon(visibility_points)
        else:
            return Point(staging_x, staging_y).buffer(50)

    def analyze_staging_area(self, staging_point: Dict, staging_id: int,
                             elevation_angle: float = 5.0,
                             max_distance: float = 3000.0,
                             num_rays: int = 360,
                             **kwargs) -> Dict:
        """
        Analyze visibility from a single staging area.

        Args:
            staging_point: Dict with 'coords' and optional 'cluster'
            staging_id: Unique identifier
            elevation_angle: Elevation angle in degrees
            max_distance: Maximum analysis distance
            num_rays: Number of rays to cast
            **kwargs: Additional parameters

        Returns:
            Dictionary with analysis results
        """
        staging_x, staging_y = staging_point['coords']
        cluster = staging_point.get('cluster', None)

        # Get staging elevation
        staging_row, staging_col = self._world_to_pixel(staging_x, staging_y)
        staging_elevation = self._get_elevation_safe(staging_row, staging_col)

        if np.isnan(staging_elevation):
            raise ValueError(f"Invalid staging location: ({staging_x}, {staging_y})")

        logger.info(f"Analyzing staging {staging_id} (cluster {cluster})...")

        # Create visibility polygon
        polygon = self._create_visibility_polygon(
            staging_x, staging_y, staging_elevation,
            elevation_angle, num_rays, max_distance,
            **kwargs
        )

        # Calculate metrics
        area_ha = polygon.area / 10000

        return {
            'staging_id': staging_id,
            'staging_coords': (staging_x, staging_y),
            'staging_elevation': staging_elevation,
            'cluster': cluster,
            'elevation_angle': elevation_angle,
            'visibility_polygon': polygon,
            'visibility_area_ha': area_ha,
            'max_distance': max_distance,
            'num_rays': num_rays
        }

    def analyze_multiple_staging_areas(self, staging_points: List[Dict],
                                       **kwargs) -> List[Dict]:
        """Analyze multiple staging areas."""
        results = []

        for i, staging_point in enumerate(staging_points):
            staging_id = i + 1

            try:
                result = self.analyze_staging_area(
                    staging_point, staging_id, **kwargs
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error analyzing staging {staging_id}: {e}")
                continue

        return results

    def export_to_gpkg(self, results: List[Dict], output_path: str) -> str:
        """
        Export analysis results to GeoPackage.

        Args:
            results: List of analysis results
            output_path: Output file path

        Returns:
            Path to created file
        """
        logger.info(f"Exporting results to: {output_path}")

        # Staging points layer
        staging_data = []
        staging_geoms = []

        for result in results:
            staging_geoms.append(Point(result['staging_coords']))
            staging_data.append({
                'staging_id': result['staging_id'],
                'staging_x': result['staging_coords'][0],
                'staging_y': result['staging_coords'][1],
                'staging_elev': result['staging_elevation'],
                'cluster': result['cluster'],
                'elevation_angle': result['elevation_angle']
            })

        staging_gdf = gpd.GeoDataFrame(
            staging_data, geometry=staging_geoms, crs=self.crs
        )

        # Visibility zones layer
        visibility_data = []
        visibility_geoms = []

        for result in results:
            if result['visibility_polygon'] is not None:
                visibility_geoms.append(result['visibility_polygon'])
                visibility_data.append({
                    'staging_id': result['staging_id'],
                    'cluster': result['cluster'],
                    'staging_x': result['staging_coords'][0],
                    'staging_y': result['staging_coords'][1],
                    'staging_elev': result['staging_elevation'],
                    'elevation_angle': result['elevation_angle'],
                    'visibility_area_ha': result['visibility_area_ha'],
                    'max_distance_m': result['max_distance'],
                    'num_rays': result['num_rays'],
                    'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                })

        visibility_gdf = gpd.GeoDataFrame(
            visibility_data, geometry=visibility_geoms, crs=self.crs
        )

        # Export layers
        staging_gdf.to_file(output_path, layer="staging_points", driver="GPKG")
        visibility_gdf.to_file(output_path, layer="visibility_zones_5deg", driver="GPKG", mode='a')

        logger.info(f"Exported {len(staging_gdf)} staging points")
        logger.info(f"Exported {len(visibility_gdf)} visibility zones")

        return output_path