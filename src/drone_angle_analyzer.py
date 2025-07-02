"""
drone_angle_analyzer.py

Drone angle visibility analysis - calculates angle to drone at specific height above terrain.
Instead of fixed elevation angle, this tool calculates the angle needed to see a drone
at a given distance and height above ground level.
"""

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


class DroneAngleVisibilityAnalyzer:
    """
    Analyzes visibility to drones at specific heights by calculating required viewing angles.
    """

    def __init__(self, dsm_path: str):
        """
        Initialize the drone angle visibility analyzer.

        Args:
            dsm_path: Path to DSM raster file
        """
        self.dsm_path = dsm_path

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

    def _get_min_elevation_in_buffer(self, x: float, y: float, buffer_radius: float = 5.0) -> float:
        """
        Get minimum elevation within a buffer around given coordinates.

        Args:
            x, y: World coordinates
            buffer_radius: Buffer radius in meters

        Returns:
            Minimum elevation in buffer area
        """
        # Convert buffer to pixels
        buffer_pixels = int(np.ceil(buffer_radius / self.pixel_size))

        # Get center pixel
        center_row, center_col = self._world_to_pixel(x, y)

        # Define buffer bounds
        min_row = max(0, center_row - buffer_pixels)
        max_row = min(self.dsm.shape[0], center_row + buffer_pixels + 1)
        min_col = max(0, center_col - buffer_pixels)
        max_col = min(self.dsm.shape[1], center_col + buffer_pixels + 1)

        # Extract buffer area
        buffer_data = self.dsm[min_row:max_row, min_col:max_col]

        # Handle nodata values
        if self.nodata is not None:
            valid_data = buffer_data[buffer_data != self.nodata]
            if len(valid_data) > 0:
                return float(np.min(valid_data))

        # Return minimum if no nodata handling needed
        if buffer_data.size > 0:
            return float(np.min(buffer_data))

        # Fallback to center point
        return self._get_elevation_safe(center_row, center_col)

    def _calculate_angle_to_drone(self, observer_x: float, observer_y: float,
                                  observer_elev: float, drone_x: float, drone_y: float,
                                  drone_height_agl: float, observer_height: float = 1.7) -> Dict:
        """
        Calculate the angle from observer to drone.

        Args:
            observer_x, observer_y: Observer coordinates
            observer_elev: Ground elevation at observer
            drone_x, drone_y: Drone coordinates
            drone_height_agl: Drone height above ground level
            observer_height: Observer eye height above ground

        Returns:
            Dictionary with angle and distance information
        """
        # Get minimum terrain elevation at drone location (within buffer)
        drone_terrain_elev = self._get_min_elevation_in_buffer(drone_x, drone_y, buffer_radius=5.0)

        # Calculate drone absolute elevation
        drone_absolute_elev = drone_terrain_elev + drone_height_agl

        # Observer eye level
        observer_eye_level = observer_elev + observer_height

        # Calculate horizontal distance
        horizontal_distance = np.sqrt((drone_x - observer_x) ** 2 + (drone_y - observer_y) ** 2)

        # Calculate vertical distance
        vertical_distance = drone_absolute_elev - observer_eye_level

        # Calculate angle (in degrees)
        angle_rad = np.arctan2(vertical_distance, horizontal_distance)
        angle_deg = np.degrees(angle_rad)

        return {
            'angle_degrees': angle_deg,
            'horizontal_distance': horizontal_distance,
            'vertical_distance': vertical_distance,
            'drone_terrain_elev': drone_terrain_elev,
            'drone_absolute_elev': drone_absolute_elev,
            'observer_eye_level': observer_eye_level
        }

    def _check_obstruction_depth(self, hit_row: int, hit_col: int,
                                 beam_elevation: float, min_depth_pixels: int = 3) -> bool:
        """
        Check if obstruction has sufficient depth to be considered valid.
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

    def _cast_ray_to_drone(self, start_x: float, start_y: float, start_elevation: float,
                           target_x: float, target_y: float, target_elevation: float,
                           observer_height: float = 1.7,
                           min_depth_pixels: int = 3) -> Dict:
        """
        Cast a ray from observer to target drone position and check for obstructions.

        Returns:
            Dictionary with ray casting results
        """
        # Observer eye level
        eye_level = start_elevation + observer_height

        # Calculate ray direction
        dx = target_x - start_x
        dy = target_y - start_y
        horizontal_distance = np.sqrt(dx ** 2 + dy ** 2)

        if horizontal_distance == 0:
            return {'clear': True, 'obstruction_distance': 0}

        # Vertical change rate
        vertical_rate = (target_elevation - eye_level) / horizontal_distance

        # Step along ray
        step_size = self.pixel_size
        num_steps = int(horizontal_distance / step_size)

        for i in range(1, num_steps):
            # Current position along ray
            t = i * step_size / horizontal_distance
            current_x = start_x + t * dx
            current_y = start_y + t * dy
            current_distance = i * step_size

            # Expected elevation along ray
            ray_elevation = eye_level + current_distance * vertical_rate

            # Get terrain elevation
            row, col = self._world_to_pixel(current_x, current_y)
            terrain_elevation = self._get_elevation_safe(row, col)

            if np.isnan(terrain_elevation):
                continue

            # Check for obstruction
            if terrain_elevation > ray_elevation:
                # Validate obstruction depth
                if self._check_obstruction_depth(row, col, ray_elevation, min_depth_pixels):
                    return {
                        'clear': False,
                        'obstruction_distance': current_distance,
                        'obstruction_coords': (current_x, current_y),
                        'obstruction_elevation': terrain_elevation
                    }

        return {'clear': True, 'obstruction_distance': horizontal_distance}

    def _create_drone_visibility_polygon(self, staging_x: float, staging_y: float,
                                         staging_elevation: float,
                                         drone_distance: float,
                                         drone_height_agl: float,
                                         num_rays: int = 360,
                                         observer_height: float = 1.7,
                                         **kwargs) -> Tuple[Polygon, Dict]:
        """
        Create visibility polygon for drone at specified distance and height.

        Args:
            staging_x, staging_y: Observer coordinates
            staging_elevation: Ground elevation at observer
            drone_distance: Distance to place drone
            drone_height_agl: Drone height above ground
            num_rays: Number of rays to cast
            observer_height: Observer eye height

        Returns:
            Tuple of (visibility polygon, analysis metadata)
        """
        visibility_points = []
        ray_results = []
        angles_used = []

        for ray_idx in range(num_rays):
            # Calculate bearing
            bearing_deg = ray_idx * (360.0 / num_rays)
            bearing_rad = np.radians(bearing_deg)

            # Calculate drone position at specified distance
            drone_x = staging_x + drone_distance * np.sin(bearing_rad)
            drone_y = staging_y + drone_distance * np.cos(bearing_rad)

            # Get drone terrain elevation and calculate angle
            angle_info = self._calculate_angle_to_drone(
                staging_x, staging_y, staging_elevation,
                drone_x, drone_y, drone_height_agl, observer_height
            )

            angles_used.append(angle_info['angle_degrees'])

            # Cast ray to drone
            ray_result = self._cast_ray_to_drone(
                staging_x, staging_y, staging_elevation,
                drone_x, drone_y, angle_info['drone_absolute_elev'],
                observer_height, **kwargs
            )

            ray_results.append({
                'bearing': bearing_deg,
                'angle_to_drone': angle_info['angle_degrees'],
                'clear_to_drone': ray_result['clear'],
                'obstruction_distance': ray_result['obstruction_distance']
            })

            # Add visibility point
            if ray_result['clear']:
                # Can see all the way to drone distance
                visibility_points.append((drone_x, drone_y))
            else:
                # Blocked before reaching drone
                if 'obstruction_coords' in ray_result:
                    visibility_points.append(ray_result['obstruction_coords'])
                else:
                    # Shouldn't happen, but fallback
                    visibility_points.append((staging_x, staging_y))

        # Close polygon
        if visibility_points and visibility_points[0] != visibility_points[-1]:
            visibility_points.append(visibility_points[0])

        # Create polygon
        if len(visibility_points) >= 3:
            polygon = Polygon(visibility_points)
        else:
            polygon = Point(staging_x, staging_y).buffer(50)

        # Calculate statistics
        metadata = {
            'min_angle': np.min(angles_used),
            'max_angle': np.max(angles_used),
            'mean_angle': np.mean(angles_used),
            'std_angle': np.std(angles_used),
            'percent_visible': sum(1 for r in ray_results if r['clear_to_drone']) / len(ray_results) * 100,
            'ray_results': ray_results
        }

        return polygon, metadata

    def analyze_staging_area(self, staging_point: Dict, staging_id: int,
                             drone_distance: float = 2000.0,
                             drone_height_agl: float = 120.0,
                             num_rays: int = 360,
                             observer_height: float = 1.7,
                             **kwargs) -> Dict:
        """
        Analyze visibility to drone from a single staging area.

        Args:
            staging_point: Dict with 'coords' and optional 'cluster'
            staging_id: Unique identifier
            drone_distance: Distance to place drone (meters)
            drone_height_agl: Drone height above ground level
            num_rays: Number of rays to cast
            observer_height: Observer eye height

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

        logger.info(f"Analyzing staging {staging_id} for drone at {drone_distance}m, {drone_height_agl}m AGL...")

        # Create visibility polygon
        polygon, metadata = self._create_drone_visibility_polygon(
            staging_x, staging_y, staging_elevation,
            drone_distance, drone_height_agl, num_rays,
            observer_height, **kwargs
        )

        # Calculate area
        area_ha = polygon.area / 10000

        return {
            'staging_id': staging_id,
            'staging_coords': (staging_x, staging_y),
            'staging_elevation': staging_elevation,
            'cluster': cluster,
            'drone_distance': drone_distance,
            'drone_height_agl': drone_height_agl,
            'visibility_polygon': polygon,
            'visibility_area_ha': area_ha,
            'mean_angle_to_drone': metadata['mean_angle'],
            'min_angle_to_drone': metadata['min_angle'],
            'max_angle_to_drone': metadata['max_angle'],
            'percent_rays_clear': metadata['percent_visible'],
            'num_rays': num_rays,
            'metadata': metadata
        }

    def analyze_multiple_staging_areas(self, staging_points: List[Dict],
                                       **kwargs) -> List[Dict]:
        """Analyze multiple staging areas for drone visibility."""
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
        Export drone visibility analysis results to GeoPackage.

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
                'drone_distance': result['drone_distance'],
                'drone_height_agl': result['drone_height_agl'],
                'mean_angle': result['mean_angle_to_drone'],
                'percent_clear': result['percent_rays_clear']
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
                    'drone_distance_m': result['drone_distance'],
                    'drone_height_agl_m': result['drone_height_agl'],
                    'visibility_area_ha': result['visibility_area_ha'],
                    'mean_angle_deg': result['mean_angle_to_drone'],
                    'min_angle_deg': result['min_angle_to_drone'],
                    'max_angle_deg': result['max_angle_to_drone'],
                    'percent_rays_clear': result['percent_rays_clear'],
                    'num_rays': result['num_rays'],
                    'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                })

        visibility_gdf = gpd.GeoDataFrame(
            visibility_data, geometry=visibility_geoms, crs=self.crs
        )

        # Export layers
        staging_gdf.to_file(output_path, layer="staging_points", driver="GPKG")
        visibility_gdf.to_file(output_path, layer="drone_visibility_zones", driver="GPKG", mode='a')

        logger.info(f"Exported {len(staging_gdf)} staging points")
        logger.info(f"Exported {len(visibility_gdf)} drone visibility zones")

        return output_path