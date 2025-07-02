import os.path
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
from rasterio.crs import CRS
import multiprocessing as mp
from functools import partial
import time
import sys
from scipy.ndimage import minimum_filter

warnings.filterwarnings('ignore')


class ParallelWindowedMobilityVisibilityAnalyzer:
    def __init__(self, dsm_path: str, drone_height_agl: float = 120.0, verbose: bool = True,
                 max_analysis_distance: int = 3000, buffer_extra: int = 500, n_cores: int = 8):
        """
        Initialize parallel analyzer that loads DSM data on-demand in windows

        Args:
            drone_height_agl: Drone height above ground level (default 120m)
            max_analysis_distance: Maximum ray casting distance (meters)
            buffer_extra: Extra buffer around analysis area for edge effects (meters)
            n_cores: Number of CPU cores to use for parallel processing
        """
        self.dsm_path = dsm_path
        self.drone_height_agl = drone_height_agl
        self.verbose = verbose
        self.max_analysis_distance = max_analysis_distance
        self.buffer_extra = buffer_extra
        self.n_cores = n_cores

        if self.verbose:
            print(f"🔍 Initializing parallel windowed DSM reader: {dsm_path}")
            print(f"🚁 Drone height AGL: {drone_height_agl}m")
            print(f"🚀 Using {n_cores} CPU cores for parallel processing")

        # Get DSM metadata without loading the entire raster
        with rasterio.open(dsm_path) as src:
            self.transform = src.transform
            self.crs = CRS.from_epsg(3157)  # Force CRS
            self.nodata = src.nodata
            self.pixel_size = abs(self.transform.a)
            self.dsm_shape = src.shape
            self.dsm_bounds = src.bounds

        if self.verbose:
            print(f"📊 DSM metadata:")
            print(f"   Shape: {self.dsm_shape} ({self.dsm_shape[0] * self.dsm_shape[1] / 1e6:.1f} million pixels)")
            print(f"   Resolution: {self.pixel_size:.2f}m")
            print(f"   CRS: {self.crs}")
            print(f"   Bounds: {self.dsm_bounds}")
            print(f"🧠 Memory-efficient: Loading data on-demand in windows per core")

    @staticmethod
    def process_single_staging_point(args):
        """
        Static method to process a single staging point (for multiprocessing)
        This function runs in a separate process
        """
        (staging_data, dsm_path, dsm_metadata, analysis_params, output_folder) = args

        original_idx, x, y = staging_data
        staging_id = original_idx + 1

        try:
            # Create a local analyzer instance in this process
            local_analyzer = SingleStagingAnalyzer(dsm_path, dsm_metadata, analysis_params)

            # Process this staging point
            result = local_analyzer.analyze_staging_point(x, y, staging_id, analysis_params)

            # Save individual result
            if output_folder and result:
                local_analyzer.export_individual_result(result, output_folder)

            return result

        except Exception as e:
            print(f"❌ Error in process for staging ID {staging_id}: {e}")
            return None

    def analyze_multiple_staging_areas_parallel(self, staging_points, output_folder=None, **kwargs):
        """Analyze multiple staging areas using parallel processing"""

        # Pre-check which points are within DSM bounds
        dsm_left, dsm_bottom, dsm_right, dsm_top = self.dsm_bounds
        valid_points = []
        skipped_count = 0

        for i, (x, y) in enumerate(staging_points):
            if dsm_left <= x <= dsm_right and dsm_bottom <= y <= dsm_top:
                valid_points.append((i, x, y))
            else:
                skipped_count += 1
                if self.verbose:
                    print(f"⚠️  Skipping staging {i + 1}: ({x:.1f}, {y:.1f}) - outside DSM bounds")

        if self.verbose:
            print(
                f"📊 Processing {len(valid_points)}/{len(staging_points)} staging points (skipped {skipped_count} outside DSM)")
            print(f"🚀 Using {self.n_cores} parallel processes")
            if output_folder:
                print(f"💾 Individual results will be saved to: {output_folder}")

        # Prepare data for parallel processing
        dsm_metadata = {
            'transform': self.transform,
            'crs': self.crs,
            'nodata': self.nodata,
            'pixel_size': self.pixel_size,
            'dsm_shape': self.dsm_shape,
            'dsm_bounds': self.dsm_bounds,
            'max_analysis_distance': self.max_analysis_distance,
            'buffer_extra': self.buffer_extra,
            'drone_height_agl': self.drone_height_agl
        }

        # Add drone_height_agl to kwargs if not present
        if 'drone_height_agl' not in kwargs:
            kwargs['drone_height_agl'] = self.drone_height_agl

        # Create argument tuples for each staging point
        process_args = [
            (staging_data, self.dsm_path, dsm_metadata, kwargs, output_folder)
            for staging_data in valid_points
        ]

        # Start timing
        start_time = time.time()

        # Create process pool and run parallel analysis
        if self.verbose:
            print(f"🚀 Starting parallel analysis with {self.n_cores} processes...")

        with mp.Pool(processes=self.n_cores) as pool:
            # Use map to distribute work across processes
            if self.verbose:
                # For verbose mode, use imap to show progress
                results_raw = []
                for i, result in enumerate(pool.imap(self.process_single_staging_point, process_args)):
                    if result:
                        results_raw.append(result)
                        staging_id = result['staging_id']
                        print(f"✅ Completed staging {staging_id} ({i + 1}/{len(process_args)}) - "
                              f"Area: {result['total_area']:.1f} ha, "
                              f"Drone visibility: {result['drone_visibility_area']:.1f} ha")
                    else:
                        print(f"❌ Failed staging point {i + 1}/{len(process_args)}")
            else:
                # Silent mode - just get all results
                results_raw = pool.map(self.process_single_staging_point, process_args)

        # Filter out None results (failed staging points)
        results = [r for r in results_raw if r is not None]

        # Calculate timing
        elapsed_time = time.time() - start_time

        if self.verbose:
            print(f"\n📊 PARALLEL ANALYSIS COMPLETE!")
            print(f"⏱️  Total time: {elapsed_time:.1f} seconds ({elapsed_time / 60:.1f} minutes)")
            print(f"🚀 Average time per staging: {elapsed_time / len(valid_points):.1f} seconds")
            print(f"💪 Speedup vs sequential: ~{self.n_cores:.1f}x faster")
            print(f"✅ Successfully analyzed: {len(results)}")
            print(f"⚠️  Skipped (outside bounds): {skipped_count}")
            print(f"❌ Failed (errors): {len(valid_points) - len(results)}")
            if output_folder:
                print(f"💾 Individual files saved in: {output_folder}")

        return results

    @classmethod
    def load_staging_points_from_gpkg(cls, gpkg_path, layer_name=None, target_crs=None):
        """Load staging points from GeoPackage and reproject if needed"""
        print(f"📍 Loading staging points from: {gpkg_path}")

        gdf = gpd.read_file(gpkg_path, layer=layer_name) if layer_name else gpd.read_file(gpkg_path)

        print(f"   Loaded {len(gdf)} staging points")
        print(f"   Original CRS: {gdf.crs}")
        print(f"   Original bounds: {gdf.total_bounds}")

        staging_points = [(geom.x, geom.y) for geom in gdf.geometry if geom.geom_type == "Point"]

        print(f"   Extracted {len(staging_points)} coordinates")
        if staging_points:
            xs, ys = zip(*staging_points)
            print(f"   Final range: X=[{min(xs):.0f}, {max(xs):.0f}]  Y=[{min(ys):.0f}, {max(ys):.0f}]")

        return staging_points

    def combine_individual_results(self, results_dir: str, final_output_path: str):
        """Combine all individual staging results into final GPKG"""
        import glob

        if self.verbose:
            print(f"\n🔗 Combining individual results from: {results_dir}")

        # Find all individual staging files
        pattern = os.path.join(results_dir, "staging_*_visibility.gpkg")
        individual_files = sorted(glob.glob(pattern))

        if not individual_files:
            print(f"❌ No individual staging files found in {results_dir}")
            return None

        # Load and combine all results
        all_staging_points = []
        all_individual_zones = []
        all_combined_zones = []
        all_sample_points = []
        all_drone_zones = []

        for file_path in individual_files:
            try:
                # Read each layer from individual file
                staging_gdf = gpd.read_file(file_path, layer="staging_points")
                individual_gdf = gpd.read_file(file_path, layer="individual_visibility_zones")
                combined_gdf = gpd.read_file(file_path, layer="combined_visibility_zones")
                sample_gdf = gpd.read_file(file_path, layer="sample_points")

                # Try to read drone visibility layer if it exists
                try:
                    drone_gdf = gpd.read_file(file_path, layer="drone_visibility_zones")
                    all_drone_zones.append(drone_gdf)
                except:
                    pass

                all_staging_points.append(staging_gdf)
                all_individual_zones.append(individual_gdf)
                all_combined_zones.append(combined_gdf)
                all_sample_points.append(sample_gdf)

                if self.verbose:
                    staging_id = staging_gdf.iloc[0]['staging_id']
                    print(f"    ✅ Loaded staging {staging_id}")

            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️  Error loading {file_path}: {e}")
                continue

        # Concatenate all layers
        final_staging_gdf = gpd.pd.concat(all_staging_points, ignore_index=True)
        final_individual_gdf = gpd.pd.concat(all_individual_zones, ignore_index=True)
        final_combined_gdf = gpd.pd.concat(all_combined_zones, ignore_index=True)
        final_sample_gdf = gpd.pd.concat(all_sample_points, ignore_index=True)

        # Export final combined file
        final_staging_gdf.to_file(final_output_path, layer="staging_points", driver="GPKG")
        final_individual_gdf.to_file(final_output_path, layer="individual_visibility_zones", driver="GPKG", mode='a')
        final_combined_gdf.to_file(final_output_path, layer="combined_visibility_zones", driver="GPKG", mode='a')
        final_sample_gdf.to_file(final_output_path, layer="sample_points", driver="GPKG", mode='a')

        # Add drone visibility layer if available
        if all_drone_zones:
            final_drone_gdf = gpd.pd.concat(all_drone_zones, ignore_index=True)
            final_drone_gdf.to_file(final_output_path, layer="drone_visibility_zones", driver="GPKG", mode='a')
            print(f"    📊 Total drone visibility zones: {len(final_drone_gdf)}")

        if self.verbose:
            print(f"    🎉 Final combined file saved: {final_output_path}")
            print(f"    📊 Total staging points: {len(final_staging_gdf)}")
            print(f"    📊 Total individual zones: {len(final_individual_gdf)}")
            print(f"    📊 Total combined zones: {len(final_combined_gdf)}")
            print(f"    📊 Total sample points: {len(final_sample_gdf)}")

        return final_output_path


class SingleStagingAnalyzer:
    """
    Analyzer for a single staging point - used in each process
    This is a simplified version that runs in worker processes
    """

    def __init__(self, dsm_path, dsm_metadata, analysis_params):
        self.dsm_path = dsm_path
        self.transform = dsm_metadata['transform']
        self.crs = dsm_metadata['crs']
        self.nodata = dsm_metadata['nodata']
        self.pixel_size = dsm_metadata['pixel_size']
        self.dsm_shape = dsm_metadata['dsm_shape']
        self.dsm_bounds = dsm_metadata['dsm_bounds']
        self.max_analysis_distance = dsm_metadata['max_analysis_distance']
        self.buffer_extra = dsm_metadata['buffer_extra']
        self.drone_height_agl = dsm_metadata.get('drone_height_agl', 120.0)

    def _get_analysis_window(self, center_x, center_y, mobility_buffer):
        """Calculate the minimum window needed for analysis around a staging point"""
        dsm_left, dsm_bottom, dsm_right, dsm_top = self.dsm_bounds

        # Total analysis radius = mobility buffer + max ray distance + extra buffer
        total_radius = mobility_buffer + self.max_analysis_distance + self.buffer_extra

        # Calculate desired window bounds in world coordinates
        window_left = max(dsm_left, center_x - total_radius)
        window_right = min(dsm_right, center_x + total_radius)
        window_bottom = max(dsm_bottom, center_y - total_radius)
        window_top = min(dsm_top, center_y + total_radius)

        # Convert to pixel coordinates
        top_row, left_col = rowcol(self.transform, window_left, window_top)
        bottom_row, right_col = rowcol(self.transform, window_right, window_bottom)

        # Ensure proper ordering and bounds
        min_row = max(0, min(top_row, bottom_row))
        max_row = min(self.dsm_shape[0], max(top_row, bottom_row))
        min_col = max(0, min(left_col, right_col))
        max_col = min(self.dsm_shape[1], max(left_col, right_col))

        # Calculate window dimensions
        width = max_col - min_col
        height = max_row - min_row

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid window dimensions: {width}×{height} pixels")

        return Window(min_col, min_row, width, height)

    def _load_dsm_window(self, window):
        """Load DSM data for the specified window"""
        with rasterio.open(self.dsm_path) as src:
            # Read the window
            data = src.read(1, window=window).astype(np.float32)

            # Get the transform for this window
            window_transform = rasterio.windows.transform(window, src.transform)

            # Handle nodata - use simple fill to avoid memory issues
            if self.nodata is not None and np.any(data == self.nodata):
                nodata_mask = data == self.nodata
                if np.any(~nodata_mask):
                    # Fill with mean of valid data in this window, but minimum 1000m
                    valid_mean = np.mean(data[~nodata_mask])
                    fill_value = max(1000.0, valid_mean)
                    data[nodata_mask] = fill_value
                else:
                    # If entire window is nodata, fill with 1000m elevation
                    data.fill(1000.0)

        return data, window_transform

    def _world_to_pixel_windowed(self, x, y, window_transform):
        """Convert world coordinates to pixel coordinates within a window"""
        row, col = rowcol(window_transform, x, y)
        return int(row), int(col)

    def _get_elevation_safe_windowed(self, row, col, dsm_window):
        """Get elevation from windowed DSM data"""
        if 0 <= row < dsm_window.shape[0] and 0 <= col < dsm_window.shape[1]:
            return float(dsm_window[row, col])
        return np.nan

    def _find_local_minima(self, dsm_window, window_transform, center_x, center_y, buffer_radius,
                           minima_radius=3.0):
        """Find local minima in DSM within buffer area"""
        from scipy import ndimage
        from scipy.ndimage import minimum_filter

        # Calculate buffer bounds in pixel coordinates
        center_row, center_col = self._world_to_pixel_windowed(center_x, center_y, window_transform)
        buffer_pixels = int(buffer_radius / self.pixel_size)

        # Define search area within buffer
        min_row = max(0, center_row - buffer_pixels)
        max_row = min(dsm_window.shape[0], center_row + buffer_pixels + 1)
        min_col = max(0, center_col - buffer_pixels)
        max_col = min(dsm_window.shape[1], center_col + buffer_pixels + 1)

        # Extract buffer area
        buffer_dsm = dsm_window[min_row:max_row, min_col:max_col]

        # Calculate minimum filter size based on minima radius
        filter_size = int(2 * minima_radius / self.pixel_size) + 1
        if filter_size < 3:
            filter_size = 3

        # Apply minimum filter to find local minima
        local_min = minimum_filter(buffer_dsm, size=filter_size, mode='constant', cval=np.inf)

        # Find where original values equal local minimum (these are local minima)
        is_minima = (buffer_dsm == local_min) & (buffer_dsm < np.inf)

        # Get coordinates of local minima
        minima_rows, minima_cols = np.where(is_minima)

        # Convert back to full window coordinates and then to world coordinates
        minima_points = []
        for i in range(len(minima_rows)):
            row = minima_rows[i] + min_row
            col = minima_cols[i] + min_col

            # Convert to world coordinates
            x, y = xy(window_transform, row, col)

            # Check if within circular buffer
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if dist <= buffer_radius and dist > 0:  # Exclude center point
                elev = float(buffer_dsm[minima_rows[i], minima_cols[i]])
                bearing = (90 - np.degrees(np.arctan2(y - center_y, x - center_x))) % 360

                minima_points.append({
                    'coords': (x, y),
                    'elevation': elev,
                    'distance_from_center': dist,
                    'bearing_from_center': bearing
                })

        return minima_points

    def _generate_sample_points(self, center_x, center_y, buffer_radius, dsm_window, window_transform,
                                num_points=20, sampling='strategic'):
        """Generate sample points at local minima within buffer"""
        points = []

        # Center point
        center_row, center_col = self._world_to_pixel_windowed(center_x, center_y, window_transform)
        center_elev = self._get_elevation_safe_windowed(center_row, center_col, dsm_window)
        points.append({
            'point_id': 0,
            'coords': (center_x, center_y),
            'distance_from_center': 0.0,
            'bearing_from_center': 0.0,
            'point_type': 'center',
            'elevation': center_elev if not np.isnan(center_elev) else 1000.0,
        })

        if num_points == 1:
            return points

        # Find local minima within buffer
        minima_points = self._find_local_minima(dsm_window, window_transform, center_x, center_y,
                                                buffer_radius, minima_radius=3.0)

        if not minima_points:
            print(f"Warning: No local minima found within buffer, using center point only")
            return points[:1]

        # Sort by elevation (lowest first) to prioritize deeper minima
        minima_points.sort(key=lambda p: p['elevation'])

        # Select up to num_points-1 minima (excluding center)
        selected_minima = minima_points[:num_points - 1]

        # Add selected minima as sample points
        pid = 1
        for minima in selected_minima:
            points.append({
                'point_id': pid,
                'coords': minima['coords'],
                'distance_from_center': minima['distance_from_center'],
                'bearing_from_center': minima['bearing_from_center'],
                'point_type': 'local_minimum',
                'elevation': minima['elevation'],
            })
            pid += 1

        # If we have fewer minima than requested points, add some distributed points
        if len(points) < num_points:
            remaining = num_points - len(points)
            # Sort remaining minima by distance to get spatial distribution
            remaining_minima = minima_points[len(selected_minima):]
            if remaining_minima:
                # Select evenly distributed by bearing
                remaining_minima.sort(key=lambda p: p['bearing_from_center'])
                step = max(1, len(remaining_minima) // remaining)
                for i in range(0, min(len(remaining_minima), remaining * step), step):
                    if len(points) >= num_points:
                        break
                    minima = remaining_minima[i]
                    points.append({
                        'point_id': pid,
                        'coords': minima['coords'],
                        'distance_from_center': minima['distance_from_center'],
                        'bearing_from_center': minima['bearing_from_center'],
                        'point_type': 'local_minimum',
                        'elevation': minima['elevation'],
                    })
                    pid += 1

        return points[:num_points]

    def _find_drone_visibility_distance(self, start_x, start_y, start_elev, bearing_deg,
                                        dsm_window, window_transform, drone_height_agl=120.0,
                                        max_distance=2000, observer_height=1.7, step_back=200):
        """
        Find the maximum distance where a drone is visible along a bearing.

        Args:
            drone_height_agl: Height of drone above ground level (meters)
            step_back: Distance to step back when drone is not visible (meters)
        """
        bearing_rad = np.radians(bearing_deg)
        step = self.pixel_size * 1.5

        # Observer eye level
        observer_eye_level = start_elev + observer_height

        # Start from max distance and work backwards if needed
        current_distance = max_distance
        drone_visible = False
        final_distance = 0

        while current_distance > 0:
            # Drone position
            drone_x = start_x + current_distance * np.sin(bearing_rad)
            drone_y = start_y + current_distance * np.cos(bearing_rad)

            # Get terrain elevation at drone position
            drone_row, drone_col = self._world_to_pixel_windowed(drone_x, drone_y, window_transform)
            drone_terrain_elev = self._get_elevation_safe_windowed(drone_row, drone_col, dsm_window)

            if np.isnan(drone_terrain_elev):
                # If outside window, step back
                current_distance -= step_back
                continue

            # Drone elevation (terrain + AGL height)
            drone_elev = drone_terrain_elev + drone_height_agl

            # Calculate angle from observer to drone
            horizontal_dist = current_distance
            vertical_dist = drone_elev - observer_eye_level
            angle_to_drone = np.degrees(np.arctan2(vertical_dist, horizontal_dist))

            # Check line of sight from observer to drone
            num_steps = int(current_distance / step)
            line_blocked = False

            for i in range(1, num_steps):
                d = i * step
                x = start_x + d * np.sin(bearing_rad)
                y = start_y + d * np.cos(bearing_rad)

                # Line of sight elevation at this distance
                los_elev = observer_eye_level + d * np.tan(np.radians(angle_to_drone))

                # Terrain elevation
                row, col = self._world_to_pixel_windowed(x, y, window_transform)
                terrain_elev = self._get_elevation_safe_windowed(row, col, dsm_window)

                if np.isnan(terrain_elev):
                    line_blocked = True
                    break

                # Check if terrain blocks line of sight
                if terrain_elev > los_elev:
                    line_blocked = True
                    break

            if not line_blocked:
                # Drone is visible at this distance
                drone_visible = True
                final_distance = current_distance
                break
            else:
                # Drone not visible, step back and try again
                current_distance -= step_back

        return {
            'visible': drone_visible,
            'distance': final_distance,
            'drone_coords': (drone_x, drone_y) if drone_visible else None,
            'angle_to_drone': angle_to_drone if drone_visible else None
        }

    def _cast_visibility_ray(self, start_x, start_y, start_elev, bearing_deg, elev_angle_deg,
                             dsm_window, window_transform, max_distance=3000, observer_height=1.7):
        """Cast UPWARD visibility ray for line-of-sight analysis (original method for comparison)"""
        bearing_rad = np.radians(bearing_deg)
        elev_rad = np.radians(elev_angle_deg)
        step = self.pixel_size * 1.5
        num_steps = int(max_distance / step)

        # Observer eye level = ground elevation + human height
        observer_eye_level = start_elev + observer_height

        for i in range(1, num_steps + 1):
            d = i * step
            x = start_x + d * np.sin(bearing_rad)
            y = start_y + d * np.cos(bearing_rad)

            # UPWARD beam at specified elevation angle
            beam_elev = observer_eye_level + d * np.tan(elev_rad)

            row, col = self._world_to_pixel_windowed(x, y, window_transform)
            terrain_elev = self._get_elevation_safe_windowed(row, col, dsm_window)

            if np.isnan(terrain_elev):
                return {
                    'hit': True,
                    'hit_distance': d,
                    'hit_coords': (x, y),
                    'hit_elevation': None,
                    'beam_elevation': beam_elev,
                    'obstruction_height': 0,
                    'hit_reason': 'window_edge'
                }

            # Check if terrain blocks the UPWARD beam
            if terrain_elev > beam_elev:
                return {
                    'hit': True,
                    'hit_distance': d,
                    'hit_coords': (x, y),
                    'hit_elevation': terrain_elev,
                    'beam_elevation': beam_elev,
                    'obstruction_height': terrain_elev - beam_elev,
                    'hit_reason': 'terrain_blocks_line_of_sight'
                }

        return {
            'hit': False,
            'hit_distance': max_distance,
            'hit_coords': None,
            'hit_elevation': None,
            'beam_elevation': observer_eye_level + max_distance * np.tan(elev_rad),
            'obstruction_height': 0,
            'hit_reason': 'max_distance'
        }

    def analyze_staging_point(self, staging_x, staging_y, staging_id, analysis_params):
        """Analyze one staging area using windowed DSM loading"""

        buffer = analysis_params['buffer']
        num_points = analysis_params['num_points']
        num_rays = analysis_params['num_rays']
        elev_angle = analysis_params['elev_angle']
        max_distance = analysis_params['max_distance']
        sampling = analysis_params['sampling']
        drone_height_agl = analysis_params.get('drone_height_agl', 120.0)

        # Calculate and load the required DSM window
        window = self._get_analysis_window(staging_x, staging_y, buffer)
        dsm_window, window_transform = self._load_dsm_window(window)

        # Get elevation at staging center
        row, col = self._world_to_pixel_windowed(staging_x, staging_y, window_transform)
        staging_elev = self._get_elevation_safe_windowed(row, col, dsm_window)
        if np.isnan(staging_elev):
            raise ValueError(f"Invalid elevation at staging ({staging_x}, {staging_y})")

        # Generate sample points using windowed data
        sample_points = self._generate_sample_points(staging_x, staging_y, buffer, dsm_window, window_transform,
                                                     num_points, sampling)
        bearings = np.linspace(0, 360, num_rays, endpoint=False)

        all_ray_data = []
        max_dist_by_bearing = {}
        drone_visibility_by_bearing = {}
        individual_sample_polygons = []

        # Cast rays for each bearing and sample point
        for b_idx, bearing in enumerate(bearings):
            ray_id = b_idx + 1
            best = {'distance': 0, 'point_id': None, 'ray_data': None}
            best_drone = {'distance': 0, 'point_id': None, 'drone_data': None}

            for sp in sample_points:
                pid = sp['point_id']
                sx, sy = sp['coords']
                srow, scol = self._world_to_pixel_windowed(sx, sy, window_transform)
                selev = self._get_elevation_safe_windowed(srow, scol, dsm_window)
                if np.isnan(selev):
                    continue

                # Original visibility ray (for comparison)
                ray = self._cast_visibility_ray(sx, sy, selev, bearing, elev_angle, dsm_window, window_transform,
                                                max_distance)

                # Find drone visibility distance
                drone_result = self._find_drone_visibility_distance(sx, sy, selev, bearing, dsm_window,
                                                                    window_transform, drone_height_agl,
                                                                    max_distance)

                ray_data = {
                    'staging_id': staging_id,
                    'point_id': pid,
                    'ray_id': ray_id,
                    'bearing_deg': bearing,
                    'elevation_angle': elev_angle,
                    'sample_coords': (sx, sy),
                    'sample_elevation': selev,
                    'point_type': sp['point_type'],
                    'distance_from_staging_center': sp['distance_from_center'],
                    'visibility_distance': ray['hit_distance'],
                    'drone_visibility_distance': drone_result['distance'],
                    'drone_visible': drone_result['visible'],
                    'angle_to_drone': drone_result['angle_to_drone'],
                    'hit_terrain': ray['hit'],
                    'hit_coords': ray['hit_coords'],
                    'hit_elevation': ray['hit_elevation'],
                    'beam_elevation': ray['beam_elevation'],
                    'obstruction_height': ray['obstruction_height']
                }
                all_ray_data.append(ray_data)

                if ray['hit_distance'] > best['distance']:
                    best = {'distance': ray['hit_distance'], 'point_id': pid, 'ray_data': ray_data}

                if drone_result['distance'] > best_drone['distance']:
                    best_drone = {'distance': drone_result['distance'], 'point_id': pid, 'drone_data': drone_result}

            max_dist_by_bearing[bearing] = best
            drone_visibility_by_bearing[bearing] = best_drone

        # Build visibility polygons (original method)
        visibility_points = []
        for bearing in bearings:
            dist = max_dist_by_bearing[bearing]['distance']
            x = staging_x + dist * np.sin(np.radians(bearing))
            y = staging_y + dist * np.cos(np.radians(bearing))
            visibility_points.append((x, y))

        if visibility_points and visibility_points[0] != visibility_points[-1]:
            visibility_points.append(visibility_points[0])

        visibility_polygon = Polygon(visibility_points) if len(visibility_points) >= 3 else Point(staging_x,
                                                                                                  staging_y).buffer(50)

        # Build drone visibility polygon
        drone_visibility_points = []
        for bearing in bearings:
            dist = drone_visibility_by_bearing[bearing]['distance']
            if dist > 0:
                x = staging_x + dist * np.sin(np.radians(bearing))
                y = staging_y + dist * np.cos(np.radians(bearing))
                drone_visibility_points.append((x, y))
            else:
                # If drone not visible in this direction, use staging point
                drone_visibility_points.append((staging_x, staging_y))

        if drone_visibility_points and drone_visibility_points[0] != drone_visibility_points[-1]:
            drone_visibility_points.append(drone_visibility_points[0])

        drone_visibility_polygon = Polygon(drone_visibility_points) if len(drone_visibility_points) >= 3 else Point(
            staging_x, staging_y).buffer(10)

        # Center-only polygon for comparison
        center_rays = [r for r in all_ray_data if r['point_id'] == 0]
        center_dist = {r['bearing_deg']: r['visibility_distance'] for r in center_rays}
        center_points = []
        for b in bearings:
            dist = center_dist.get(b, max_distance)
            x = staging_x + dist * np.sin(np.radians(b))
            y = staging_y + dist * np.cos(np.radians(b))
            center_points.append((x, y))

        if center_points and center_points[0] != center_points[-1]:
            center_points.append(center_points[0])

        center_polygon = Polygon(center_points) if len(center_points) >= 3 else Point(staging_x, staging_y).buffer(50)

        # Calculate metrics
        total_area = visibility_polygon.area / 10000  # hectares
        center_area = center_polygon.area / 10000
        drone_visibility_area = drone_visibility_polygon.area / 10000
        improvement = ((total_area - center_area) / center_area * 100) if center_area > 0 else 0

        # Point usage stats
        point_usage = {}
        drone_point_usage = {}
        for b in bearings:
            pid = max_dist_by_bearing[b]['point_id']
            if pid is not None:
                point_usage[pid] = point_usage.get(pid, 0) + 1

            drone_pid = drone_visibility_by_bearing[b]['point_id']
            if drone_pid is not None:
                drone_point_usage[drone_pid] = drone_point_usage.get(drone_pid, 0) + 1

        # Individual sample polygons
        for sp in sample_points:
            pid = sp['point_id']
            sx, sy = sp['coords']
            srow, scol = self._world_to_pixel_windowed(sx, sy, window_transform)
            selev = self._get_elevation_safe_windowed(srow, scol, dsm_window)
            if np.isnan(selev):
                continue

            poly_pts = []
            drone_poly_pts = []
            for b in bearings:
                rays = [r for r in all_ray_data if r['point_id'] == pid and r['bearing_deg'] == b]
                if rays:
                    dist = rays[0]['visibility_distance']
                    drone_dist = rays[0]['drone_visibility_distance']

                    # Original visibility
                    x = sx + dist * np.sin(np.radians(b))
                    y = sy + dist * np.cos(np.radians(b))
                    poly_pts.append((x, y))

                    # Drone visibility
                    if drone_dist > 0:
                        dx = sx + drone_dist * np.sin(np.radians(b))
                        dy = sy + drone_dist * np.cos(np.radians(b))
                        drone_poly_pts.append((dx, dy))
                    else:
                        drone_poly_pts.append((sx, sy))

            if poly_pts and poly_pts[0] != poly_pts[-1]:
                poly_pts.append(poly_pts[0])

            if drone_poly_pts and drone_poly_pts[0] != drone_poly_pts[-1]:
                drone_poly_pts.append(drone_poly_pts[0])

            if len(poly_pts) >= 3:
                sample_poly = Polygon(poly_pts)
                sample_area = sample_poly.area / 10000

                drone_sample_poly = Polygon(drone_poly_pts) if len(drone_poly_pts) >= 3 else Point(sx, sy).buffer(10)
                drone_sample_area = drone_sample_poly.area / 10000

                individual_sample_polygons.append({
                    'staging_id': staging_id,
                    'point_id': pid,
                    'staging_point_id': f"{staging_id}_{pid}",
                    'sample_coords': (sx, sy),
                    'sample_elevation': selev,
                    'point_type': sp['point_type'],
                    'visibility_polygon': sample_poly,
                    'visibility_area_ha': sample_area,
                    'drone_visibility_polygon': drone_sample_poly,
                    'drone_visibility_area_ha': drone_sample_area,
                    'distance_from_center': sp['distance_from_center'],
                    'bearing_from_center': sp['bearing_from_center'],
                })

        # Clean up window data from memory
        del dsm_window

        return {
            'staging_id': staging_id,
            'staging_coords': (staging_x, staging_y),
            'staging_elevation': staging_elev,
            'mobility_buffer': buffer,
            'elevation_angle': elev_angle,
            'drone_height_agl': drone_height_agl,
            'num_sample_points': len(sample_points),
            'num_rays': num_rays,
            'total_ray_calculations': len(all_ray_data),
            'visibility_polygon': visibility_polygon,
            'drone_visibility_polygon': drone_visibility_polygon,
            'center_area': center_area,
            'total_area': total_area,
            'drone_visibility_area': drone_visibility_area,
            'improvement_pct': improvement,
            'sample_points': sample_points,
            'individual_sample_polygons': individual_sample_polygons,
            'all_ray_data': all_ray_data,
            'max_distances_by_bearing': max_dist_by_bearing,
            'drone_visibility_by_bearing': drone_visibility_by_bearing,
            'point_usage_stats': point_usage,
            'drone_point_usage_stats': drone_point_usage,
            'sampling_method': sampling
        }

    def export_individual_result(self, result: Dict, output_folder: str):
        """Export single staging result to a GeoPackage with multiple layers"""
        os.makedirs(output_folder, exist_ok=True)
        staging_id = result["staging_id"]
        output_path = os.path.join(output_folder, f"staging_{staging_id:03d}_visibility.gpkg")

        # Staging point
        staging_data = [{
            'staging_id': result['staging_id'],
            'staging_x': result['staging_coords'][0],
            'staging_y': result['staging_coords'][1],
            'staging_elev': result['staging_elevation'],
            'mobility_buffer': result['mobility_buffer'],
            'elevation_angle': result['elevation_angle'],
            'drone_height_agl': result['drone_height_agl'],
            'num_sample_points': result['num_sample_points'],
            'num_rays': result['num_rays'],
            'sampling_method': result['sampling_method'],
            'visibility_area_ha': result['total_area'],
            'drone_visibility_area_ha': result['drone_visibility_area'],
            'area_improvement_pct': result['improvement_pct']
        }]

        staging_gdf = gpd.GeoDataFrame(
            staging_data,
            geometry=[Point(result["staging_coords"])],
            crs=self.crs
        )
        staging_gdf.to_file(output_path, layer="staging_points", driver="GPKG")

        # Combined visibility polygon (original method)
        combined_data = [{
            'staging_id': result['staging_id'],
            'staging_point_id': result['staging_id'],
            'staging_x': result['staging_coords'][0],
            'staging_y': result['staging_coords'][1],
            'mobility_buffer': result['mobility_buffer'],
            'elevation_angle': result['elevation_angle'],
            'visibility_area_ha': result['total_area'],
            'center_area_ha': result['center_area'],
            'area_improvement_pct': result['improvement_pct'],
            'num_sample_points': result['num_sample_points'],
            'sampling_method': result['sampling_method'],
            'analysis_method': 'mobility_enhanced_combined'
        }]

        combined_gdf = gpd.GeoDataFrame(
            combined_data,
            geometry=[result["visibility_polygon"]],
            crs=self.crs
        )
        combined_gdf.to_file(output_path, layer="combined_visibility_zones", driver="GPKG", mode='a')

        # Drone visibility polygon
        drone_data = [{
            'staging_id': result['staging_id'],
            'staging_point_id': result['staging_id'],
            'staging_x': result['staging_coords'][0],
            'staging_y': result['staging_coords'][1],
            'drone_height_agl': result['drone_height_agl'],
            'drone_visibility_area_ha': result['drone_visibility_area'],
            'num_sample_points': result['num_sample_points'],
            'sampling_method': result['sampling_method'],
            'analysis_method': 'drone_visibility'
        }]

        drone_gdf = gpd.GeoDataFrame(
            drone_data,
            geometry=[result["drone_visibility_polygon"]],
            crs=self.crs
        )
        drone_gdf.to_file(output_path, layer="drone_visibility_zones", driver="GPKG", mode='a')

        # Individual polygons
        if result["individual_sample_polygons"]:
            individual_data = []
            individual_geoms = []
            for sample_poly_data in result["individual_sample_polygons"]:
                individual_geoms.append(sample_poly_data["visibility_polygon"])
                individual_data.append({
                    'staging_id': sample_poly_data['staging_id'],
                    'point_id': sample_poly_data['point_id'],
                    'staging_point_id': sample_poly_data['staging_point_id'],
                    'sample_x': sample_poly_data['sample_coords'][0],
                    'sample_y': sample_poly_data['sample_coords'][1],
                    'point_type': sample_poly_data['point_type'],
                    'visibility_area_ha': sample_poly_data['visibility_area_ha'],
                    'drone_visibility_area_ha': sample_poly_data['drone_visibility_area_ha'],
                    'distance_from_center': sample_poly_data['distance_from_center'],
                    'elevation_angle': result['elevation_angle'],
                    'analysis_method': 'individual_sample_point'
                })

            individual_gdf = gpd.GeoDataFrame(
                individual_data,
                geometry=individual_geoms,
                crs=self.crs
            )
            individual_gdf.to_file(output_path, layer="individual_visibility_zones", driver="GPKG", mode='a')

        # Sample points
        sample_data = []
        sample_geoms = []
        for pt in result["sample_points"]:
            sample_geoms.append(Point(pt["coords"]))
            sample_data.append({
                'staging_id': result['staging_id'],
                'point_id': pt['point_id'],
                'point_type': pt['point_type'],
                'distance_from_center': pt['distance_from_center'],
                'bearing_from_center': pt['bearing_from_center'],
                'elevation': pt['elevation'],
            })

        sample_gdf = gpd.GeoDataFrame(
            sample_data,
            geometry=sample_geoms,
            crs=self.crs
        )
        sample_gdf.to_file(output_path, layer="sample_points", driver="GPKG", mode='a')

        return output_path


# Example usage for parallel processing
if __name__ == "__main__":
    # Ensure this runs only in the main process (required for multiprocessing)
    if __name__ == "__main__":
        # Set multiprocessing start method (important for some systems)
        mp.set_start_method('spawn', force=True)

        # Your file paths
        dsm_path = "file:///media/irina/My Book/Petronas/DATA/FullData/DSM_may25.tif"
        staging_gpkg = "file:///home/irina/Desktop/petronas_staging_day2.gpkg"

        # Output paths
        output_folder = "/media/irina/My Book/Petronas/DATA/visibility_petronas_day2min"
        final_output_path = "/media/irina/My Book/Petronas/DATA/petronas_staging_day2min.gpkg"

        # Analysis parameters
        MOBILITY_BUFFER = 150.0
        NUM_SAMPLE_POINTS = 150  # Keep reasonable for parallel processing
        NUM_RAYS = 180  # 2° increments
        ELEVATION_ANGLE = 5.0
        MAX_DISTANCE = 2000
        DRONE_HEIGHT_AGL = 120.0  # Drone height above ground level
        SAMPLING_METHOD = 'strategic'
        N_CORES = 16  # Use 16 CPU cores

        print("=" * 80)
        print("🚀 PARALLEL WINDOWED VISIBILITY ANALYZER WITH DRONE DETECTION")
        print("=" * 80)
        print("Features:")
        print(f"- 🚀 Parallel processing: {N_CORES} CPU cores")
        print(f"- 🚁 Drone visibility analysis: {DRONE_HEIGHT_AGL}m AGL")
        print(f"- 💾 Loads DSM data on-demand in small windows per core")
        print(f"- 🎯 Analysis radius: {MAX_DISTANCE}m per staging point")
        print(f"- 📊 Sample points: {NUM_SAMPLE_POINTS} in {MOBILITY_BUFFER}m buffer")
        print(f"- 🎯 Ray accuracy: {NUM_RAYS} rays ({360 / NUM_RAYS:.1f}° increments)")
        print(f"- 💾 Individual saving: Each staging point saved immediately")
        print(f"- 🔧 Nodata fill: 1000m elevation for missing data")
        print(f"- ⚡ Expected speedup: ~{N_CORES}x faster than sequential")

        # Initialize parallel analyzer
        analyzer = ParallelWindowedMobilityVisibilityAnalyzer(
            dsm_path,
            drone_height_agl=DRONE_HEIGHT_AGL,
            verbose=True,
            max_analysis_distance=MAX_DISTANCE,
            buffer_extra=200,
            n_cores=N_CORES
        )

        # Load staging points
        staging_points = ParallelWindowedMobilityVisibilityAnalyzer.load_staging_points_from_gpkg(staging_gpkg)

        # Run parallel analysis
        print(f"\n🚀 Starting parallel analysis...")
        start_total_time = time.time()

        results = analyzer.analyze_multiple_staging_areas_parallel(
            staging_points,
            output_folder=output_folder,
            buffer=MOBILITY_BUFFER,
            num_points=NUM_SAMPLE_POINTS,
            num_rays=NUM_RAYS,
            elev_angle=ELEVATION_ANGLE,
            max_distance=MAX_DISTANCE,
            drone_height_agl=DRONE_HEIGHT_AGL,
            sampling=SAMPLING_METHOD
        )

        total_time = time.time() - start_total_time

        # Combine all individual results into final file
        if results:
            print(f"\n📁 Creating final combined file...")
            final_file = analyzer.combine_individual_results(output_folder, final_output_path)

            print(f"\n🎉 PARALLEL ANALYSIS COMPLETE!")
            print(f"⏱️  Total processing time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
            print(f"✅ Successfully processed: {len(results)} staging areas")
            print(f"🚀 Average time per staging: {total_time / len(results):.1f} seconds")
            print(f"💾 Individual files: {output_folder}/staging_*_visibility.gpkg")
            print(f"📁 Final combined file: {final_file}")

            # Show summary statistics
            total_area = sum(r['total_area'] for r in results)
            total_drone_area = sum(r['drone_visibility_area'] for r in results)
            avg_improvement = sum(r['improvement_pct'] for r in results) / len(results)
            total_calculations = sum(r['total_ray_calculations'] for r in results)

            print(f"\n📊 SUMMARY STATISTICS:")
            print(f"   Total visibility area (original): {total_area:.1f} hectares")
            print(f"   Total drone visibility area: {total_drone_area:.1f} hectares")
            print(f"   Average area per staging: {total_area / len(results):.1f} hectares")
            print(f"   Average drone visibility per staging: {total_drone_area / len(results):.1f} hectares")
            print(f"   Average improvement: +{avg_improvement:.1f}%")
            print(f"   Total ray calculations: {total_calculations:,}")
            print(f"   Processing rate: {total_calculations / total_time:.0f} rays/second")

            print(f"\n💪 PERFORMANCE COMPARISON:")
            sequential_time_estimate = total_time * N_CORES
            print(f"   Estimated sequential time: {sequential_time_estimate / 60:.1f} minutes")
            print(f"   Actual parallel time: {total_time / 60:.1f} minutes")
            print(f"   Speedup achieved: {sequential_time_estimate / total_time:.1f}x")
            print(f"   Efficiency: {(sequential_time_estimate / total_time) / N_CORES * 100:.1f}%")

        else:
            print(f"\n❌ No results to process - check staging points and DSM alignment")

        print(f"\n🎯 Parallel processing complete! Check output files:")
        print(f"   Individual results: {output_folder}")
        print(f"   Final combined: {final_output_path}")