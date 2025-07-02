import os.path

import numpy as np
import rasterio
from rasterio.transform import rowcol
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
from rasterio.crs import CRS
warnings.filterwarnings('ignore')


class DetailedMobilityVisibilityAnalyzer:
    def __init__(self, dsm_path: str, drone_height_agl: float = 120.0, verbose: bool = True):
        self.dsm_path = dsm_path
        self.drone_height_agl = drone_height_agl
        self.verbose = verbose

        if self.verbose:
            print(f"Loading DSM from: {dsm_path}")

        with rasterio.open(dsm_path) as src:
            self.dsm = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
            self.crs = CRS.from_epsg(3157)

            self.nodata = src.nodata
            self.pixel_size = abs(self.transform.a)

        if self.verbose:
            print(f"DSM shape: {self.dsm.shape}")
            print(f"DSM resolution: {self.pixel_size:.2f}m")
            print(f"DSM CRS: {self.crs}")

        # # Nodata filling
        # if self.nodata is not None:
        #     mask = self.dsm == self.nodata
        #     if np.any(mask):
        #         if self.verbose:
        #             print(f"Filling {np.sum(mask)} nodata cells...")
        #         self.dsm = self._fill_nodata(self.dsm, mask)

    def _fill_nodata(self, data, mask):
        from scipy.ndimage import distance_transform_edt
        valid_mask = ~mask
        if not np.any(valid_mask):
            return data
        indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
        data[mask] = data[tuple(indices[:, mask])]
        return data

    @classmethod
    def load_staging_points_from_gpkg(cls, gpkg_path, layer_name=None):
        """Load staging points from GeoPackage"""
        if hasattr(cls, '_verbose_instance') and cls._verbose_instance:
            print(f"Loading staging points from: {gpkg_path}")

        gdf = gpd.read_file(gpkg_path, layer=layer_name) if layer_name else gpd.read_file(gpkg_path)

        if hasattr(cls, '_verbose_instance') and cls._verbose_instance:
            print(f"Loaded {len(gdf)} staging points")
            print(f"Staging points CRS: {gdf.crs}")

        staging_points = [(geom.x, geom.y) for geom in gdf.geometry if geom.geom_type == "Point"]

        if hasattr(cls, '_verbose_instance') and cls._verbose_instance:
            print(f"Extracted {len(staging_points)} point coordinates")

        return staging_points

    def _world_to_pixel(self, x, y):
        row, col = rowcol(self.transform, x, y)
        return int(row), int(col)

    def _get_elevation_safe(self, row, col):
        if 0 <= row < self.dsm.shape[0] and 0 <= col < self.dsm.shape[1]:
            return float(self.dsm[row, col])
        return np.nan

    def _find_suitable_location(self, center_x, center_y, search_radius=10.0):
        """
        Find the lowest elevation within a radius (meters) of (center_x, center_y).
        Returns (x, y, elevation).
        """
        row_c, col_c = self._world_to_pixel(center_x, center_y)
        radius_pix = int(np.ceil(search_radius / self.pixel_size))

        # Clip a square window around center
        row_min = max(0, row_c - radius_pix)
        row_max = min(self.dsm.shape[0], row_c + radius_pix + 1)
        col_min = max(0, col_c - radius_pix)
        col_max = min(self.dsm.shape[1], col_c + radius_pix + 1)

        window = self.dsm[row_min:row_max, col_min:col_max]

        # Make a circular mask
        y_idx, x_idx = np.ogrid[:window.shape[0], :window.shape[1]]
        dist = np.sqrt((y_idx - (row_c - row_min)) ** 2 + (x_idx - (col_c - col_min)) ** 2) * self.pixel_size
        mask = dist <= search_radius

        valid = window.copy()
        valid[~mask] = np.nan
        if np.all(np.isnan(valid)):
            return None  # No valid points

        min_idx = np.nanargmin(valid)
        min_row_local, min_col_local = np.unravel_index(min_idx, valid.shape)
        min_row = row_min + min_row_local
        min_col = col_min + min_col_local
        min_elev = valid[min_row_local, min_col_local]
        # Convert back to x,y
        from rasterio.transform import xy
        min_x, min_y = xy(self.transform, min_row, min_col, offset='center')
        return min_x, min_y, min_elev

    def _generate_sample_points(self, center_x, center_y, buffer_radius, num_points=20, sampling='strategic'):
        """Generate sample points with terrain validation for suitable operator locations"""
        points = []

        # Center point - validate it's suitable
        center_location = self._find_suitable_location(center_x, center_y)
        if center_location:
            actual_x, actual_y, actual_elev = center_location
            points.append({
                'point_id': 0,
                'coords': (actual_x, actual_y),
                'distance_from_center': 0.0,
                'bearing_from_center': 0.0,
                'point_type': 'center',
                'elevation': actual_elev,
                # 'terrain_suitability': score
            })
            if self.verbose:
                print(
                    f"      Center point adjusted by {np.sqrt((actual_x - center_x) ** 2 + (actual_y - center_y) ** 2):.1f}m for terrain suitability")
        else:
            # Fallback to original center if no suitable location found
            center_row, center_col = self._world_to_pixel(center_x, center_y)
            center_elev = self._get_elevation_safe(center_row, center_col)
            points.append({
                'point_id': 0,
                'coords': (center_x, center_y),
                'distance_from_center': 0.0,
                'bearing_from_center': 0.0,
                'point_type': 'center',
                'elevation': center_elev,
                # 'terrain_suitability': 999.0  # High score = poor suitability
            })

        if num_points == 1:
            return points

        if sampling == 'random':
            for i in range(1, num_points):
                max_attempts = 20  # More attempts for random sampling
                placed = False

                for attempt in range(max_attempts):
                    r = buffer_radius * np.sqrt(np.random.uniform(0, 1))
                    theta = np.random.uniform(0, 2 * np.pi)
                    target_x = center_x + r * np.cos(theta)
                    target_y = center_y + r * np.sin(theta)

                    # Find suitable location near this target
                    suitable_location = self._find_suitable_location(target_x, target_y, search_radius=5.0)

                    if suitable_location:
                        actual_x, actual_y, actual_elev = suitable_location
                        actual_r = np.sqrt((actual_x - center_x) ** 2 + (actual_y - center_y) ** 2)
                        bearing = np.degrees(np.arctan2(actual_y - center_y, actual_x - center_x)) % 360

                        points.append({
                            'point_id': i,
                            'coords': (actual_x, actual_y),
                            'distance_from_center': actual_r,
                            'bearing_from_center': bearing,
                            'point_type': 'random',
                            'elevation': actual_elev,
                            # 'terrain_suitability': score
                        })
                        placed = True
                        break

                if not placed and self.verbose:
                    print(f"      Warning: Could not find suitable terrain for random point {i}")
        else:
            # Strategic radial pattern with terrain validation
            num_rings = 5
            points_per_ring = (num_points - 1) // num_rings
            pid = 1

            for ring in range(1, num_rings + 1):
                ring_radius = buffer_radius * ring / num_rings
                points_in_ring = points_per_ring + (
                    1 if (ring == num_rings and (num_points - 1) % num_rings > 0) else 0)

                for i in range(points_in_ring):
                    base_angle = 2 * np.pi * i / points_in_ring
                    angle = base_angle + np.random.uniform(-0.2, 0.2)
                    r = ring_radius + np.random.uniform(-0.1 * ring_radius, 0.1 * ring_radius)

                    # Initial target location
                    target_x = center_x + r * np.cos(angle)
                    target_y = center_y + r * np.sin(angle)

                    # Find suitable terrain near this target
                    suitable_location = self._find_suitable_location(target_x, target_y, search_radius=5.0)

                    if suitable_location:
                        actual_x, actual_y, actual_elev = suitable_location
                        actual_r = np.sqrt((actual_x - center_x) ** 2 + (actual_y - center_y) ** 2)
                        bearing = (90 - np.degrees(np.arctan2(actual_y - center_y, actual_x - center_x))) % 360

                        points.append({
                            'point_id': pid,
                            'coords': (actual_x, actual_y),
                            'distance_from_center': actual_r,
                            'bearing_from_center': bearing,
                            'point_type': f'ring_{ring}',
                            'elevation': actual_elev,
                            # 'terrain_suitability': score
                        })
                    else:
                        # Fallback to original location if no suitable terrain found
                        test_row, test_col = self._world_to_pixel(target_x, target_y)
                        test_elev = self._get_elevation_safe(test_row, test_col)
                        bearing = (90 - np.degrees(angle)) % 360

                        points.append({
                            'point_id': pid,
                            'coords': (target_x, target_y),
                            'distance_from_center': r,
                            'bearing_from_center': bearing,
                            'point_type': f'ring_{ring}',
                            'elevation': test_elev if not np.isnan(test_elev) else 0.0,
                            # 'terrain_suitability': 999.0  # Poor suitability flag
                        })

                        if self.verbose:
                            print(
                                f"      Warning: Using original location for ring {ring} point {i} (no suitable terrain)")

                    pid += 1
                    if pid >= num_points:
                        break
                if pid >= num_points:
                    break

        return points[:num_points]

    def _cast_visibility_ray(self, start_x, start_y, start_elev, bearing_deg, elev_angle_deg, max_distance=3000):
        """Cast visibility ray with detailed hit information"""
        bearing_rad = np.radians(bearing_deg)
        elev_rad = np.radians(elev_angle_deg)
        step = self.pixel_size * 1.5
        num_steps = int(max_distance / step)

        for i in range(1, num_steps + 1):
            d = i * step
            x = start_x + d * np.sin(bearing_rad)
            y = start_y + d * np.cos(bearing_rad)
            beam_elev = start_elev + d * np.tan(elev_rad)
            row, col = self._world_to_pixel(x, y)
            terrain_elev = self._get_elevation_safe(row, col)

            if np.isnan(terrain_elev):
                continue

            if beam_elev <= terrain_elev:
                return {
                    'hit': True,
                    'hit_distance': d,
                    'hit_coords': (x, y),
                    'hit_elevation': terrain_elev,
                    'beam_elevation': beam_elev,
                    'obstruction_height': terrain_elev - beam_elev
                }

        return {
            'hit': False,
            'hit_distance': max_distance,
            'hit_coords': None,
            'hit_elevation': None,
            'beam_elevation': start_elev + max_distance * np.tan(elev_rad),
            'obstruction_height': 0
        }

    def _analyze_one_staging(self, staging_x, staging_y, staging_id, buffer, num_points, num_rays, elev_angle,
                             max_distance, sampling='strategic'):
        """Analyze one staging area with detailed tracking"""
        # Get elevation at staging center
        row, col = self._world_to_pixel(staging_x, staging_y)
        staging_elev = self._get_elevation_safe(row, col)
        if np.isnan(staging_elev):
            raise ValueError(f"Invalid elevation at staging ({staging_x}, {staging_y})")

        if self.verbose:
            print(f"  Analyzing staging ID {staging_id}: ({staging_x:.1f}, {staging_y:.1f})")
            print(f"    {num_points} {sampling} sample points × {num_rays} rays = {num_points * num_rays} calculations")

        sample_points = self._generate_sample_points(staging_x, staging_y, buffer, num_points, sampling)
        bearings = np.linspace(0, 360, num_rays, endpoint=False)

        all_ray_data = []
        max_dist_by_bearing = {}
        individual_sample_polygons = []

        # For each bearing, for each sample point, cast a ray and keep longest
        for b_idx, bearing in enumerate(bearings):
            ray_id = b_idx + 1
            best = {'distance': 0, 'point_id': None, 'ray_data': None}

            for sp in sample_points:
                pid = sp['point_id']
                sx, sy = sp['coords']
                srow, scol = self._world_to_pixel(sx, sy)
                selev = self._get_elevation_safe(srow, scol)
                if np.isnan(selev):
                    continue

                ray = self._cast_visibility_ray(sx, sy, selev, bearing, elev_angle, max_distance)
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
                    'hit_terrain': ray['hit'],
                    'hit_coords': ray['hit_coords'],
                    'hit_elevation': ray['hit_elevation'],
                    'beam_elevation': ray['beam_elevation'],
                    'obstruction_height': ray['obstruction_height']
                }
                all_ray_data.append(ray_data)

                if ray['hit_distance'] > best['distance']:
                    best = {'distance': ray['hit_distance'], 'point_id': pid, 'ray_data': ray_data}

            max_dist_by_bearing[bearing] = best

        # Build final mobility-enhanced polygon from max distances per bearing
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

        # Build center-only polygon for improvement comparison
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

        # Calculate areas and improvement
        total_area = visibility_polygon.area / 10000  # hectares
        center_area = center_polygon.area / 10000
        improvement = ((total_area - center_area) / center_area * 100) if center_area > 0 else 0

        # Usage statistics: which point was most useful
        point_usage = {}
        for b in bearings:
            pid = max_dist_by_bearing[b]['point_id']
            if pid is not None:
                point_usage[pid] = point_usage.get(pid, 0) + 1

        # Individual polygons for each sample point
        for sp in sample_points:
            pid = sp['point_id']
            sx, sy = sp['coords']
            srow, scol = self._world_to_pixel(sx, sy)
            selev = self._get_elevation_safe(srow, scol)
            if np.isnan(selev):
                continue

            poly_pts = []
            for b in bearings:
                rays = [r for r in all_ray_data if r['point_id'] == pid and r['bearing_deg'] == b]
                if rays:
                    dist = rays[0]['visibility_distance']
                    x = sx + dist * np.sin(np.radians(b))
                    y = sy + dist * np.cos(np.radians(b))
                    poly_pts.append((x, y))

            if poly_pts and poly_pts[0] != poly_pts[-1]:
                poly_pts.append(poly_pts[0])

            if len(poly_pts) >= 3:
                sample_poly = Polygon(poly_pts)
                sample_area = sample_poly.area / 10000
                individual_sample_polygons.append({
                    'staging_id': staging_id,
                    'point_id': pid,
                    'staging_point_id': f"{staging_id}_{pid}",
                    'sample_coords': (sx, sy),
                    'sample_elevation': selev,
                    'point_type': sp['point_type'],
                    'visibility_polygon': sample_poly,
                    'visibility_area_ha': sample_area,
                    'distance_from_center': sp['distance_from_center'],
                    'bearing_from_center': sp['bearing_from_center'],
                    # 'terrain_suitability': sp.get('terrain_suitability', 999.0)
                })

        if self.verbose:
            print(f"    Combined visibility area: {total_area:.1f} ha (+{improvement:.1f}% vs center only)")
            top_points = sorted(point_usage.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    Most useful sample points: {top_points}")
            print(f"    Created {len(individual_sample_polygons)} individual visibility polygons")

        # Return comprehensive result
        return {
            'staging_id': staging_id,
            'staging_coords': (staging_x, staging_y),
            'staging_elevation': staging_elev,
            'mobility_buffer': buffer,
            'elevation_angle': elev_angle,
            'num_sample_points': len(sample_points),
            'num_rays': num_rays,
            'total_ray_calculations': len(all_ray_data),
            'visibility_polygon': visibility_polygon,
            'center_area': center_area,
            'total_area': total_area,
            'improvement_pct': improvement,
            'sample_points': sample_points,
            'individual_sample_polygons': individual_sample_polygons,
            'all_ray_data': all_ray_data,
            'max_distances_by_bearing': max_dist_by_bearing,
            'point_usage_stats': point_usage,
            'sampling_method': sampling
        }

    def analyze_multiple_staging_areas(self, staging_points, **kwargs):
        """Analyze multiple staging areas with improved error handling"""
        # Set class variable for verbose mode in classmethod
        DetailedMobilityVisibilityAnalyzer._verbose_instance = self.verbose

        results = []
        for i, (x, y) in enumerate(staging_points):
            staging_id = i + 1
            if self.verbose:
                print(f"\nAnalyzing staging area ID {staging_id}/{len(staging_points)}: ({x:.1f}, {y:.1f})")
            try:
                res = self._analyze_one_staging(x, y, staging_id, **kwargs)
                results.append(res)
            except Exception as e:
                if self.verbose:
                    print(f"Error analyzing staging ID {staging_id}: {e}")
                continue
        return results

    def export_detailed_to_gpkg(self, results: List[Dict], output_path: str):
        """Export only 3 essential layers: staging points, individual zones, combined zones"""
        if self.verbose:
            print(f"\nExporting essential results to: {output_path}")

        # Layer 1: Staging points (main locations)
        staging_geoms = []
        staging_data = []

        for result in results:
            staging_geoms.append(Point(result['staging_coords']))
            staging_data.append({
                'staging_id': result['staging_id'],
                'staging_x': result['staging_coords'][0],
                'staging_y': result['staging_coords'][1],
                'staging_elev': result['staging_elevation'],
                'mobility_buffer': result['mobility_buffer'],
                'elevation_angle': result['elevation_angle'],
                'num_sample_points': result['num_sample_points'],
                'num_rays': result['num_rays'],
                'sampling_method': result['sampling_method'],
                'visibility_area_ha': result['total_area'],
                'area_improvement_pct': result['improvement_pct']
            })

        staging_gdf = gpd.GeoDataFrame(staging_data, geometry=staging_geoms, crs=self.crs)

        # Layer 2: Individual visibility zones (one polygon per sample point)
        individual_visibility_geoms = []
        individual_visibility_data = []

        for result in results:
            for sample_poly_data in result['individual_sample_polygons']:
                individual_visibility_geoms.append(sample_poly_data['visibility_polygon'])
                individual_visibility_data.append({
                    'staging_id': sample_poly_data['staging_id'],
                    'point_id': sample_poly_data['point_id'],
                    'staging_point_id': sample_poly_data['staging_point_id'],
                    'sample_x': sample_poly_data['sample_coords'][0],
                    'sample_y': sample_poly_data['sample_coords'][1],
                    'point_type': sample_poly_data['point_type'],
                    'visibility_area_ha': sample_poly_data['visibility_area_ha'],
                    'distance_from_center': sample_poly_data['distance_from_center'],
                    # 'terrain_suitability': sample_poly_data.get('terrain_suitability', 999.0),
                    'elevation_angle': result['elevation_angle'],
                    'analysis_method': 'individual_sample_point'
                })

        individual_visibility_gdf = gpd.GeoDataFrame(individual_visibility_data,
                                                     geometry=individual_visibility_geoms, crs=self.crs)

        # Layer 3: Combined mobility-enhanced visibility zones
        combined_visibility_geoms = []
        combined_visibility_data = []

        for result in results:
            if result['visibility_polygon'] is not None:
                combined_visibility_geoms.append(result['visibility_polygon'])
                combined_visibility_data.append({
                    'staging_id': result['staging_id'],
                    'staging_point_id': result['staging_id'],  # Same as staging_id for combined
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
                })

        # NEW LAYER: Save strategic points as layer "sample_points"
        sample_points_geoms = []
        sample_points_data = []
        for result in results:
            for pt in result['sample_points']:
                sample_points_geoms.append(Point(pt['coords']))
                sample_points_data.append({
                    'staging_id': result['staging_id'],
                    'point_id': pt['point_id'],
                    'point_type': pt['point_type'],
                    'distance_from_center': pt['distance_from_center'],
                    'bearing_from_center': pt['bearing_from_center'],
                    'elevation': pt['elevation'],
                })
        sample_points_gdf = gpd.GeoDataFrame(sample_points_data, geometry=sample_points_geoms, crs=self.crs)
        sample_points_gdf.to_file(output_path, layer="sample_points", driver="GPKG", mode='a')
        if self.verbose:
            print(f"    ✅ Exported {len(sample_points_gdf)} sample points (strategic locations)")

        combined_visibility_gdf = gpd.GeoDataFrame(combined_visibility_data,
                                                   geometry=combined_visibility_geoms, crs=self.crs)
        print('combined_visibility_gdf head', combined_visibility_gdf.columns)

        # Export only 3 essential layers to GeoPackage
        staging_gdf.to_file(output_path, layer="staging_points", driver="GPKG")
        individual_visibility_gdf.to_file(output_path, layer="individual_visibility_zones", driver="GPKG", mode='a')
        combined_visibility_gdf.to_file(output_path, layer="combined_visibility_zones", driver="GPKG", mode='a')

        if self.verbose:
            print(f"    ✅ Exported {len(staging_gdf)} staging points")
            print(f"    ✅ Exported {len(individual_visibility_gdf)} individual visibility zones")
            print(f"    ✅ Exported {len(combined_visibility_gdf)} combined mobility-enhanced zones")
            print(f"    🔗 All linked by staging_id + staging_point_id")
            print(f"    📁 3 essential layers only (no sample points, no ray CSV)")

        return output_path

    def plot_detailed_results(self, results: List[Dict], show_sample_points: bool = True,
                              show_individual_zones: bool = False, staging_ids: List[int] = None):
        """Create detailed visualization with multiple options"""
        import matplotlib.pyplot as plt

        if not results:
            print("No data to plot")
            return

        # Filter by staging IDs if specified
        if staging_ids:
            results = [r for r in results if r['staging_id'] in staging_ids]
            if not results:
                print(f"No results found for staging IDs: {staging_ids}")
                return

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Plot 1: Combined visibility zones with sample points
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))

        for i, result in enumerate(results):
            staging_id = result['staging_id']
            staging_x, staging_y = result['staging_coords']
            polygon = result['visibility_polygon']
            color = colors[i]

            if polygon:
                x, y = polygon.exterior.xy
                ax1.fill(x, y, alpha=0.3, color=color, label=f'Staging {staging_id} Combined')
                ax1.plot(x, y, color=color, linewidth=2)

                # Plot staging point
                ax1.plot(staging_x, staging_y, 'o', color=color, markersize=10,
                         markeredgecolor='black', markeredgewidth=2, zorder=10)
                ax1.annotate(f'S{staging_id}', (staging_x, staging_y),
                             xytext=(5, 5), textcoords='offset points', fontweight='bold', fontsize=12)

                # Plot mobility buffer
                circle = plt.Circle((staging_x, staging_y), result['mobility_buffer'],
                                    fill=False, color=color, linestyle='--', alpha=0.5)
                ax1.add_patch(circle)

                # Plot sample points if requested
                if show_sample_points:
                    for sample_point in result['sample_points']:
                        sx, sy = sample_point['coords']
                        point_id = sample_point['point_id']
                        usage = result['point_usage_stats'].get(point_id, 0)

                        if point_id == 0:  # Center point
                            ax1.plot(sx, sy, 's', color=color, markersize=8,
                                     markeredgecolor='black', markeredgewidth=1, zorder=5)
                        else:
                            size = max(3, usage * 2)
                            ax1.plot(sx, sy, 'o', color=color, markersize=size,
                                     alpha=0.7, markeredgecolor='white', markeredgewidth=0.5, zorder=5)

                            if usage >= 5:
                                ax1.annotate(f'P{point_id}', (sx, sy),
                                             xytext=(3, 3), textcoords='offset points',
                                             fontsize=8, alpha=0.8)

        ax1.set_xlabel('Easting (m)')
        ax1.set_ylabel('Northing (m)')
        ax1.set_title('Mobility-Enhanced Visibility Analysis\n(Point size = usage frequency)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Plot 2: Performance metrics
        ax2 = axes[0, 1]
        staging_ids_plot = [r['staging_id'] for r in results]
        areas = [r['total_area'] for r in results]
        improvements = [r['improvement_pct'] for r in results]

        ax2_twin = ax2.twinx()

        bars1 = ax2.bar([x - 0.2 for x in staging_ids_plot], areas, 0.4,
                        label='Visibility Area (ha)', color='skyblue')
        bars2 = ax2_twin.bar([x + 0.2 for x in staging_ids_plot], improvements, 0.4,
                             label='Improvement (%)', color='orange')

        ax2.set_xlabel('Staging ID')
        ax2.set_ylabel('Visibility Area (hectares)', color='skyblue')
        ax2_twin.set_ylabel('Improvement (%)', color='orange')
        ax2.set_title('Performance Metrics by Staging Area')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars1, areas):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(areas) * 0.01,
                     f'{value:.1f}', ha='center', va='bottom', fontsize=10)

        for bar, value in zip(bars2, improvements):
            ax2_twin.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(improvements) * 0.01,
                          f'+{value:.1f}%', ha='center', va='bottom', fontsize=10)

        # Plot 3: Sample point usage analysis
        ax3 = axes[1, 0]

        all_usage_data = []
        for result in results:
            for point_id, usage_count in result['point_usage_stats'].items():
                sample_point = next(p for p in result['sample_points'] if p['point_id'] == point_id)
                all_usage_data.append({
                    'point_id': point_id,
                    'point_type': sample_point['point_type'],
                    'distance_from_center': sample_point['distance_from_center'],
                    'usage_count': usage_count,
                    'staging_id': result['staging_id']
                })

        if all_usage_data:
            usage_df = pd.DataFrame(all_usage_data)
            type_usage = usage_df.groupby('point_type')['usage_count'].mean()
            type_usage.plot(kind='bar', ax=ax3, color='lightcoral')
            ax3.set_title('Average Ray Usage by Sample Point Type')
            ax3.set_ylabel('Average Usage Count')
            ax3.tick_params(axis='x', rotation=45)

        # Plot 4: Distance vs Usage scatter
        ax4 = axes[1, 1]

        if all_usage_data:
            distances = usage_df['distance_from_center']
            usages = usage_df['usage_count']
            point_types = usage_df['point_type']

            for ptype in point_types.unique():
                mask = point_types == ptype
                ax4.scatter(distances[mask], usages[mask], label=ptype, alpha=0.7, s=50)

            ax4.set_xlabel('Distance from Staging Center (m)')
            ax4.set_ylabel('Usage Count')
            ax4.set_title('Sample Point Usage vs Distance from Center')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Main execution
if __name__ == "__main__":
    # File paths - update these to your actual paths
    dsm_path = "file:///media/irina/My Book/Petronas/DATA/FullData/DSM_may25.tif"
    staging_gpkg = "file:///home/irina/Desktop/petronas_staging_day2.gpkg"

    # Analysis parameters
    MOBILITY_BUFFER = 100.0  # 100m operator movement radius
    NUM_SAMPLE_POINTS = 100  # Strategic sample points
    NUM_RAYS = 180  # 5° increments for high accuracy
    ELEVATION_ANGLE = 5.0  # 5° elevation angle
    MAX_DISTANCE = 2000  # 3km analysis radius
    SAMPLING_METHOD = 'strategic'  # 'strategic' or 'random'

    output_gpkg = staging_gpkg.replace('.gpkg', f'_visibility_{ELEVATION_ANGLE}angle.gpkg')

    print("=" * 80)
    print("IMPROVED DETAILED MOBILITY-ENHANCED DRONE VISIBILITY ANALYZER")
    print("=" * 80)
    print("Features:")
    print(f"- Sampling method: {SAMPLING_METHOD}")
    print(f"- Sample points: {NUM_SAMPLE_POINTS} points in {MOBILITY_BUFFER}m buffer")
    print(f"- Ray accuracy: {NUM_RAYS} rays ({360 / NUM_RAYS}° increments)")
    print(f"- Detailed tracking: staging_id + point_id + ray_id")
    print(f"- Individual visibility zones for each sample point")
    print(f"- Combined mobility-enhanced zones")

    # Load and analyze
    staging_points = DetailedMobilityVisibilityAnalyzer.load_staging_points_from_gpkg(staging_gpkg)
    analyzer = DetailedMobilityVisibilityAnalyzer(dsm_path, drone_height_agl=120.0, verbose=True)

    results = analyzer.analyze_multiple_staging_areas(
        staging_points,
        buffer=MOBILITY_BUFFER,
        num_points=NUM_SAMPLE_POINTS,
        num_rays=NUM_RAYS,
        elev_angle=ELEVATION_ANGLE,
        max_distance=MAX_DISTANCE,
        sampling=SAMPLING_METHOD
    )

    # Export essential results (3 layers only)
    output_file = analyzer.export_detailed_to_gpkg(results, output_gpkg)

    # Results summary
    print(f"\n{'=' * 80}")
    print("DETAILED ANALYSIS SUMMARY")
    print(f"{'=' * 80}")

    total_area = 0
    total_improvement = 0
    total_calculations = 0

    for result in results:
        staging_id = result['staging_id']
        coords = result['staging_coords']
        area_ha = result['total_area']
        improvement = result['improvement_pct']
        calculations = result['total_ray_calculations']
        individual_zones = len(result['individual_sample_polygons'])

        total_area += area_ha
        total_improvement += improvement
        total_calculations += calculations

        # Show top 3 most useful sample points
        top_points = sorted(result['point_usage_stats'].items(), key=lambda x: x[1], reverse=True)[:3]

        print(f"Staging ID {staging_id:2d}: ({coords[0]:.1f}, {coords[1]:.1f})")
        print(f"  Combined area: {area_ha:.1f} ha (+{improvement:.1f}% improvement)")
        print(f"  Individual zones: {individual_zones} visibility polygons")
        print(f"  Ray calculations: {calculations} ({NUM_SAMPLE_POINTS} points × {NUM_RAYS} rays)")
        print(f"  Most useful points: {top_points}")

    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  Total visibility area: {total_area:.1f} hectares")
    print(f"  Average area per staging: {total_area / len(results):.1f} hectares")
    print(f"  Average improvement: +{total_improvement / len(results):.1f}%")
    print(f"  Total ray calculations: {total_calculations:,}")
    print(f"  Total individual zones: {sum(len(r['individual_sample_polygons']) for r in results)}")

    # Create detailed visualization
    print(f"\nCreating detailed visualization...")
    analyzer.plot_detailed_results(results, show_sample_points=True)

    print(f"\n🎉 IMPROVED Analysis Complete!")
    print(f"✅ Results exported to: {output_file}")
    print(f"📋 Processed {len(results)} staging areas with essential data only")
