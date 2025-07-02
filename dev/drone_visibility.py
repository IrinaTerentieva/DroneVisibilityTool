import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class SpatialDroneVisibilityAnalyzer:
    def __init__(self, dsm_path: str, drone_height_agl: float = 120.0):
        """
        Initialize the spatial drone visibility analyzer

        Args:
            dsm_path: Path to DSM raster file
            drone_height_agl: Drone height above ground level in meters
        """
        self.dsm_path = dsm_path
        self.drone_height_agl = drone_height_agl

        print(f"Loading DSM from: {dsm_path}")
        # Load DSM
        with rasterio.open(dsm_path) as src:
            self.dsm = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
            self.nodata = src.nodata
            self.pixel_size = abs(self.transform.a)
            print(f"DSM shape: {self.dsm.shape}")
            print(f"DSM resolution: {self.pixel_size:.2f}m")
            print(f"DSM CRS: {self.crs}")

        # # Replace nodata with interpolated values
        # if self.nodata is not None:
        #     mask = self.dsm == self.nodata
        #     if np.any(mask):
        #         print(f"Filling {np.sum(mask)} nodata cells...")
        #         self.dsm = self._fill_nodata(self.dsm, mask)

    def _fill_nodata(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Simple nodata filling using nearest valid neighbors"""
        from scipy.ndimage import distance_transform_edt

        valid_mask = ~mask
        if not np.any(valid_mask):
            return data

        indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
        data[mask] = data[tuple(indices[:, mask])]
        return data

    @classmethod
    def load_staging_points_from_gpkg(cls, gpkg_path: str, layer_name: str = None) -> List[Dict]:
        """Load staging points with cluster information from GeoPackage"""
        print(f"Loading staging points from: {gpkg_path}")

        try:
            if layer_name:
                gdf = gpd.read_file(gpkg_path, layer=layer_name)
            else:
                gdf = gpd.read_file(gpkg_path)

            print(f"Loaded {len(gdf)} staging points")
            print(f"Staging points CRS: {gdf.crs}")
            print(f"Available columns: {list(gdf.columns)}")

            staging_points = []
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Point':
                    # Extract cluster information if available
                    cluster = row.get('cluster', None)
                    if cluster is None:
                        cluster = row.get('Cluster', None)  # Try capitalized version
                    if cluster is None:
                        cluster = row.get('CLUSTER', None)  # Try uppercase version

                    staging_points.append({
                        'coords': (geom.x, geom.y),
                        'cluster': cluster,
                        'original_index': idx
                    })
                elif geom.geom_type in ['MultiPoint', 'GeometryCollection']:
                    for point in geom.geoms:
                        if point.geom_type == 'Point':
                            cluster = row.get('cluster', None)
                            if cluster is None:
                                cluster = row.get('Cluster', None)
                            if cluster is None:
                                cluster = row.get('CLUSTER', None)

                            staging_points.append({
                                'coords': (point.x, point.y),
                                'cluster': cluster,
                                'original_index': idx
                            })

            print(f"Extracted {len(staging_points)} point coordinates")

            # Show cluster information
            clusters = [pt['cluster'] for pt in staging_points if pt['cluster'] is not None]
            if clusters:
                unique_clusters = set(clusters)
                print(f"Found clusters: {sorted(unique_clusters)}")
                cluster_counts = pd.Series(clusters).value_counts()
                print(f"Points per cluster: {dict(cluster_counts)}")
            else:
                print("Warning: No 'cluster' column found or all values are None")

            return staging_points

        except Exception as e:
            print(f"Error loading staging points: {e}")
            raise

    def _world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates"""
        row, col = rowcol(self.transform, x, y)
        return int(row), int(col)

    def _pixel_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates"""
        return xy(self.transform, row, col)

    def _get_elevation_safe(self, row: int, col: int) -> float:
        """Safely get elevation value with bounds checking"""
        if 0 <= row < self.dsm.shape[0] and 0 <= col < self.dsm.shape[1]:
            return float(self.dsm[row, col])
        return np.nan

    def _check_obstruction_depth(self, hit_row: int, hit_col: int,
                                 beam_elevation: float, min_depth_pixels: int = 2) -> bool:
        """
        Check if obstruction has sufficient depth (width) to be considered a real obstacle

        Args:
            hit_row, hit_col: Pixel coordinates of hit point
            beam_elevation: Elevation of beam at hit point
            min_depth_pixels: Minimum number of pixels the obstruction should span

        Returns:
            True if obstruction is significant enough
        """
        if not (0 <= hit_row < self.dsm.shape[0] and 0 <= hit_col < self.dsm.shape[1]):
            return False

        # Check pixels in a small neighborhood around hit point
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

        # Consider it a real obstruction if enough nearby pixels are also elevated
        return obstruction_count >= min_depth_pixels

    def _cast_elevation_beam_with_depth_check(self, start_x: float, start_y: float,
                                              start_elevation: float, bearing_deg: float,
                                              elevation_angle_deg: float,
                                              max_distance: float = 5000.0) -> Optional[Tuple[float, float, float]]:
        """
        Cast a beam and find where it hits a substantial obstruction

        Returns:
            Tuple of (distance, hit_x, hit_y) where beam hits substantial terrain, or None
        """
        bearing_rad = np.radians(bearing_deg)
        elevation_rad = np.radians(elevation_angle_deg)

        # Step size - sample every pixel for accuracy
        step_size = self.pixel_size
        num_steps = int(max_distance / step_size)

        for i in range(1, num_steps + 1):
            horizontal_distance = i * step_size

            # Calculate 3D position along the beam
            x = start_x + horizontal_distance * np.sin(bearing_rad)
            y = start_y + horizontal_distance * np.cos(bearing_rad)
            beam_elevation = start_elevation + horizontal_distance * np.tan(elevation_rad)

            # Get terrain elevation at this point
            row, col = self._world_to_pixel(x, y)
            terrain_elevation = self._get_elevation_safe(row, col)

            if np.isnan(terrain_elevation):
                continue

            # Check if beam intersects terrain
            if beam_elevation <= terrain_elevation:
                # Check if this is a substantial obstruction
                if self._check_obstruction_depth(row, col, beam_elevation, min_depth_pixels=3):
                    return horizontal_distance, x, y

        return None

    def _create_visibility_polygon(self, staging_x: float, staging_y: float,
                                   staging_elevation: float, elevation_angle: float,
                                   num_rays: int = 360, max_distance: float = 5000.0) -> Polygon:
        """
        Create a polygon representing the visible area for a given elevation angle

        Args:
            staging_x, staging_y: Staging point coordinates
            staging_elevation: Staging point elevation
            elevation_angle: Elevation angle for beam casting
            num_rays: Number of rays to cast (360 = 1° increments)
            max_distance: Maximum analysis distance

        Returns:
            Shapely Polygon representing visible area
        """
        visibility_points = []

        for ray_idx in range(num_rays):
            bearing = ray_idx * (360.0 / num_rays)

            # Cast beam and find obstruction
            hit = self._cast_elevation_beam_with_depth_check(
                staging_x, staging_y, staging_elevation,
                bearing, elevation_angle, max_distance
            )

            if hit:
                distance, hit_x, hit_y = hit
                visibility_points.append((hit_x, hit_y))
            else:
                # No obstruction found - extend to max distance
                bearing_rad = np.radians(bearing)
                edge_x = staging_x + max_distance * np.sin(bearing_rad)
                edge_y = staging_y + max_distance * np.cos(bearing_rad)
                visibility_points.append((edge_x, edge_y))

        # Close the polygon by adding the first point at the end
        if visibility_points and visibility_points[0] != visibility_points[-1]:
            visibility_points.append(visibility_points[0])

        # Create polygon
        if len(visibility_points) >= 3:
            return Polygon(visibility_points)
        else:
            # Fallback - create small circle around staging point
            return Point(staging_x, staging_y).buffer(50)

    def analyze_staging_area_spatial(self, staging_point: Dict,
                                     staging_id: int,
                                     elevation_angle: float = 5.0,
                                     max_distance: float = 3000.0,
                                     num_rays: int = 360) -> Dict:
        """
        Analyze visibility from a staging area and create spatial polygon for 5° elevation

        Args:
            staging_point: Dictionary with coords and cluster info
            staging_id: Unique ID for this staging point
            elevation_angle: Elevation angle to analyze (default 5°)
            max_distance: Maximum analysis distance
            num_rays: Number of rays to cast

        Returns:
            Dictionary with staging info and visibility polygon
        """
        staging_x, staging_y = staging_point['coords']
        cluster = staging_point['cluster']

        # Get staging elevation
        staging_row, staging_col = self._world_to_pixel(staging_x, staging_y)
        staging_elevation = self._get_elevation_safe(staging_row, staging_col)

        if np.isnan(staging_elevation):
            raise ValueError(f"Invalid staging location: ({staging_x}, {staging_y})")

        print(f"  Creating {elevation_angle}° visibility polygon for cluster {cluster}...")

        results = {
            'staging_id': staging_id,
            'staging_coords': (staging_x, staging_y),
            'staging_elevation': staging_elevation,
            'cluster': cluster,
            'elevation_angle': elevation_angle,
            'visibility_polygon': None
        }

        # Create visibility polygon
        polygon = self._create_visibility_polygon(
            staging_x, staging_y, staging_elevation,
            elevation_angle, num_rays, max_distance
        )

        results['visibility_polygon'] = polygon

        # Calculate area
        area_ha = polygon.area / 10000  # Convert m² to hectares
        print(f"    Visibility area: {area_ha:.1f} hectares")

        return results

    def analyze_multiple_staging_areas_spatial(self, staging_points: List[Dict],
                                               **kwargs) -> List[Dict]:
        """Analyze multiple staging areas and create spatial outputs with unique IDs"""
        results = []
        for i, staging_point in enumerate(staging_points):
            staging_id = i + 1  # Start IDs from 1
            x, y = staging_point['coords']
            cluster = staging_point['cluster']
            print(f"\nAnalyzing staging area ID {staging_id} (Cluster {cluster}): ({x:.1f}, {y:.1f})")
            try:
                result = self.analyze_staging_area_spatial(staging_point, staging_id, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing staging area ID {staging_id}: {e}")
                continue
        return results

    def export_visibility_to_gpkg(self, results: List[Dict], output_path: str):
        """
        Export 5° visibility analysis results to GeoPackage with cluster information

        Args:
            results: List of analysis results
            output_path: Output GeoPackage path
        """
        print(f"\nExporting results to: {output_path}")

        # Create staging points layer
        print("  Creating staging points layer...")
        staging_geoms = []
        staging_data = []

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

        staging_gdf = gpd.GeoDataFrame(staging_data, geometry=staging_geoms, crs=self.crs)

        # Create visibility zones layer with cluster information
        print("  Creating 5° visibility zones layer with cluster attributes...")
        visibility_geoms = []
        visibility_data = []

        for result in results:
            if result['visibility_polygon'] is not None:
                visibility_geoms.append(result['visibility_polygon'])
                visibility_data.append({
                    'staging_id': result['staging_id'],  # Matching ID to staging points
                    'cluster': result['cluster'],  # CLUSTER INFORMATION ADDED HERE
                    'staging_x': result['staging_coords'][0],
                    'staging_y': result['staging_coords'][1],
                    'staging_elev': result['staging_elevation'],
                    'elevation_angle': result['elevation_angle'],
                    'visibility_area_ha': result['visibility_polygon'].area / 10000,
                    'max_distance_m': 3000,
                    'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                })

        if visibility_geoms:
            visibility_gdf = gpd.GeoDataFrame(visibility_data, geometry=visibility_geoms, crs=self.crs)

            # Export both layers to GeoPackage
            staging_gdf.to_file(output_path, layer="staging_points", driver="GPKG")
            visibility_gdf.to_file(output_path, layer="visibility_zones_5deg", driver="GPKG", mode='a')

            print(f"    ✅ Exported {len(staging_gdf)} staging points to layer 'staging_points'")
            print(f"    ✅ Exported {len(visibility_gdf)} visibility zones to layer 'visibility_zones_5deg'")
            print(f"    🔗 Both layers linked by 'staging_id' and include 'cluster' attribute")

            # Show cluster summary
            cluster_summary = visibility_gdf['cluster'].value_counts()
            print(f"    📊 Visibility zones by cluster: {dict(cluster_summary)}")
        else:
            print("    ❌ No valid visibility polygons to export")

        print(f"\n✅ Export complete! GeoPackage saved to: {output_path}")
        return output_path

    def plot_visibility_spatial(self, results: List[Dict], show_staging_ids: List[int] = None,
                                color_by_cluster: bool = True):
        """Create a spatial plot showing 5° visibility polygons with cluster-based coloring"""
        if not results:
            print("No data to plot")
            return

        # Filter results by staging IDs if specified
        if show_staging_ids:
            filtered_results = [r for r in results if r['staging_id'] in show_staging_ids]
            if not filtered_results:
                print(f"No results found for staging IDs: {show_staging_ids}")
                return
        else:
            filtered_results = results

        fig, ax = plt.subplots(1, 1, figsize=(15, 12))

        if color_by_cluster:
            # Color by cluster
            clusters = [r['cluster'] for r in filtered_results if r['cluster'] is not None]
            unique_clusters = sorted(list(set(clusters))) if clusters else []

            if unique_clusters:
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
                cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

                # Plot visibility polygons grouped by cluster
                for result in filtered_results:
                    staging_id = result['staging_id']
                    staging_x, staging_y = result['staging_coords']
                    polygon = result['visibility_polygon']
                    cluster = result['cluster']

                    if polygon and cluster in cluster_colors:
                        x, y = polygon.exterior.xy
                        color = cluster_colors[cluster]

                        # Fill polygon
                        ax.fill(x, y, alpha=0.3, color=color,
                                label=f'Cluster {cluster}' if staging_id == filtered_results[0]['staging_id'] or
                                                              cluster not in [r['cluster'] for r in filtered_results[
                                                                                                    :filtered_results.index(
                                                                                                        result)]] else "")
                        # Outline
                        ax.plot(x, y, color=color, linewidth=2)

                        # Plot staging point
                        ax.plot(staging_x, staging_y, 'o', color=color, markersize=8,
                                markeredgecolor='black', markeredgewidth=1)

                        # Add staging ID label
                        ax.annotate(f'ID {staging_id}\nC{cluster}', (staging_x, staging_y),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=9, fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:
                color_by_cluster = False
                print("Warning: No cluster information found, using staging ID colors")

        if not color_by_cluster:
            # Color by staging ID (fallback)
            colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_results)))

            for i, result in enumerate(filtered_results):
                staging_id = result['staging_id']
                staging_x, staging_y = result['staging_coords']
                polygon = result['visibility_polygon']
                cluster = result['cluster']

                if polygon:
                    x, y = polygon.exterior.xy
                    color = colors[i]

                    # Fill polygon
                    ax.fill(x, y, alpha=0.3, color=color,
                            label=f'Staging {staging_id} (C{cluster})')
                    # Outline
                    ax.plot(x, y, color=color, linewidth=2)

                    # Plot staging point
                    ax.plot(staging_x, staging_y, 'o', color=color, markersize=8,
                            markeredgecolor='black', markeredgewidth=1)

                    # Add staging ID label
                    ax.annotate(f'ID {staging_id}', (staging_x, staging_y),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Formatting
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')

        if color_by_cluster and unique_clusters:
            ax.set_title('Drone Visibility Analysis - 5° Elevation Zones by Cluster\n'
                         'Each color represents a different cluster')
        else:
            ax.set_title('Drone Visibility Analysis - 5° Elevation Zones\n'
                         'Each color represents a different staging point')

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

        print(f"Displayed visibility zones for {len(filtered_results)} staging points")
        if show_staging_ids:
            print(f"Filtered to show only staging IDs: {show_staging_ids}")


# Main execution
if __name__ == "__main__":
    # File paths - update these to your actual paths
    dsm_path = "file:///media/irina/My Book/Petronas/DATA/FullData/DSM_may25.tif"
    staging_gpkg = "file:///home/irina/Desktop/petronas_staging_day2.gpkg"
    output_gpkg = "/media/irina/My Book/Petronas/DATA/tmp/petronas_staging_day2.gpkg"

    # Load staging points with cluster information
    staging_points = SpatialDroneVisibilityAnalyzer.load_staging_points_from_gpkg(staging_gpkg)

    # Initialize analyzer
    analyzer = SpatialDroneVisibilityAnalyzer(dsm_path, drone_height_agl=120.0)

    # Analyze all staging areas - create ONLY 5° visibility polygons
    print(f"\nStarting spatial visibility analysis of {len(staging_points)} staging areas...")
    print("Creating visibility polygons for 5° elevation angle with cluster information")

    results = analyzer.analyze_multiple_staging_areas_spatial(
        staging_points,
        elevation_angle=5.0,  # ONLY 5° elevation angle
        max_distance=3000,  # 3km analysis radius
        num_rays=360  # 1-degree increments for smooth polygons
    )

    # Export results to GeoPackage with cluster information
    output_file = analyzer.export_visibility_to_gpkg(results, output_gpkg)

    # Show summary with cluster information
    print(f"\n{'=' * 60}")
    print("ANALYSIS SUMMARY WITH CLUSTER INFORMATION")
    print(f"{'=' * 60}")

    total_area = 0
    cluster_areas = {}

    for result in results:
        staging_id = result['staging_id']
        coords = result['staging_coords']
        cluster = result['cluster']
        area_ha = result['visibility_polygon'].area / 10000
        total_area += area_ha

        # Track area by cluster
        if cluster not in cluster_areas:
            cluster_areas[cluster] = []
        cluster_areas[cluster].append(area_ha)

        print(f"Staging ID {staging_id:2d} (Cluster {cluster}): ({coords[0]:.1f}, {coords[1]:.1f}) - "
              f"Visibility area: {area_ha:.1f} ha")

    print(f"\nTotal visibility area: {total_area:.1f} hectares")
    print(f"Average area per staging point: {total_area / len(results):.1f} hectares")

    # Cluster summary
    print(f"\nCLUSTER SUMMARY:")
    for cluster, areas in cluster_areas.items():
        cluster_total = sum(areas)
        cluster_avg = np.mean(areas)
        cluster_count = len(areas)
        print(
            f"  Cluster {cluster}: {cluster_count} points, {cluster_total:.1f} ha total, {cluster_avg:.1f} ha average")

    # Create visualization showing all staging points colored by cluster
    print(f"\nCreating cluster-based visualization...")
    analyzer.plot_visibility_spatial(results, color_by_cluster=True)

    print(f"\n🎉 Analysis complete!")
    print(f"✅ Results exported to: {output_file}")
    print(f"📊 Processed {len(results)} staging areas with cluster information")
    print(f"📁 GeoPackage contains:")
    print(f"   - staging_points: Staging point locations (with staging_id and cluster)")
    print(f"   - visibility_zones_5deg: 5° elevation visibility zones (with cluster attribute)")
    print(f"\n💡 In QGIS, you can:")
    print(f"   1. Load both layers")
    print(f"   2. Filter visibility zones by cluster: cluster = 'A' or cluster IN ('A', 'B')")
    print(f"   3. Style visibility zones by cluster for easy identification")
    print(f"   4. Calculate cluster-based statistics")
    print(f"   5. Join layers on staging_id for advanced analysis")