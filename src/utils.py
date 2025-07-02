import geopandas as gpd
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_staging_points_from_gpkg(gpkg_path: str,
                                  layer_name: Optional[str] = None) -> List[Dict]:
    """
    Load staging points from GeoPackage.

    Args:
        gpkg_path: Path to GeoPackage file
        layer_name: Optional specific layer name

    Returns:
        List of dictionaries with staging point information
    """
    logger.info(f"Loading staging points from: {gpkg_path}")

    try:
        # Load GeoDataFrame
        if layer_name:
            gdf = gpd.read_file(gpkg_path, layer=layer_name)
        else:
            gdf = gpd.read_file(gpkg_path)

        logger.info(f"Loaded {len(gdf)} features")
        logger.info(f"CRS: {gdf.crs}")
        logger.info(f"Columns: {list(gdf.columns)}")

        # Extract point coordinates and attributes
        staging_points = []

        for idx, row in gdf.iterrows():
            geom = row.geometry

            if geom.geom_type == 'Point':
                # Try different cluster column names
                cluster = None
                for col_name in ['cluster', 'Cluster', 'CLUSTER']:
                    if col_name in row and pd.notna(row[col_name]):
                        cluster = row[col_name]
                        break

                staging_points.append({
                    'coords': (geom.x, geom.y),
                    'cluster': cluster,
                    'original_index': idx,
                    'attributes': row.to_dict()
                })

            elif geom.geom_type in ['MultiPoint', 'GeometryCollection']:
                # Handle multi-point geometries
                for point in geom.geoms:
                    if point.geom_type == 'Point':
                        cluster = None
                        for col_name in ['cluster', 'Cluster', 'CLUSTER']:
                            if col_name in row and pd.notna(row[col_name]):
                                cluster = row[col_name]
                                break

                        staging_points.append({
                            'coords': (point.x, point.y),
                            'cluster': cluster,
                            'original_index': idx,
                            'attributes': row.to_dict()
                        })

        logger.info(f"Extracted {len(staging_points)} point coordinates")

        # Log cluster information if available
        clusters = [pt['cluster'] for pt in staging_points if pt['cluster'] is not None]
        if clusters:
            unique_clusters = sorted(set(clusters))
            logger.info(f"Found {len(unique_clusters)} clusters: {unique_clusters}")

            # Count points per cluster
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                logger.info(f"  Cluster {cluster}: {count} points")
        else:
            logger.warning("No cluster information found in data")

        return staging_points

    except Exception as e:
        logger.error(f"Error loading staging points: {e}")
        raise


def validate_config(config: dict) -> bool:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_paths = ['dsm_path', 'staging_gpkg', 'output_gpkg']
    required_params = ['drone_height_agl', 'elevation_angle', 'max_distance', 'num_rays']

    # Check paths
    for path_key in required_paths:
        if path_key not in config.get('paths', {}):
            raise ValueError(f"Missing required path: {path_key}")

    # Check parameters
    for param_key in required_params:
        if param_key not in config.get('params', {}):
            raise ValueError(f"Missing required parameter: {param_key}")

    # Validate parameter ranges
    params = config['params']

    if not 0 < params['drone_height_agl'] < 1000:
        raise ValueError("drone_height_agl must be between 0 and 1000 meters")

    if not -90 <= params['elevation_angle'] <= 90:
        raise ValueError("elevation_angle must be between -90 and 90 degrees")

    if not 100 <= params['max_distance'] <= 50000:
        raise ValueError("max_distance must be between 100 and 50000 meters")

    if not 4 <= params['num_rays'] <= 3600:
        raise ValueError("num_rays must be between 4 and 3600")

    return True


def summarize_results(results: List[Dict]) -> Dict:
    """
    Generate summary statistics from analysis results.

    Args:
        results: List of analysis results

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}

    total_area = sum(r['visibility_area_ha'] for r in results)
    areas = [r['visibility_area_ha'] for r in results]

    # Cluster statistics
    cluster_stats = {}
    for result in results:
        cluster = result.get('cluster', 'Unknown')
        if cluster not in cluster_stats:
            cluster_stats[cluster] = {
                'count': 0,
                'total_area': 0,
                'areas': []
            }
        cluster_stats[cluster]['count'] += 1
        cluster_stats[cluster]['total_area'] += result['visibility_area_ha']
        cluster_stats[cluster]['areas'].append(result['visibility_area_ha'])

    # Calculate averages
    for cluster, stats in cluster_stats.items():
        stats['average_area'] = stats['total_area'] / stats['count']
        stats['min_area'] = min(stats['areas'])
        stats['max_area'] = max(stats['areas'])

    return {
        'total_staging_points': len(results),
        'total_visibility_area_ha': total_area,
        'average_area_ha': total_area / len(results),
        'min_area_ha': min(areas),
        'max_area_ha': max(areas),
        'cluster_statistics': cluster_stats
    }