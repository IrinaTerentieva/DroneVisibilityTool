#!/usr/bin/env python3
"""
Main script for drone angle visibility analysis using Hydra configuration.
This version calculates angles to drones at specific heights rather than using fixed angles.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
from pathlib import Path

from src.drone_angle_analyzer import DroneAngleVisibilityAnalyzer
from src.utils import load_staging_points_from_gpkg, validate_config, summarize_results
from src.visualization import plot_visibility_results, create_summary_plot

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for drone angle visibility analysis.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    logger.info("Drone Angle Visibility Analysis")
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    try:
        # Extract configuration
        paths = cfg.paths
        params = cfg.params
        advanced = cfg.get('advanced', {})

        # Load staging points
        logger.info(f"\nLoading staging points from: {paths.staging_gpkg}")
        staging_points = load_staging_points_from_gpkg(paths.staging_gpkg)

        if not staging_points:
            logger.error("No staging points found!")
            sys.exit(1)

        logger.info(f"Loaded {len(staging_points)} staging points")

        # Initialize analyzer
        logger.info(f"\nInitializing analyzer with DSM: {paths.dsm_path}")
        analyzer = DroneAngleVisibilityAnalyzer(paths.dsm_path)

        # Run analysis
        logger.info(f"\nStarting drone visibility analysis...")
        logger.info(f"Parameters:")
        logger.info(f"  - Drone distance: {params.drone_distance}m")
        logger.info(f"  - Drone height AGL: {params.drone_height_agl}m")
        logger.info(f"  - Number of rays: {params.num_rays}")
        logger.info(f"  - Observer height: {params.observer_height}m")
        logger.info(f"  - Terrain buffer for drone: {advanced.get('drone_terrain_buffer', 5.0)}m")

        results = analyzer.analyze_multiple_staging_areas(
            staging_points,
            drone_distance=params.drone_distance,
            drone_height_agl=params.drone_height_agl,
            num_rays=params.num_rays,
            observer_height=params.observer_height,
            min_depth_pixels=advanced.get('min_obstruction_depth', 3)
        )

        if not results:
            logger.error("No analysis results generated!")
            sys.exit(1)

        # Export results
        output_path = Path(paths.output_gpkg)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nExporting results to: {paths.output_gpkg}")
        analyzer.export_to_gpkg(results, str(output_path))

        # Generate summary
        logger.info("\nAnalysis Summary:")
        logger.info(f"Total staging points analyzed: {len(results)}")

        # Calculate summary statistics
        total_area = sum(r['visibility_area_ha'] for r in results)
        clear_percentages = [r['percent_rays_clear'] for r in results]
        angles = [r['mean_angle_to_drone'] for r in results]

        logger.info(f"Total visibility area: {total_area:.1f} ha")
        logger.info(f"Average area per point: {total_area / len(results):.1f} ha")
        logger.info(f"Average angle to drone: {np.mean(angles):.1f}°")
        logger.info(f"Angle range: {min(angles):.1f}° to {max(angles):.1f}°")
        logger.info(f"Average clear visibility: {np.mean(clear_percentages):.1f}%")

        # Cluster summary if available
        clusters = {}
        for result in results:
            cluster = result.get('cluster', 'Unknown')
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(result)

        if len(clusters) > 1 or list(clusters.keys())[0] != 'Unknown':
            logger.info("\nCluster Statistics:")
            for cluster, cluster_results in clusters.items():
                cluster_area = sum(r['visibility_area_ha'] for r in cluster_results)
                cluster_clear = np.mean([r['percent_rays_clear'] for r in cluster_results])
                logger.info(f"  Cluster {cluster}:")
                logger.info(f"    - Points: {len(cluster_results)}")
                logger.info(f"    - Total area: {cluster_area:.1f} ha")
                logger.info(f"    - Average clear: {cluster_clear:.1f}%")

        # Create visualizations if requested
        if cfg.analysis.export_visualization:
            logger.info("\nCreating visualizations...")

            # Update results to include angle in title
            for r in results:
                r['analysis_type'] = f"Drone at {r['drone_distance']}m, {r['drone_height_agl']}m AGL"

            # Main visibility plot
            vis_path = output_path.parent / f"drone_visibility_{params.drone_distance}m_{params.drone_height_agl}m.png"
            plot_visibility_results(
                results,
                color_by_cluster=cfg.analysis.color_by_cluster,
                save_path=str(vis_path),
                title_suffix=f"Drone at {params.drone_distance}m, {params.drone_height_agl}m AGL"
            )

        logger.info("\n✅ Analysis complete!")
        logger.info(f"Results saved to: {output_path}")

        # Print example angles for reference
        logger.info("\nExample viewing angles:")
        for i, result in enumerate(results[:5]):  # First 5 results
            logger.info(f"  Staging {result['staging_id']}: "
                        f"{result['mean_angle_to_drone']:.1f}° "
                        f"(range: {result['min_angle_to_drone']:.1f}° to {result['max_angle_to_drone']:.1f}°)")

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import numpy as np  # Add this import

    main()