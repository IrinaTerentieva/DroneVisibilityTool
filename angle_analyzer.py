#!/usr/bin/env python3
"""
Main script for drone visibility analysis using Hydra configuration.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
from pathlib import Path

from src import (
    SpatialDroneVisibilityAnalyzer,
    load_staging_points_from_gpkg,
    plot_visibility_results
)
from src.utils import validate_config, summarize_results
from src.visualization import create_summary_plot

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for drone visibility analysis.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    print('Config: ', cfg)

    try:
        # Validate configuration
        validate_config(cfg)

        # Extract paths and parameters
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
        analyzer = SpatialDroneVisibilityAnalyzer(
            paths.dsm_path,
            drone_height_agl=params.drone_height_agl
        )

        # Run analysis
        logger.info(f"\nStarting visibility analysis...")
        logger.info(f"Parameters:")
        logger.info(f"  - Elevation angle: {params.elevation_angle}°")
        logger.info(f"  - Max distance: {params.max_distance}m")
        logger.info(f"  - Number of rays: {params.num_rays}")
        logger.info(f"  - Observer height: {advanced.get('observer_height', 1.7)}m")

        results = analyzer.analyze_multiple_staging_areas(
            staging_points,
            elevation_angle=params.elevation_angle,
            max_distance=params.max_distance,
            num_rays=params.num_rays,
            min_depth_pixels=advanced.get('min_obstruction_depth', 3),
            observer_height=advanced.get('observer_height', 1.7)
        )

        if not results:
            logger.error("No analysis results generated!")
            sys.exit(1)

        # Export results
        logger.info(f"\nExporting results to: {paths.output_gpkg}")
        output_path = analyzer.export_to_gpkg(results, paths.output_gpkg)

        # Generate summary
        logger.info("\nAnalysis Summary:")
        summary = summarize_results(results)

        logger.info(f"Total staging points analyzed: {summary['total_staging_points']}")
        logger.info(f"Total visibility area: {summary['total_visibility_area_ha']:.1f} ha")
        logger.info(f"Average area per point: {summary['average_area_ha']:.1f} ha")
        logger.info(f"Area range: {summary['min_area_ha']:.1f} - {summary['max_area_ha']:.1f} ha")

        # Cluster summary
        if 'cluster_statistics' in summary:
            logger.info("\nCluster Statistics:")
            for cluster, stats in summary['cluster_statistics'].items():
                logger.info(f"  Cluster {cluster}:")
                logger.info(f"    - Points: {stats['count']}")
                logger.info(f"    - Total area: {stats['total_area']:.1f} ha")
                logger.info(f"    - Average area: {stats['average_area']:.1f} ha")

        # Create visualizations if requested
        if cfg.analysis.export_visualization:
            logger.info("\nCreating visualizations...")

            # Main visibility plot
            vis_path = Path(paths.output_gpkg).parent / "visibility_plot.png"
            plot_visibility_results(
                results,
                color_by_cluster=cfg.analysis.color_by_cluster,
                save_path=str(vis_path)
            )

            # Summary plot
            summary_path = Path(paths.output_gpkg).parent / "summary_plot.png"
            # create_summary_plot(summary, save_path=str(summary_path))

        logger.info("\n✅ Analysis complete!")
        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()