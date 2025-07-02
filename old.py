import hydra
from omegaconf import DictConfig
from visibility_tool import SpatialDroneVisibilityAnalyzer

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Load staging points
    staging_points = SpatialDroneVisibilityAnalyzer.load_staging_points_from_gpkg(cfg.paths.staging_gpkg)

    # Init analyzer
    analyzer = SpatialDroneVisibilityAnalyzer(cfg.paths.dsm_path, drone_height_agl=cfg.params.drone_height_agl)

    # Analyze
    results = analyzer.analyze_multiple_staging_areas_spatial(
        staging_points,
        elevation_angle=cfg.params.elevation_angle,
        max_distance=cfg.params.max_distance,
        num_rays=cfg.params.num_rays
    )

    # Export
    output_file = analyzer.export_visibility_to_gpkg(results, cfg.paths.output_gpkg)

    # Optionally plot
    analyzer.plot_visibility_spatial(results, color_by_cluster=True)

if __name__ == "__main__":
    main()
