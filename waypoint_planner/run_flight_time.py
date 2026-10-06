"""Estimate M4E mapping flight time per polygon for a grid of GSD / overlap / camera zoom.

Run from PyCharm (right-click -> Run). Edit config/config.yaml to change inputs.
Results: printed table + CSV (+ waypoint gpkg) in outputs/<date>_<time>/.
"""
import logging
from pathlib import Path

import geopandas as gpd
import hydra
import itertools
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.areas import load_polygons
from src.flight_model import evaluate, photo_points

log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("\n" + OmegaConf.to_yaml(cfg))
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    rows, wp_layers = [], []

    for area in cfg.areas:
        for name, poly, crs in load_polygons(area):
            # altitude_m (AGL, flat terrain) takes precedence over gsd_cm when set
            alts = cfg.sweep.get("altitude_m")
            levels = [(None, a) for a in alts] if alts else [(g, None) for g in cfg.sweep.gsd_cm]
            grid = itertools.product(cfg.sweep.camera, levels,
                                     cfg.sweep.front_overlap, cfg.sweep.side_overlap)
            for cam_key, (gsd, alt), front, side in grid:
                cam = cfg.cameras[cam_key]
                fl = OmegaConf.merge(cfg.flight, cam.get("flight_override", {}))  # per-aircraft values
                m, lines, (segs, pspace) = evaluate(poly, cam, fl, cfg.path, gsd, front, side, alt)
                gsd = m["gsd_cm"]
                rows.append({"area": name, "camera": cam_key, **m})
                if cfg.output.export_waypoints:
                    tag = f"{name}_{cam_key}_gsd{gsd}_f{front}_s{side}".replace(".", "p")
                    for i, s in enumerate(segs):
                        wp_layers.append(("lines", tag, crs, {"scenario": tag, "leg": i}, s))
                    for i, p in photo_points(segs, pspace):
                        wp_layers.append(("photos", tag, crs, {"scenario": tag, "leg": i}, p))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    log.info("\n" + df.to_string(index=False))

    if cfg.output.save_csv:
        csv = out_dir / "flight_time_estimates.csv"
        df.to_csv(csv, index=False)
        log.info(f"CSV: {csv}")
    if wp_layers:
        gpkg = out_dir / "waypoints.gpkg"
        for layer in ("lines", "photos"):
            items = [x for x in wp_layers if x[0] == layer]
            g = gpd.GeoDataFrame([x[3] for x in items], geometry=[x[4] for x in items], crs=items[0][2])
            g.to_file(gpkg, layer=layer, driver="GPKG")
        log.info(f"Waypoints: {gpkg}")


if __name__ == "__main__":
    main()
