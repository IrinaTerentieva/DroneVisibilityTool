"""Build a DJI Pilot 2 waypoint mission (.kmz) + Google Earth preview (.kml) for a polygon.

Run from PyCharm. Edit config/mission.yaml. Each flight line is just a start + end waypoint; photos are
triggered automatically every `photo spacing` metres between them.
"""
import logging
import shutil
from pathlib import Path

import geopandas as gpd
import hydra
from omegaconf import DictConfig, OmegaConf
from shapely.geometry import LineString, Point

from src.areas import load_polygons
from src.dji_wpml import build_kmz, enums_from_reference
from src.flight_model import evaluate

log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


@hydra.main(version_base=None, config_path="config", config_name="mission")
def main(cfg: DictConfig) -> None:
    m, d = cfg.mission, cfg.dji
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    polys = [x for area in cfg.areas for x in load_polygons(area)]
    name, poly, crs = polys[m.area_index]
    cam = cfg.cameras[m.camera]
    fl = OmegaConf.merge(cfg.flight, cam.get("flight_override", {}), {"min_speed_ms": m.min_speed_ms})
    metrics, lines, (segs, photo_spacing) = evaluate(
        poly, cam, fl, cfg.path, None, m.front_overlap, m.side_overlap, m.altitude_m)

    speed = metrics["speed_ms"]
    if metrics["speed_limit"].startswith("min_speed"):
        log.warning(f"Blur-limited speed is below {m.min_speed_ms} m/s; using {speed} m/s (more motion blur)")

    # heading: fixed = line axis (bearing from north, -180..180); image orientation stays constant
    ang = metrics["angle_deg"]
    hd = (90 - ang) % 180
    p_heading = hd if hd <= 180 else hd - 360

    p = {k: v for k, v in OmegaConf.to_container(d, resolve=True).items() if k != "rc_lost"}
    p.update(exit_on_rc_lost=d.rc_lost.exit_on_rc_lost, rc_action=d.rc_lost.action,
             heading_deg=round(p_heading, 1))
    p["trigger_param"] = photo_spacing   # metres between photos (multipleDistance)
    if d.reference_kmz:
        p.update(enums_from_reference(d.reference_kmz))
        log.info(f"Enums copied from {d.reference_kmz}")

    # pack consecutive lines into legs of <= leg_min minutes of wayline (one KMZ per battery)
    seg_t = [s.length / speed + fl.turn_time_s for s in segs]
    legs, cur, t = [], [], 0.0
    for i, ts in enumerate(seg_t):
        if cur and m.leg_min and t + ts > m.leg_min * 60:
            legs.append(cur); cur, t = [], 0.0
        cur.append(i); t += ts
    legs.append(cur)

    to_ll = lambda c: gpd.GeoSeries([Point(c)], crs=crs).to_crs(4326).iloc[0]
    kmzs, all_wps = [], []
    for li, leg in enumerate(legs, 1):
        wps = []
        for i in leg:
            for c in (segs[i].coords[0], segs[i].coords[-1]):
                q = to_ll(c); wps.append((q.x, q.y))
        line_idx = [(2 * k, 2 * k + 1) for k in range(len(leg))]
        dist = sum(segs[i].length for i in leg)
        name_ = m.name if len(legs) == 1 else f"{m.name}_leg{li:02d}_of{len(legs):02d}"
        kmzs.append(build_kmz(out_dir / f"{name_}.kmz", wps, line_idx, m.altitude_m, speed, p, dist, dist / speed + len(leg) * fl.turn_time_s))
        all_wps += wps
    wps = all_wps

    # Google Earth preview (not a mission): lines + start/end points
    ll = gpd.GeoSeries([LineString([a, b]) for a, b in zip(wps[::2], wps[1::2])], crs=4326)
    prev = gpd.GeoDataFrame({"Name": [f"line_{i}" for i in range(len(ll))]}, geometry=ll)
    prev.to_file(out_dir / f"{m.name}_preview.kml", driver="KML")

    # boundary of the area flown (WGS84), for QGIS / Google Earth
    bnd = gpd.GeoDataFrame({"name": [name]}, geometry=[poly], crs=crs).to_crs(4326)
    bnd.to_file(out_dir / f"{m.name}_boundary.gpkg", layer="boundary", driver="GPKG")
    bnd.to_file(out_dir / f"{m.name}_boundary.kml", driver="KML")
    (out_dir / f"{m.name}_summary.txt").write_text(
        f"{m.name}\narea: {name} ({metrics['area_ha']} ha), source: {[str(a.path) for a in cfg.areas]}\n"
        f"camera {m.camera} @ {m.altitude_m} m AGL, GSD {metrics['gsd_cm']} cm, overlap front {m.front_overlap} / side {m.side_overlap}\n"
        f"footprint {metrics['footprint_m']} m, line spacing {metrics['line_spacing_m']} m, photo every {photo_spacing:.2f} m\n"
        f"{len(segs)} lines in {len(legs)} legs {[len(l) for l in legs]}, ~{metrics['n_photos']} photos\n"
        f"speed {speed} m/s ({metrics['speed_limit']}), estimated {metrics['time_min']} min = {metrics['batteries']} batteries\n"
        f"heading {p['heading_mode']}, photo trigger {p['photo_trigger']} every {photo_spacing:.2f} m, zoom {p['zoom_focal_length_mm']} mm\n"
        f"flat terrain: {m.altitude_m} m relative to take-off point; RTH {p['rth_height_m']} m\n")

    if m.export_dir:   # copy results + a snapshot of the code that made them
        dst = Path(m.export_dir); dst.mkdir(parents=True, exist_ok=True)
        for f in out_dir.iterdir():
            if f.is_file() and f.suffix in {".kmz", ".kml", ".gpkg", ".txt"}:
                shutil.copy2(f, dst / f.name)
        code = dst / "scripts"
        for rel in ("build_mission.py", "run_flight_time.py", "README.md", "src", "config"):
            src_ = HERE / rel
            if src_.is_dir():
                shutil.copytree(src_, code / rel, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                code.mkdir(exist_ok=True); shutil.copy2(src_, code / rel)
        OmegaConf.save(cfg, dst / f"{m.name}_resolved_config.yaml")
        log.info(f"Copied to {dst}")

    t = metrics["time_min"]
    kmz = "\n         ".join(map(str, kmzs))
    log.info(f"\nArea {name} ({metrics['area_ha']} ha) | {m.camera} @ {m.altitude_m} m AGL | GSD {metrics['gsd_cm']} cm\n"
             f"footprint {metrics['footprint_m']} m | line spacing {metrics['line_spacing_m']} m | photo every {photo_spacing:.2f} m\n"
             f"{len(segs)} lines in {len(legs)} leg(s) of {[len(l) for l in legs]}, {len(wps)} waypoints, ~{metrics['n_photos']} photos, speed {speed} m/s\n"
             f"estimated time {t} min = {metrics['batteries']} batteries\n"
             f"KMZ:     {kmz}\nPreview: {out_dir / (m.name + '_preview.kml')}")


if __name__ == "__main__":
    main()
