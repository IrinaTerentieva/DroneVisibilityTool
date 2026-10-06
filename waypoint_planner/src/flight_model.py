"""Geometry + time model for a lawn-mower photogrammetry mission over a polygon."""
import math
from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry

DIAG_35MM = 43.267
SENSOR_ASPECT_W = 0.8  # width / diagonal for 4:3


@dataclass
class Footprint:
    altitude_m: float
    width_m: float    # across-track
    height_m: float   # along-track
    gsd_m: float


def footprint(cam, gsd_cm=None, across: str = "width", altitude_m=None) -> Footprint:
    if cam.get("sensor_w_mm"):
        ratio = cam.sensor_w_mm / cam.focal_mm                      # sensor_w / f
    else:
        ratio = SENSOR_ASPECT_W * DIAG_35MM / cam.equiv_focal_mm
    if altitude_m is not None:
        alt = float(altitude_m)
        gsd = alt * ratio / cam.img_w_px
    else:
        gsd = gsd_cm / 100.0
        alt = gsd * cam.img_w_px / ratio
    w_img = gsd * cam.img_w_px
    h_img = gsd * cam.img_h_px
    if across == "width":
        return Footprint(alt, w_img, h_img, gsd)
    return Footprint(alt, h_img, w_img, gsd)


def _lines_at_angle(poly: BaseGeometry, angle: float, spacing: float, cross: float):
    """Parallel lines (rotated frame -> back) clipped to poly. Returns list of lists of segments."""
    c = poly.centroid
    rp = affinity.rotate(poly, -angle, origin=c)
    minx, miny, maxx, maxy = rp.bounds
    height = maxy - miny
    n = max(1, math.ceil(max(height - cross, 0) / spacing) + 1)
    used = (n - 1) * spacing
    y0 = miny + (height - used) / 2          # centre the line block on the polygon
    out = []
    for k in range(n):
        y = y0 + k * spacing
        inter = rp.intersection(LineString([(minx - 1, y), (maxx + 1, y)]))
        segs = []
        if not inter.is_empty:
            geoms = inter.geoms if hasattr(inter, "geoms") else [inter]
            for g in geoms:
                if isinstance(g, LineString) and g.length > 0:
                    segs.append(g)
        if k % 2:
            segs = [LineString(s.coords[::-1]) for s in segs[::-1]]
        out.append(segs)
    rot_back = lambda g: affinity.rotate(g, angle, origin=c)
    return [[rot_back(s) for s in segs] for segs in out]


def plan_lines(poly, spacing, cross, angle=None, step=5.0):
    angles = [angle] if angle is not None else list(np.arange(0, 180, step))
    best = None
    for a in angles:
        lines = _lines_at_angle(poly, a, spacing, cross)
        nlines = sum(1 for l in lines if l)
        length = sum(s.length for l in lines for s in l)
        key = (nlines, length)
        if best is None or key < best[0]:
            best = (key, a, lines)
    return best[1], best[2]


def speed_limits(cam, fp: Footprint, fl, front_overlap: float):
    photo_spacing = fp.height_m * (1 - front_overlap)
    v_photo = photo_spacing / cam.min_photo_interval_s
    v_blur = fl.max_blur_px * fp.gsd_m / cam.shutter_s
    v = min(fl.max_speed_ms, v_photo, v_blur)
    limiter = {fl.max_speed_ms: "aircraft", v_photo: "photo_interval", v_blur: "motion_blur"}[v]
    floor = fl.get("min_speed_ms") or 0
    if v < floor:
        v, limiter = floor, f"min_speed({limiter})"
    return v, photo_spacing, limiter


def evaluate(poly, cam, fl, pa, gsd_cm, front, side, altitude_m=None):
    """poly: shapely polygon in a metric CRS. Returns (metrics dict, lines, photo points)."""
    fp = footprint(cam, gsd_cm, fl.image_across_track, altitude_m)
    spacing = fp.width_m * (1 - side)
    v, photo_spacing, limiter = speed_limits(cam, fp, fl, front)
    work = poly.buffer(pa.edge_buffer_m) if pa.edge_buffer_m else poly
    angle, lines = plan_lines(work, spacing, fp.width_m, pa.angle_deg, pa.angle_step_deg)

    segs = [s for l in lines for s in l]
    n_lines = sum(1 for l in lines if l)
    line_len = sum(s.length for s in segs)
    transit = sum(
        Point(a.coords[-1]).distance(Point(b.coords[0])) for a, b in zip(segs[:-1], segs[1:])
    )
    n_photos = sum(math.floor(s.length / photo_spacing) + 1 for s in segs)

    t_lines = line_len / v
    if fl.stop_and_go:
        t_lines += n_photos * fl.stop_and_go_overhead_s
    t_transit = transit / fl.transit_speed_ms
    t_turns = max(len(segs) - 1, 0) * fl.turn_time_s
    t_total = t_lines + t_transit + t_turns + fl.overhead_s

    usable_s = fl.battery_flight_min * 60 * fl.battery_usable_frac
    flags = []
    if fp.altitude_m > fl.max_agl_m:
        flags.append("ALT>max")
    if fp.altitude_m < fl.min_agl_m:
        flags.append("ALT<min")

    m = dict(
        gsd_cm=round(fp.gsd_m * 100, 3), front=front, side=side,
        altitude_m=round(fp.altitude_m, 1),
        footprint_m=f"{fp.width_m:.0f}x{fp.height_m:.0f}",
        line_spacing_m=round(spacing, 1), photo_spacing_m=round(photo_spacing, 1),
        angle_deg=float(angle), n_lines=n_lines, n_photos=n_photos,
        speed_ms=round(v, 2), speed_limit=limiter,
        path_km=round((line_len + transit) / 1000, 2),
        time_min=round(t_total / 60, 1),
        batteries=round(t_total / usable_s, 2),
        data_gb=round(n_photos * cam.mb_per_photo / 1024, 2),
        area_ha=round(poly.area / 1e4, 2),
        flags=",".join(flags),
    )
    return m, lines, (segs, photo_spacing)


def photo_points(segs, photo_spacing):
    pts = []
    for i, s in enumerate(segs):
        n = math.floor(s.length / photo_spacing) + 1
        for k in range(n):
            pts.append((i, s.interpolate(k * photo_spacing)))
    return pts
