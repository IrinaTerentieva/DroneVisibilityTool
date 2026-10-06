# waypoint_planner — M4E flight-time estimator

Secondary subproject. Estimates mapping-mission time for a polygon (KML/GPKG) over a grid of
camera zoom / GSD / front+side overlap.

Run `run_flight_time.py` in PyCharm (no CLI args). All inputs are in `config/config.yaml`
(areas, sweep lists, camera specs, speeds, turn/battery assumptions). Results go to
`outputs/<DDMonYYYY>_<HH-MM-SS>/`: `flight_time_estimates.csv` and `waypoints.gpkg` (layers `lines`, `photos`).

Model: footprint from 35mm-equiv focal length; line spacing = width*(1-side); photo spacing = height*(1-front);
speed = min(aircraft max, photo interval limit, motion-blur limit); lines at the angle giving fewest passes;
time = legs + transits + turns + overhead. Flat terrain, constant AGL. Camera specs are assumptions — verify them.

## Mission builder (DJI Pilot 2 KMZ)

Run `build_mission.py` in PyCharm; edit `config/mission.yaml` (camera, AGL, overlaps, `leg_min`, `heading_mode`, `export_dir`).
Each flight line = start + end waypoint; photos fire every `photo spacing` m (`multipleDistance`). Heading `followWayline`
points the nose along the flight direction. One KMZ per ~`leg_min` minutes (battery). Also writes `*_boundary.kml/gpkg`,
`*_preview.kml`, `*_summary.txt`, and (if `export_dir` is set) copies them plus a scripts snapshot there (e.g. the SD card).
Format follows your flown 8 Sep mission (WPML 1.0.6, aircraft 99/0, payload 88/0). Terrain is flat (AGL relative to take-off).
`min_photo_interval_s` (sustained shot interval, tele = 2.0 s assumed) sets the speed at high front overlap: measure it in Pilot 2.
