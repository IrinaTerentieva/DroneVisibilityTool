"""Load polygons from KML/GPKG as 2D shapes in a local UTM CRS."""
from pathlib import Path

import geopandas as gpd
from shapely import force_2d


def load_polygons(area_cfg):
    path = Path(area_cfg.path)
    gdf = gpd.read_file(path, layer=area_cfg.layer) if area_cfg.layer else gpd.read_file(path)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    if gdf.empty:
        raise ValueError(f"No polygons in {path}")
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)
    gdf["geometry"] = gdf.geometry.apply(force_2d)
    for i, row in gdf.iterrows():
        name = str(row[area_cfg.name_field]) if area_cfg.name_field else f"{path.stem}_{i}"
        yield name, row.geometry, utm
