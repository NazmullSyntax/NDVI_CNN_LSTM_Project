"""Utilities for preparing NDVI sequences for modeling."""

from pathlib import Path
from typing import Tuple

"""
data_preprocessing.py
=====================
Data loading, validation, clipping, resizing, normalization, and sequence
creation for the Dhaka City CNN-LSTM NDVI prediction project.
"""

import warnings
from pathlib import Path

import numpy as np
import yaml

try:
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box
except ImportError:
    gpd = None
    rasterio = None
    rio_mask = None
    box = None

warnings.filterwarnings("ignore")


def _require_geospatial() -> None:
    """Raise a focused error when a raster or vector operation needs extras."""
    if gpd is None or rasterio is None or rio_mask is None or box is None:
        raise ImportError(
            "Geospatial features require rasterio, geopandas, and shapely. "
            "Install them with: pip install -r requirements.txt"
        )


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found at {path}")
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def validate_raster_consistency(raster_paths: list) -> dict:
    """Validate CRS, transform, dimensions, band count, and data type."""
    _require_geospatial()
    if len(raster_paths) < 2:
        raise ValueError("Need at least 2 rasters to validate consistency.")

    reference = None
    reference_path = None
    for path in raster_paths:
        with rasterio.open(path) as source:
            metadata = {
                "crs": str(source.crs),
                "transform": tuple(source.transform),
                "width": source.width,
                "height": source.height,
                "count": source.count,
                "dtype": str(source.dtypes[0]),
            }
        if reference is None:
            reference = metadata
            reference_path = path
            continue
        if metadata["crs"] != reference["crs"]:
            raise ValueError(f"CRS mismatch between {reference_path} and {path}")
        if (metadata["width"], metadata["height"]) != (
            reference["width"],
            reference["height"],
        ):
            raise ValueError(f"Dimension mismatch between {reference_path} and {path}")
        if not np.allclose(metadata["transform"], reference["transform"], atol=1e-6):
            raise ValueError(f"Geotransform mismatch between {reference_path} and {path}")
    return reference


def find_ndvi_files(data_dir: str, start_year: int, end_year: int) -> dict:
    """Find continuous yearly files named ndvi_YYYY.tif or .tiff."""
    directory = Path(data_dir)
    files = {}
    missing = []
    for year in range(start_year, end_year + 1):
        candidates = [
            directory / f"ndvi_{year}.tif",
            directory / f"ndvi_{year}.tiff",
            directory / f"NDVI_{year}.tif",
            directory / f"ndvi_{year}.TIF",
        ]
        match = next((candidate for candidate in candidates if candidate.exists()), None)
        if match is None:
            missing.append(year)
        else:
            files[year] = match
    if missing:
        raise FileNotFoundError(f"Missing NDVI years: {missing}")
    return files


def load_boundary(boundary_path: str):
    """Load a non-empty boundary file with a defined CRS."""
    _require_geospatial()
    path = Path(boundary_path)
    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found at {path}")
    boundary = gpd.read_file(path)
    if boundary.empty:
        raise ValueError("Boundary file contains no geometries.")
    if boundary.crs is None:
        raise ValueError("Boundary file has no CRS defined.")
    return boundary


def clip_raster_to_boundary(raster_path: Path, boundary_gdf, nodata=-9999):
    """Clip a raster to a boundary and return a float array plus its profile."""
    _require_geospatial()
    with rasterio.open(raster_path) as source:
        boundary = boundary_gdf.to_crs(source.crs)
        geometries = [geometry.__geo_interface__ for geometry in boundary.geometry]
        try:
            clipped, transform = rio_mask(source, geometries, crop=True, nodata=nodata)
        except Exception as error:
            raise ValueError(f"Failed to clip {raster_path.name} to boundary: {error}") from error
        profile = source.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=transform,
            nodata=nodata,
        )
    array = clipped[0].astype(np.float32)
    array[array == nodata] = np.nan
    return array, profile


def resize_ndvi(array: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2D NDVI array while preserving invalid-value locations."""
    from PIL import Image

    invalid = np.isnan(array)
    filled = np.where(invalid, 0.0, array)
    resized = np.asarray(
        Image.fromarray(filled.astype(np.float32), mode="F").resize(
            (size, size), resample=Image.BILINEAR
        ),
        dtype=np.float32,
    )
    valid_mask = np.asarray(
        Image.fromarray((~invalid).astype(np.float32), mode="F").resize(
            (size, size), resample=Image.NEAREST
        )
    ) > 0.5
    resized[~valid_mask] = np.nan
    return resized


def normalize_ndvi(array: np.ndarray, ndvi_min=-1.0, ndvi_max=1.0) -> np.ndarray:
    """Scale NDVI from the configured range to [0, 1]."""
    clipped = np.clip(array, ndvi_min, ndvi_max)
    return (clipped - ndvi_min) / (ndvi_max - ndvi_min)


def denormalize_ndvi(array: np.ndarray, ndvi_min=-1.0, ndvi_max=1.0) -> np.ndarray:
    """Convert normalized NDVI values back to the configured range."""
    return array * (ndvi_max - ndvi_min) + ndvi_min


def build_stacked_dataset(
    ndvi_files: dict,
    boundary_gdf,
    image_size: int,
    ndvi_min: float,
    ndvi_max: float,
    output_npz: Path = None,
):
    """Build a normalized array shaped (years, image_size, image_size, 1)."""
    years = sorted(ndvi_files)
    if not years:
        raise ValueError("No NDVI files were provided.")
    stacked = []
    nan_fractions = []
    for year in years:
        array, _ = clip_raster_to_boundary(ndvi_files[year], boundary_gdf)
        nan_fractions.append(float(np.isnan(array).mean()))
        resized = np.nan_to_num(resize_ndvi(array, image_size), nan=0.0)
        stacked.append(normalize_ndvi(resized, ndvi_min, ndvi_max))
    stacked_array = np.stack(stacked, axis=0)[..., np.newaxis].astype(np.float32)
    if output_npz is not None:
        destination = Path(output_npz)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            years=np.array(years),
            data=stacked_array,
            nan_fractions=np.array(nan_fractions),
        )
    return years, stacked_array, nan_fractions


def generate_demo_ndvi(
    start_year: int,
    end_year: int,
    size: int = 128,
    output_dir: str = "data/demo",
    seed: int = 42,
):
    """Generate synthetic yearly GeoTIFF rasters for pipeline testing."""
    _require_geospatial()
    rng = np.random.default_rng(seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size]
    center = size / 2
    distance = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / center
    base_vegetation = 0.15 + 0.55 * distance + 0.1 * np.sin(xx / 10) * np.cos(yy / 10)
    files = []
    for index, year in enumerate(range(start_year, end_year + 1)):
        ndvi = base_vegetation - 0.012 * index + 0.05 * np.sin(2 * np.pi * index / 5)
        ndvi += rng.normal(0, 0.03, size=(size, size))
        ndvi = np.clip(ndvi, -0.2, 0.9).astype(np.float32)
        river = np.abs(xx - (center + 8 * np.sin(yy / 15))) < 3
        ndvi[river] = rng.normal(-0.05, 0.05, size=river.sum())
        ndvi[rng.random((size, size)) < 0.05] = np.nan
        path = destination / f"ndvi_{year}.tif"
        profile = {
            "driver": "GTiff",
            "height": size,
            "width": size,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:32646",
            "transform": rasterio.transform.from_bounds(0, 0, size * 30, size * 30, size, size),
            "nodata": -9999.0,
        }
        with rasterio.open(path, "w", **profile) as destination_file:
            destination_file.write(np.nan_to_num(ndvi, nan=-9999.0), 1)
        files.append(path)
    return files


def generate_demo_boundary(output_path: str = "data/boundary/dhaka_boundary.geojson"):
    """Create a square demo boundary matching the synthetic raster CRS."""
    _require_geospatial()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = box(0, 0, 128 * 30, 128 * 30)
    boundary = gpd.GeoDataFrame({"name": ["Dhaka_demo"]}, geometry=[geometry], crs="EPSG:32646")
    boundary.to_file(path, driver="GeoJSON")
    return path


def create_sequences(stacked: np.ndarray, years: list, sequence_length: int):
    """Build X/y pairs where each target is the next yearly NDVI image."""
    if len(years) <= sequence_length:
        raise ValueError("Need more years than sequence_length to create sequences.")
    inputs = [stacked[index : index + sequence_length] for index in range(len(years) - sequence_length)]
    targets = [stacked[index + sequence_length] for index in range(len(years) - sequence_length)]
    target_years = years[sequence_length:]
    return np.stack(inputs).astype(np.float32), np.stack(targets).astype(np.float32), target_years


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess NDVI data for Dhaka City.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--boundary", default=None, help="Path to a boundary file.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.demo:
        generate_demo_ndvi(
            config["start_year"],
            config["end_year"],
            output_dir=config["paths"]["demo_data"],
            seed=config["random_seed"],
        )
        generate_demo_boundary(Path(config["paths"]["boundary"]) / "dhaka_boundary.geojson")
        return
    data_dir = config["paths"]["raw_data"]
    boundary_path = args.boundary or Path(config["paths"]["boundary"]) / "dhaka_boundary.shp"
    files = find_ndvi_files(data_dir, config["start_year"], config["end_year"])
    validate_raster_consistency(list(files.values()))
    boundary = load_boundary(boundary_path)
    build_stacked_dataset(
        files,
        boundary,
        config["image_size"],
        config["ndvi_min"],
        config["ndvi_max"],
        Path(config["paths"]["processed_data"]) / "ndvi_stacked.npz",
    )


if __name__ == "__main__":
    main()


def normalize_ndvi(ndvi: np.ndarray, lower: float = -1.0, upper: float = 1.0) -> np.ndarray:
    """Clip NDVI values to the physical range and scale to [0, 1]."""
    values = np.asarray(ndvi, dtype=np.float32)
    values = np.clip(values, lower, upper)
    return (values - lower) / (upper - lower)


def make_sequences(array: np.ndarray, sequence_length: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window inputs and future targets from time-first data."""
    values = np.asarray(array, dtype=np.float32)
    if values.ndim < 1 or sequence_length < 1 or horizon < 1:
        raise ValueError("array must be time-first and sequence_length/horizon must be positive")
    if len(values) < sequence_length + horizon:
        raise ValueError("array is shorter than sequence_length + horizon")
    inputs = []
    targets = []
    for start in range(len(values) - sequence_length - horizon + 1):
        inputs.append(values[start : start + sequence_length])
        targets.append(values[start + sequence_length : start + sequence_length + horizon])
    return np.stack(inputs), np.stack(targets)


def ensure_directory(path: str | Path) -> Path:
    """Create and return an output directory."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
