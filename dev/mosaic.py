import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob
import os
from pathlib import Path
import numpy as np


def mosaic_asc_to_tif(input_folder: str, output_tif: str = None):
    """
    Mosaic all .asc files in a folder into a single .tif file

    Args:
        input_folder: Path to folder containing .asc files
        output_tif: Output .tif file path (optional, will auto-generate if None)
    """
    input_path = Path(input_folder)

    # Find all .asc files
    asc_files = list(input_path.glob("*.asc"))

    if not asc_files:
        print(f"No .asc files found in {input_folder}")
        return

    print(f"Found {len(asc_files)} .asc files:")
    for i, file in enumerate(asc_files[:10]):  # Show first 10
        print(f"  {i + 1}: {file.name}")
    if len(asc_files) > 10:
        print(f"  ... and {len(asc_files) - 10} more files")

    # Generate output filename if not provided
    if output_tif is None:
        folder_name = input_path.name
        output_tif = input_path / f"{folder_name}_mosaic.tif"

    try:
        # Open all raster files
        src_files_to_mosaic = []

        print("\nOpening raster files...")
        for i, file in enumerate(asc_files):
            try:
                src = rasterio.open(file)
                src_files_to_mosaic.append(src)

                # Print info for first file
                if i == 0:
                    print(f"\nFirst file info ({file.name}):")
                    print(f"  Shape: {src.shape}")
                    print(f"  CRS: {src.crs}")
                    print(f"  Transform: {src.transform}")
                    print(f"  Data type: {src.dtypes[0]}")
                    print(f"  NoData: {src.nodata}")

            except Exception as e:
                print(f"Error opening {file}: {e}")
                continue

        if not src_files_to_mosaic:
            print("No valid raster files to mosaic")
            return

        print(f"\nSuccessfully opened {len(src_files_to_mosaic)} files")

        # Perform the mosaic
        print("Creating mosaic...")
        mosaic, out_trans = merge(src_files_to_mosaic)

        # Get metadata from first file
        out_meta = src_files_to_mosaic[0].meta.copy()

        # Update metadata for mosaic
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"  # Add compression to reduce file size
        })

        print(f"Mosaic shape: {mosaic.shape}")
        print(f"Output file: {output_tif}")

        # Write the mosaic
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(mosaic)

        # Close all source files
        for src in src_files_to_mosaic:
            src.close()

        print(f"\n✅ Mosaic created successfully!")
        print(f"Output file: {output_tif}")

        # Check output file
        with rasterio.open(output_tif) as check:
            print(f"\nOutput file verification:")
            print(f"  Shape: {check.shape}")
            print(f"  CRS: {check.crs}")
            print(f"  Data type: {check.dtypes[0]}")
            print(f"  File size: {os.path.getsize(output_tif) / (1024 * 1024):.1f} MB")

            # Check for valid data
            data = check.read(1)
            valid_data = data[data != check.nodata] if check.nodata is not None else data
            if len(valid_data) > 0:
                print(f"  Elevation range: {valid_data.min():.2f} - {valid_data.max():.2f}")

        return str(output_tif)

    except Exception as e:
        print(f"Error during mosaicking: {e}")
        # Clean up opened files
        for src in src_files_to_mosaic:
            try:
                src.close()
            except:
                pass
        raise


def main():
    """Main function to run the mosaicking"""

    # Input folder containing .asc files
    input_folder = "/media/irina/My Book1/Petronas/test"

    # Output file (will be created in the same folder)
    output_file = "/media/irina/My Book1/Petronas/test/petronas_test_mosaic.tif"

    print("=" * 60)
    print("DSM MOSAIC TOOL")
    print("=" * 60)
    print(f"Input folder: {input_folder}")
    print(f"Output file: {output_file}")

    try:
        result = mosaic_asc_to_tif(input_folder, output_file)
        print(f"\n🎉 Mosaicking complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()