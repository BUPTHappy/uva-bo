import os
import sys
import subprocess
import time
import multiprocessing as mp
import click


def _is_zarr_extracted(zarr_dir: str) -> bool:
    """
    True if zarr_dir looks like a successfully extracted zarr store.
    We purposely check for common zarr markers to avoid skipping empty folders
    that were created before a failed extraction.
    """
    if not os.path.isdir(zarr_dir):
        return False
    # Zarr v2 marker files or v3-style structure.
    markers = [
        os.path.join(zarr_dir, ".zgroup"),
        os.path.join(zarr_dir, ".zattrs"),
        os.path.join(zarr_dir, "zarr.json"),
        os.path.join(zarr_dir, "meta"),
        os.path.join(zarr_dir, "data"),
    ]
    if any(os.path.exists(p) for p in markers):
        return True
    # Also treat any non-empty directory as extracted.
    try:
        return any(os.scandir(zarr_dir))
    except OSError:
        return False


def extract_data(dataset_name: str, data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    zarr_dir = f"{output_dir}/{dataset_name}.zarr"
    if _is_zarr_extracted(zarr_dir):
        print(f"Skipping {dataset_name} because it already exists in {output_dir}")
        return
    # If an empty directory exists (from a previous failed extraction), remove it.
    if os.path.isdir(zarr_dir):
        try:
            if not any(os.scandir(zarr_dir)):
                print(f"Removing empty directory {zarr_dir} from previous failure")
                subprocess.run([f"rm -rf {zarr_dir}"], shell=True, check=True)
        except OSError:
            pass
    print(
        f"Decompressing {data_dir}/{dataset_name}.zarr.tar.lz4 to {output_dir}/{dataset_name}.zarr"
    )
    os.makedirs(zarr_dir, exist_ok=True)
    subprocess.run(
        [
            f"lz4 -d -c {data_dir}/{dataset_name}.zarr.tar.lz4 | tar xf - -C {output_dir}"
        ],
        cwd=output_dir,
        shell=True,
        check=True,
    )


def compress_data(dataset_name: str, data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(f"{output_dir}/{dataset_name}.zarr.tar.lz4"):
        print(f"Skipping {dataset_name} because it already exists in {output_dir}")
        return
    print(
        f"Compressing {data_dir}/{dataset_name}.zarr to {output_dir}/{dataset_name}.zarr.tar.lz4"
    )
    subprocess.run(
        [
            f"tar cf - {dataset_name}.zarr | lz4 -c > {output_dir}/{dataset_name}.zarr.tar.lz4"
        ],
        cwd=data_dir,
        shell=True,
        check=True,
    )


def clean_all_data(output_dir: str):
    subprocess.run(
        [f"rm -rf {output_dir}/*.zarr"], cwd=output_dir, shell=True, check=True
    )


@click.command()
@click.argument("datasets", type=str, required=True)
@click.option("--data_dir", type=str, default="uva/umi_data/lz4")
@click.option("--output_dir", type=str, default="uva/umi_data/zarr")
def main(data_dir: str, output_dir: str, datasets: str):
    datasets_list = datasets.split(",")
    num_processes = mp.cpu_count()
    with mp.Pool(num_processes) as pool:
        pool.starmap(
            extract_data,
            [(dataset_name, data_dir, output_dir) for dataset_name in datasets_list],
        )


if __name__ == "__main__":
    main()
