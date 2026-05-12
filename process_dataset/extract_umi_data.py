"""
Decompress *.zarr.tar.lz4 into *.zarr. Requires lz4 and tar.

Run from repo root after download_dataset.py produced uva/umi_data/lz4/*.zarr.tar.lz4.
See process_dataset/download_dataset.py module docstring for the full pipeline.
"""
import os
import sys
import subprocess
import time
import multiprocessing as mp
import click


def extract_data(dataset_name: str, data_dir: str, output_dir: str):
    # Must be absolute: subprocess cwd is output_dir, so relative data_dir would break.
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)
    archive = os.path.join(data_dir, f"{dataset_name}.zarr.tar.lz4")
    zarr_out = os.path.join(output_dir, f"{dataset_name}.zarr")

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(zarr_out):
        print(f"Skipping {dataset_name} because it already exists in {output_dir}")
        return
    if not os.path.isfile(archive):
        raise FileNotFoundError(
            f"Missing compressed archive:\n  {archive}\n"
            f"Run from the repo root first (so paths like uva/umi_data resolve), then:\n"
            f"  python process_dataset/download_dataset.py --data_dir uva/umi_data\n"
            f"That downloads .zarr.zip under uva/umi_data/zip/ and writes "
            f"uva/umi_data/lz4/<name>.zarr.tar.lz4.\n"
            f"Or pass --data_dir to this script pointing at the folder that contains "
            f"{{dataset_name}}.zarr.tar.lz4 files."
        )
    print(f"Decompressing {archive} to {zarr_out}")
    # Pipe lz4 -> tar (no shell, no cwd tricks). Relative paths in old versions broke
    # when cwd was set to output_dir: uva/... was resolved under zarr/.
    lz4_p = subprocess.Popen(
        ["lz4", "-d", "-c", archive],
        stdout=subprocess.PIPE,
    )
    assert lz4_p.stdout is not None
    try:
        subprocess.run(
            ["tar", "xf", "-", "-C", output_dir],
            stdin=lz4_p.stdout,
            check=True,
        )
    finally:
        lz4_p.stdout.close()
    lz4_rc = lz4_p.wait()
    if lz4_rc != 0:
        raise subprocess.CalledProcessError(
            lz4_rc, ["lz4", "-d", "-c", archive]
        )


def compress_data(dataset_name: str, data_dir: str, output_dir: str):
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_lz4 = os.path.join(output_dir, f"{dataset_name}.zarr.tar.lz4")
    if os.path.exists(out_lz4):
        print(f"Skipping {dataset_name} because it already exists in {output_dir}")
        return
    print(
        f"Compressing {os.path.join(data_dir, dataset_name)}.zarr to {out_lz4}"
    )
    subprocess.run(
        [f"tar cf - {dataset_name}.zarr | lz4 -c > {out_lz4}"],
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
