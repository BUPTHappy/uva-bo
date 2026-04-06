import os
import shlex
import subprocess
import multiprocessing as mp
import click


def extract_data(dataset_name: str, data_dir: str, output_dir: str):
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    zarr_path = os.path.join(output_dir, f"{dataset_name}.zarr")
    if os.path.isdir(zarr_path) and os.listdir(zarr_path):
        print(f"Skipping {dataset_name} because it already exists in {output_dir}")
        return
    if os.path.isdir(zarr_path) and not os.listdir(zarr_path):
        os.rmdir(zarr_path)

    lz4_path = os.path.join(data_dir, f"{dataset_name}.zarr.tar.lz4")
    if not os.path.isfile(lz4_path):
        raise FileNotFoundError(
            f"Missing compressed dataset: {lz4_path}\n"
            "  From the repository root, run first:\n"
            f"    python process_dataset/download_dataset.py\n"
            f"  Or put {dataset_name}.zarr.tar.lz4 under --data_dir and retry.\n"
            "  You can use an absolute path, e.g. --data_dir /full/path/to/lz4"
        )

    print(f"Decompressing {lz4_path} -> {zarr_path}")
    # Do not set cwd: relative paths in the pipeline are resolved from cwd; previously
    # cwd=output_dir broke paths like uva/umi_data/lz4/... (resolved under zarr/).
    cmd = "lz4 -d -c {} | tar xf - -C {}".format(
        shlex.quote(lz4_path),
        shlex.quote(output_dir),
    )
    subprocess.run(cmd, shell=True, check=True)


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
    datasets_list = [d.strip() for d in datasets.split(",") if d.strip()]
    num_processes = min(mp.cpu_count(), len(datasets_list)) or 1
    with mp.Pool(num_processes) as pool:
        pool.starmap(
            extract_data,
            [(dataset_name, data_dir, output_dir) for dataset_name in datasets_list],
        )


if __name__ == "__main__":
    main()
