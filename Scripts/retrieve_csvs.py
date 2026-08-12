from pathlib import Path


def list_csv_files():
    base_dir = Path(__file__).resolve().parent.parent
    datasets_dir = base_dir / "DataSets"

    csv_files = sorted(datasets_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {datasets_dir}")
        return []

    for csv_file in csv_files:
        print(csv_file.name)

    return csv_files


if __name__ == "__main__":
    list_csv_files()
