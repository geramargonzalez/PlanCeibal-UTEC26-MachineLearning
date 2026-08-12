import csv
from pathlib import Path


def get_csv_files():
    base_dir = Path(__file__).resolve().parent.parent
    datasets_dir = base_dir / "DataSets"
    csv_files = sorted(datasets_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {datasets_dir}")
        return []

    return csv_files


def detect_delimiter(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        return dialect.delimiter
    except csv.Error:
        return ";" if ";" in sample else ","


def print_first_rows(csv_file: Path, rows_to_show: int = 5):
    delimiter = detect_delimiter(csv_file)

    with csv_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        print(f"\n=== {csv_file.name} ===")
        print("File is empty.")
        return

    print(f"\n=== {csv_file.name} ===")
    for row in rows[: rows_to_show]:
        print(row)


def main():
    csv_files = get_csv_files()

    for csv_file in csv_files:
        print_first_rows(csv_file)


if __name__ == "__main__":
    main()
