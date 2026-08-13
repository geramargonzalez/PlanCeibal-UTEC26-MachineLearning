import csv
from pathlib import Path

import pandas as pd


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


def standardize_column_name(column_name):
    return str(column_name).strip().lower().replace(" ", "_")


def get_standardized_columns(csv_file: Path):
    delimiter = detect_delimiter(csv_file)

    with csv_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter=delimiter)
        header = next(reader, None)

    if header is None:
        return []

    standardized = [standardize_column_name(column) for column in header]
    return standardized


def load_csv_as_dataframe(csv_file: Path):
    delimiter = detect_delimiter(csv_file)
    df = pd.read_csv(csv_file, sep=delimiter, encoding="utf-8-sig")
    df.columns = [standardize_column_name(col) for col in df.columns]
    return df


def create_combined_dataset(output_name: str = "datos_estudiantes_total.csv"):
    csv_files = get_csv_files()
    if not csv_files:
        return None

    dataframes = [load_csv_as_dataframe(csv_file) for csv_file in csv_files]
    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.drop_duplicates()

    output_path = Path(__file__).resolve().parent.parent / "DataSets" / output_name
    combined.to_csv(output_path, index=False)

    print(f"Combined dataset saved to: {output_path}")
    print(f"Rows: {len(combined)}")
    print(f"Columns: {list(combined.columns)}")
    return combined


def clean_combined_dataset(df: pd.DataFrame):
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].astype(str).str.strip().str.lower()
            cleaned[column] = cleaned[column].replace(
                {
                    "": pd.NA,
                    "na": pd.NA,
                    "n/a": pd.NA,
                    "nan": pd.NA,
                    "null": pd.NA,
                    "none": pd.NA,
                    "sin dato": pd.NA,
                    "sin_dato": pd.NA,
                    "sin-dato": pd.NA,
                    "unknown": pd.NA,
                    "desconocido": pd.NA,
                }
            )

    if "id_persona" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["id_persona"])

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            numeric_values = pd.to_numeric(cleaned[column], errors="coerce")
            valid_ratio = numeric_values.notna().sum() / max(cleaned[column].notna().sum(), 1)
            if valid_ratio > 0.8:
                cleaned[column] = numeric_values

    numeric_columns = [
        column
        for column in cleaned.columns
        if pd.api.types.is_numeric_dtype(cleaned[column])
        and column not in {"id_persona", "año_lectivo"}
    ]

    for column in numeric_columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[column] = cleaned[column].fillna("unknown")

    for column in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            if cleaned[column].dropna().mod(1).eq(0).all():
                cleaned[column] = cleaned[column].astype("Int64")

    return cleaned


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
    print("Columns:", get_standardized_columns(csv_file))
    for row in rows[: rows_to_show]:
        print(row)




def main():
    csv_files = get_csv_files()

    for csv_file in csv_files:
        print_first_rows(csv_file)

    combined = create_combined_dataset()
    if combined is not None:
        cleaned = clean_combined_dataset(combined)
        output_path = Path(__file__).resolve().parent.parent / "DataSets" / "datos_estudiantes_total_clean.csv"
        cleaned.to_csv(output_path, index=False)
        print(f"Clean dataset saved to: {output_path}")
        print(f"Cleaned rows: {len(cleaned)}")
        print(f"Cleaned columns: {list(cleaned.columns)}")


if __name__ == "__main__":
    main()
