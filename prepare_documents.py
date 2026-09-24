import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. File locations
# --------------------------------------------------

DATA_FOLDER = Path("data")

csv_files = list(DATA_FOLDER.rglob("*.csv"))

if not csv_files:
    print("ERROR: No CSV file found anywhere inside the data folder.")
    raise SystemExit(1)

if len(csv_files) > 1:
    print("Multiple CSV files found:")
    for file in csv_files:
        print(f" - {file}")

    print("\nPlease keep only the dengue dataset CSV in the data folder.")
    raise SystemExit(1)

INPUT_FILE = csv_files[0]

print(f"Using dataset: {INPUT_FILE}")
OUTPUT_FOLDER = Path("processed")


# --------------------------------------------------
# 2. Create output folder
# --------------------------------------------------

OUTPUT_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# 3. Read the dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)


# --------------------------------------------------
# 4. Remove completely empty columns
# --------------------------------------------------

df = df.dropna(axis=1, how="all")


# --------------------------------------------------
# 5. Helper function
# --------------------------------------------------

def value_or_not_available(value):
    if pd.isna(value):
        return "Not available"
    return str(value)


# --------------------------------------------------
# 6. Create one document for every record
# --------------------------------------------------

for _, row in df.iterrows():

    record_number = int(row["SN"])

    document = f"""
DENGUE CLINICAL RECORD

Record Number: {record_number}

DEMOGRAPHICS
Age: {value_or_not_available(row["Age"])} years
Sex: {value_or_not_available(row["Sex"])}

DENGUE TESTING
Dengue NS1: {value_or_not_available(row["Dengue NS1"])}

LABORATORY RESULTS
Platelet Count (PLT): {value_or_not_available(row["PLT"])}
White Blood Cell Count (WBC): {value_or_not_available(row["WBC"])}
Hematocrit (HCT): {value_or_not_available(row["HCT"])}
Red Blood Cell Count (RBC): {value_or_not_available(row["RBC"])}
Lymphocytes: {value_or_not_available(row["Lymph %"])}%
Neutrophils: {value_or_not_available(row["Neut %"])}%
ALT: {value_or_not_available(row["ALT"])}
AST: {value_or_not_available(row["AST"])}

SOURCE
Comprehensive Dengue Hematology and Clinical Dataset from Bangladesh.

DATA USAGE NOTICE
This record comes from a publicly released research dataset.
It is not a live medical record and must not be used for diagnosis,
treatment, or medication decisions.
""".strip()


    # --------------------------------------------------
    # 7. Save document
    # --------------------------------------------------

    output_file = OUTPUT_FOLDER / f"dengue_record_{record_number:04d}.txt"

    output_file.write_text(
        document,
        encoding="utf-8"
    )


print(f"Created {len(df)} RAG documents.")
print(f"Output folder: {OUTPUT_FOLDER.resolve()}")