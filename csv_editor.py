import csv
import os

csv_file = "ExtremeAlternative.csv"

# Read and filter rows
rows = []
with open(csv_file, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        count_str = row.get("number_of_repetitions", "")
        try:
            count_val = int(float(count_str)) if count_str != "" else 0
        except Exception:
            # keep row if count can't be parsed
            rows.append(row)
            continue
        if count_val <= 8:
            rows.append(row)

# Write filtered data back safely to the same file
tmp_file = csv_file + ".tmp"
with open(tmp_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

os.replace(tmp_file, csv_file)
