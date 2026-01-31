import csv
import numpy as np


def load_csv(file_path):
    data = []

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        # Read header
        headers = next(reader, None)
        if headers is None:
            return np.empty((0, 0))

        column_count = len(headers)
        is_numeric = [True] * column_count

        first_data_row = None

        # Find first non-empty data row
        for row in reader:
            if all(field.strip() == "" for field in row):
                continue

            first_data_row = row
            for i in range(column_count):
                if i >= len(row) or row[i].strip() == "":
                    is_numeric[i] = False
                else:
                    try:
                        float(row[i])
                    except ValueError:
                        is_numeric[i] = False
            break

        if first_data_row is None:
            return np.empty((0, 0))

        # Process first data row
        numeric_row = []
        for i in range(column_count):
            if is_numeric[i]:
                try:
                    numeric_row.append(float(first_data_row[i]))
                except (ValueError, IndexError):
                    numeric_row.append(float("nan"))

        data.append(numeric_row)

        # Process remaining rows
        for row in reader:
            if all(field.strip() == "" for field in row):
                continue

            numeric_row = []
            for i in range(column_count):
                if is_numeric[i]:
                    try:
                        numeric_row.append(float(row[i]))
                    except (ValueError, IndexError):
                        numeric_row.append(float("nan"))

            data.append(numeric_row)

    return np.array(data, dtype=float)