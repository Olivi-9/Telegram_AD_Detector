import sys
import csv


def clean_csv(input_path: str, output_path: str, min_length: int = 10):
    with open(input_path, newline="", encoding="utf-8") as infile, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            line = ",".join(row)  # reconstruct line
            if len(line) >= min_length:
                writer.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_short_rows.py input.csv output.csv")
        sys.exit(1)

    inp, outp = sys.argv[1], sys.argv[2]
    clean_csv(inp, outp)
    print(f"Cleaned {inp} -> {outp}")
