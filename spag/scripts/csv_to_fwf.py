import pandas as pd
import sys

def fwf_to_csv(input_fwf, output_csv):
    """Convert fixed-width file to CSV."""
    df_fwf = pd.read_fwf(input_fwf, header=None)
    df_fwf.to_csv(output_csv, index=False, header=None)
    print(f"FWF converted to CSV: {output_csv}")

def csv_to_fwf(input_csv, output_fwf):
    """Convert CSV to fixed-width file (auto-calculated widths)."""
    # Always treat the CSV as having **no header**
    df_csv = pd.read_csv(input_csv, header=None)

    # Calculate column widths based on data, not assuming headers
    col_widths = [df_csv[col].astype(str).map(len).max() + 2 for col in df_csv.columns]

    with open(output_fwf, 'w') as f:
        for _, row in df_csv.iterrows():
            line = ''.join(f'{str(val):<{col_widths[i]}}' for i, val in enumerate(row))
            f.write(line.rstrip() + '\n')
    print(f"CSV converted to FWF: {output_fwf}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py [fwf_to_csv|csv_to_fwf] input_file output_file")
        sys.exit(1)

    mode, input_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    if mode == 'fwf_to_csv':
        fwf_to_csv(input_file, output_file)
    elif mode == 'csv_to_fwf':
        csv_to_fwf(input_file, output_file)
    else:
        print("Invalid mode. Use 'fwf_to_csv' or 'csv_to_fwf'.")
