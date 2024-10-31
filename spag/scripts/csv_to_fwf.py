import csv
import sys

def get_max_widths(input_csv):
    """Calculate the maximum width of each column."""
    with open(input_csv, 'r') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)
        num_columns = len(headers)
        max_widths = [len(header)+4 for header in headers]
        
        print(max_widths)
        
        for row in reader:
            for i in range(num_columns):
                max_widths[i] = max(max_widths[i], len(row[i]))
    
    # Add 2 characters of padding
    return [width + 2 for width in max_widths]

def csv_to_fwf(input_csv, output_txt):
    """Convert CSV to fixed-width format based on column widths."""
    col_widths = get_max_widths(input_csv)
    
    with open(input_csv, 'r') as csv_file, open(output_txt, 'w') as txt_file:
        reader = csv.reader(csv_file)
        
        for row in reader:
            fwf_row = ""
            for i, col in enumerate(row):
                # Adjust each column to the calculated width, pad with spaces if necessary
                fwf_row += f"{col:<{col_widths[i]}}"
            txt_file.write(fwf_row.rstrip() + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_csv output_txt")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_txt = sys.argv[2]

    csv_to_fwf(input_csv, output_txt)
    print(f"Fixed-width file saved as {output_txt}")
