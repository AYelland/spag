import sys

def markdown_table_to_csv(md_filepath, csv_filepath):
    with open(md_filepath, 'r') as md_file:
        lines = md_file.readlines()
        
    header = lines[0].strip().split('|')[1:-1]  # Exclude leading and trailing '|'
    rows = [line.strip().split('|')[1:-1] for line in lines[2:]]  # Skip the header separator line

    # Write to CSV
    with open(csv_filepath, 'w') as csv_file:
        csv_file.write(','.join(header) + '\n')
        for row in rows:
            csv_file.write(','.join(cell.strip() for cell in row) + '\n')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_md_filepath> <output_csv_filepath>")
    else:
        input_md_filepath = sys.argv[1]
        output_csv_filepath = sys.argv[2]
        markdown_table_to_csv(input_md_filepath, output_csv_filepath)
