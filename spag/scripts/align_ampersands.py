def align_ampersands_in_file(filename, start_line, end_line):
    # Read the file and split lines into a list
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Process only the lines in the specified range
    selected_lines = lines[start_line-1:end_line]
    
    # Split each line by "&" and calculate max widths for each column
    split_lines = [line.split('&') for line in selected_lines]
    max_col_widths = [max(len(col.strip()) for col in column) for column in zip(*split_lines)]
    
    # Reformat each line with aligned "&"
    aligned_lines = []
    for line_parts in split_lines:
        aligned_line = ' & '.join(col.strip().ljust(width) for col, width in zip(line_parts, max_col_widths))
        aligned_lines.append(aligned_line + '\n')
    
    # Replace original lines in range with aligned lines
    lines[start_line-1:end_line] = aligned_lines
    
    # Write the modified lines back to the file
    with open(filename, 'w') as file:
        file.writelines(lines)

# Usage example
align_ampersands_in_file('/Users/ayelland/Research/metal-poor-stars/project/sgr-project/tex-files/sgr_observation_parameters_v1.tex', 8, 44)
