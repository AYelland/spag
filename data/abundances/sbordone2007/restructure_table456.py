import pandas as pd
import numpy as np

# Load your CSV data (you can replace this with pd.read_csv('filename.csv') if from file)
df_logeps = pd.read_csv("/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sbordone2007/table456a.csv")

# Extract the list of star IDs from the first row (excluding the first two columns)
star_ids = df_logeps.columns[1:].to_list()
sun_col = star_ids[0]
star_ids = star_ids[1:]

# Create column groupings: every pair (logepsX, e_logepsX)
star_pairs = [(star_ids[i], star_ids[i+1]) for i in range(0, len(star_ids), 2)]
print(star_pairs)

# Initialize the output list
records = []

# Loop through each species (row)
for idx, row in df_logeps.iterrows():
    if idx == 0: continue
    species = row['Species']
    logepsX_sun = row[sun_col]

    for star, (log_col, err_col) in zip(star_pairs, star_pairs):
        star_id = int(log_col)  # e.g., 432, 628
        logeps = row[log_col]
        e_logeps = row[err_col]

        records.append({
            'StarID': star_id,
            'Species': species,
            'logepsX_sun': logepsX_sun,
            'logepsX': logeps,
            'e_logepsX': e_logeps
        })

# Convert to a DataFrame
df_logeps_long = pd.DataFrame.from_records(records)

# Optionally sort and reset index
df_logeps_long = df_logeps_long.sort_values(['StarID']).reset_index(drop=True)

# Display or export
# display(df_logeps_long.head(50))
df_logeps_long.to_csv("/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sbordone2007/table456a_long.csv", index=False)


######################################

import pandas as pd
import numpy as np

# Load your CSV data (you can replace this with pd.read_csv('filename.csv') if from file)
df = pd.read_csv("/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sbordone2007/table456b.csv")

# Extract the list of star IDs from the first row (excluding the first two columns)
star_ids = df.columns[1:].to_list()
print(star_ids)

# sun_col = star_ids[0]
# star_ids = star_ids[1:]

# Create column groupings: every pair (logepsX, e_logepsX)
star_pairs = [(star_ids[i], star_ids[i+1]) for i in range(0, len(star_ids), 2)]
print(star_pairs)

# Initialize the output list
records = []

# Loop through each species (row)
for idx, row in df.iterrows():
    if idx == 0: continue
    species = row['Species']


    for star, (log_col, err_col) in zip(star_pairs, star_pairs):
        star_id = int(log_col)  # e.g., 432, 628
        xfe = row[log_col]
        e_xfe = row[err_col]

        records.append({
            'StarID': star_id,
            'Species': species,
            # 'logepsX_sun': logepsX_sun,
            '[X/Fe]': xfe,
            'e_[X/Fe]': e_xfe
        })

# Convert to a DataFrame
df_long = pd.DataFrame.from_records(records)

# Optionally sort and reset index
df_long = df_long.sort_values(['StarID']).reset_index(drop=True)

# Display or export
# display(df_long.head(50))
df_long.to_csv("/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sbordone2007/table456b_long.csv", index=False)
