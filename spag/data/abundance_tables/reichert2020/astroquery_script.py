import pandas as pd
from astroquery.simbad import Simbad
from astropy.table import Table

# Set Simbad configuration
Simbad.ROW_LIMIT = 10000
Simbad.add_votable_fields(
    'flux(U)', 'flux(B)', 'flux(V)', 'flux(R)', 'flux(I)', 'flux(J)', 'flux(H)', 'flux(K)',
    'otype', 'sp', 'ra', 'dec', 'pmra', 'pmdec', 'plx', 'rv_value'
)
# Read in identifiers
with open('/Users/ayelland/Research/metal-poor-stars/spag/spag/data/abundance_tables/reichert2020/id_query.txt', 'r') as f:
    list_of_identifiers = [line.strip() for line in f if line.strip()]

# Prepare results storage
results_list = []

# Query Simbad and collect results
for identifier in list_of_identifiers:
    try:
        query_result = Simbad.query_object(identifier)
        if query_result is not None:
            # Convert to pandas DataFrame and add identifier and found flag
            result_df = query_result.to_pandas()
            result_df['Query_ID'] = identifier
            result_df['Found'] = True
        else:
            # Object not found
            result_df = pd.DataFrame([{'Query_ID': identifier, 'Found': False}])
    except Exception as e:
        # Handle errors (e.g., malformed names or timeouts)
        result_df = pd.DataFrame([{'Query_ID': identifier, 'Found': False, 'Error': str(e)}])
    
    results_list.append(result_df)

# Combine all into a single DataFrame
final_df = pd.concat(results_list, ignore_index=True)

# Move Query_ID and Found to front
cols = final_df.columns.tolist()
for col in ['Query_ID', 'Found']:
    if col in cols:
        cols.insert(0, cols.pop(cols.index(col)))
final_df = final_df[cols]

# Save to CSV
final_df.to_csv('/Users/ayelland/Research/metal-poor-stars/spag/spag/data/abundance_tables/reichert2020/id_result.csv', index=False)

# Display a preview
print(final_df.head())
