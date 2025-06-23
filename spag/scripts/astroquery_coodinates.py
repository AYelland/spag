import pandas as pd
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

# Set Simbad configuration
Simbad.ROW_LIMIT = 10000
Simbad.add_votable_fields(
    'flux(U)', 'flux(B)', 'flux(V)', 'flux(R)', 'flux(I)', 'flux(J)', 'flux(H)', 'flux(K)',
    'otype', 'sp', 'ra', 'dec', 'pmra', 'pmdec', 'plx', 'rv_value'
)

# Read RA and DEC from file (assume tab- or comma-separated, or adjust as needed)
coord_file = '/Users/ayelland/Research/metal-poor-stars/spag/spag/data/abundance_tables/gull2021/astroquery.csv'
coord_df = pd.read_csv(coord_file)

# Expected columns: RA_hms, DEC_dms (e.g., "12 34 56.7", "-45 23 12")
# Adjust column names as necessary

# Prepare results storage
results_list = []

# Loop through each row and perform coordinate-based query
for idx, row in coord_df.iterrows():
    ra_hms = row['RA_hms']
    dec_dms = row['DEC_dms']
    query_id = row.get('Name', f'coord_{idx}')  # Optional ID column, else fallback to index

    try:
        coord = SkyCoord(ra=ra_hms, dec=dec_dms, unit=(u.hourangle, u.deg))
        query_result = Simbad.query_region(coord, radius='5s')  # Small radius around point

        if query_result is not None:
            result_df = query_result.to_pandas()
            result_df['Query_ID'] = query_id
            result_df['RA_input'] = ra_hms
            result_df['DEC_input'] = dec_dms
            result_df['Found'] = True
        else:
            result_df = pd.DataFrame([{
                'Query_ID': query_id,
                'RA_input': ra_hms,
                'DEC_input': dec_dms,
                'Found': False
            }])

    except Exception as e:
        result_df = pd.DataFrame([{
            'Query_ID': query_id,
            'RA_input': ra_hms,
            'DEC_input': dec_dms,
            'Found': False,
            'Error': str(e)
        }])

    results_list.append(result_df)

# Combine and reorder
final_df = pd.concat(results_list, ignore_index=True)
cols = final_df.columns.tolist()
for col in ['Query_ID', 'RA_input', 'DEC_input', 'Found']:
    if col in cols:
        cols.insert(0, cols.pop(cols.index(col)))
final_df = final_df[cols]

# Save to CSV
final_df.to_csv('/Users/ayelland/Research/metal-poor-stars/spag/spag/data/abundance_tables/gull2021/astroquery_results.csv', index=False)

# Display a preview
print(final_df.head())
