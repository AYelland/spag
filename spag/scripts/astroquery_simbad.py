import pandas as pd
import sys, os
import argparse
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

# Setup Simbad fields
Simbad.ROW_LIMIT = 10000
Simbad.add_votable_fields(
    'flux(U)', 'flux(B)', 'flux(V)', 'flux(R)', 'flux(I)', 'flux(J)', 'flux(H)', 'flux(K)',
    'otype', 'sp', 'ra', 'dec', 'pmra', 'pmdec', 'plx', 'rv_value'
)

# Argument parser
parser = argparse.ArgumentParser(description="Query SIMBAD with identifiers or coordinates.")
parser.add_argument("reference", type=str, help="Name of the subdirectory inside abundance_tables/")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-i", "--identifier", action="store_true", help="Use object identifiers for query")
group.add_argument("-c", "--coordinates", action="store_true", help="Use coordinates (RA_hms, DEC_dms) for query")

args = parser.parse_args()
reference = args.reference
# base_path = f"/Users/ayelland/Research/metal-poor-stars/spag/spag/data/abundance_tables/{reference}"
base_path = "/Users/ayelland/Research/metal-poor-stars/project/carbon-project-2025/"
input_file = os.path.join(base_path, "astroquery.csv")

# Collect results
results_list = []

## Load the 'astroquery.csv' file
coord_df = pd.read_csv(input_file)

for idx, row in coord_df.iterrows():

    if args.identifier:
        identifier = row.get('Name', f'coord_{idx}').strip()
        if 'RA_hms' in row and 'DEC_dms' in row:
            ra_hms = row['RA_hms']
            dec_dms = row['DEC_dms']
        if 'JINA_ID' in row:
            jinaid = row['JINA_ID']
        try:
            result = Simbad.query_object(identifier)
            if result is not None:
                df = result.to_pandas()
                df['Found'] = True
                df['JINA_ID'] = jinaid if 'jinaid' in locals() else None
                df['Query_ID'] = identifier
                df['RA_input'] = ra_hms if 'ra_hms' in locals() else None
                df['DEC_input'] = dec_dms if 'dec_dms' in locals() else None
            else:
                df = pd.DataFrame([{
                    'Found': False,
                    'JINA_ID': jinaid if 'jinaid' in locals() else None,
                    'Query_ID': identifier, 
                    'RA_input': ra_hms if 'ra_hms' in locals() else None,
                    'DEC_input': dec_dms if 'dec_dms' in locals() else None 
                }])
        except Exception as e:
            df = pd.DataFrame([{
                'Found': False, 
                'JINA_ID': jinaid if 'jinaid' in locals() else None,
                'Query_ID': identifier,
                'RA_input': ra_hms if 'ra_hms' in locals() else None,
                'DEC_input': dec_dms if 'dec_dms' in locals() else None,
                'Error': str(e)
            }])
        results_list.append(df)

    elif args.coordinates:
        identifier = row.get('Name', f'coord_{idx}').strip()
        ra_hms = row['RA_hms']
        dec_dms = row['DEC_dms']
        if 'JINA_ID' in row:
            jinaid = row['JINA_ID']
        try:
            coord = SkyCoord(ra=ra_hms, dec=dec_dms, unit=(u.hourangle, u.deg))
            result = Simbad.query_region(coord, radius='5s')
            if result is not None:
                df = result.to_pandas()
                df['Found'] = True
                df['JINA_ID'] = jinaid if 'jinaid' in locals() else None
                df['Query_ID'] = identifier
                df['RA_input'] = ra_hms
                df['DEC_input'] = dec_dms
            else:
                df = pd.DataFrame([{
                    'Found': False,
                    'JINA_ID': jinaid if 'jinaid' in locals() else None,
                    'Query_ID': identifier,
                    'RA_input': ra_hms,
                    'DEC_input': dec_dms
                }])
        except Exception as e:
            df = pd.DataFrame([{
                'Found': False,
                'JINA_ID': jinaid if 'jinaid' in locals() else None,
                'Query_ID': identifier,
                'RA_input': ra_hms,
                'DEC_input': dec_dms,
                'Error': str(e)
            }])
        results_list.append(df)

# Combine results and reorder columns
final_df = pd.concat(results_list, ignore_index=True)

priority_cols = ['Found', 'JINA_ID', 'Query_ID', 'RA_input', 'DEC_input']
for col in priority_cols:
    if col not in final_df.columns:
        final_df[col] = pd.NA
cols = final_df.columns.tolist()

for col in reversed(priority_cols):
    if col in cols:
        cols.insert(0, cols.pop(cols.index(col)))
final_df = final_df[cols]

# Save and preview
output_file = os.path.join(base_path, "astroquery_results.csv")
final_df.to_csv(output_file, index=False)
print(final_df.head())
