f07_abund_df_wide = pd.read_csv("/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/francois2007/table3_4_5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
f07_abund_df = transpose_abund_df(f07_abund_df_wide, 'grevesse1998', init_abund='[X/Fe]')
#drop the "Fe I" rows
f07_abund_df = f07_abund_df[f07_abund_df['Ion'] != 'Fe I']
f07_abund_df.to_csv("/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/francois2007/table3_4_5_transposed.csv", index=False)

c04_abund_df = pd.read_csv("/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/cayrel2004/table8.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
c04_abund_df['Z'] = c04_abund_df['Ion'].apply(lambda elem: ion_to_species(elem))

c04_cols = c04_abund_df.columns
f07_cols = f07_abund_df.columns
shared_columns = list(c04_cols.intersection(f07_cols))

abund_df = pd.concat([c04_abund_df[shared_columns + ['e_[X/H]']], f07_abund_df[shared_columns]], ignore_index=True)
abund_df = abund_df[['Name', 'Z', 'Ion', 'l_logepsX', 'logepsX', 'e_[X/H]']]
abund_df.sort_values(['Name', 'Z'], inplace=True)

display(abund_df.head(20))
abund_df.to_csv("/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/cayrel2004/abundances-c04_f07_combined.csv", index=False)
abund_df.to_csv("/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/francois2007/abundances-c04_f07_combined.csv", index=False)