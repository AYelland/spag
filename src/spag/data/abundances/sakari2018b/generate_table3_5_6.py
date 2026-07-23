tab1 = pd.read_csv(f"/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sakari2018b/table1.csv", comment="#")
name_id_dict = dict(zip(tab1['Name'], tab1['ID']))
tab3 = pd.read_csv(f"/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sakari2018b/table3.csv", comment="#")
tab3.drop(columns=['Std', 'Teff', 'log(g)', 'xi', 'CEMP'], inplace=True)
tab5 = pd.read_csv(f"/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sakari2018b/table5.csv", comment="#")
tab5.drop(columns=['Set', 'Class', 'f_Class'], inplace=True)
tab6 = pd.read_csv(f"/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sakari2018b/table6.csv", comment="#")
tab6['ID'] = tab6['Name'].map(name_id_dict)
tab9 = pd.read_csv(f"/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sakari2018b/table9.csv", comment="#")
tab9['ID'] = tab9['Name'].map(name_id_dict)

# display(tab3.head())
# display(tab5.head())
# display(tab6.head())
# display(tab6[['ID', 'Name']])

# combine tables 3, 5, 6 into a single dataframe using the 'ID' column as the key
combined_df = pd.merge(tab3, tab5, on='ID', how='outer', suffixes=('_tab3', '_tab5'))
combined_df = pd.merge(combined_df, tab6, on='ID', how='outer', suffixes=('', '_tab6'))

# sort the columns in the combined dataframe
XH_cols = XHcolnames(combined_df)
lXH_cols = ['l_' + col for col in XH_cols]
eXH_cols = ['e_' + col for col in XH_cols]
nXH_cols = ['N_' + col for col in XH_cols]
XFe_cols = XFecolnames(combined_df)
lXFe_cols = ['l_' + col for col in XFe_cols]
eXFe_cols = ['e_' + col for col in XFe_cols]
nXFe_cols = ['N_' + col for col in XFe_cols]

col_order = []
for i in range(len(XH_cols)):
    col_order.append(nXH_cols[i])
    col_order.append(lXH_cols[i])
    col_order.append(XH_cols[i])
    col_order.append(eXH_cols[i])
for i in range(len(XFe_cols)):
    col_order.append(nXFe_cols[i])
    col_order.append(lXFe_cols[i])
    col_order.append(XFe_cols[i])
    col_order.append(eXFe_cols[i])
col_order = ['ID', 'Name'] + col_order
other_cols = [col for col in combined_df.columns if col not in col_order]

combined_df = combined_df.reindex(columns=col_order)
#cast the N_ columns to integers
for col in combined_df.columns:
    if col.startswith('N_'):
        combined_df[col] = combined_df[col].astype('Int64')

display(combined_df.head())
combined_df.to_csv(f"/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sakari2018b/table3_5_6.csv", index=False)

##############
# manually find-replace ",<," with ",,<" in the table3_5_6.csv file to fix a formatting issue
##############

combined_df = pd.read_csv(f"/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sakari2018b/table3_5_6.csv")
l_cols = [col for col in combined_df.columns if 'l_' in col]

for col in l_cols:
    # if all values in the column are NaN, drop the column
    if combined_df[col].isna().all():
        combined_df.drop(columns=[col], inplace=True)

l_cols_remaining = [col for col in combined_df.columns if 'l_' in col]
print(f"Remaining l_ columns: {l_cols_remaining}")

combined_df.to_csv(f"/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/sakari2018b/table3_5_6_T.csv", index=False)

##############
# manually added the Li abundances from table 7
##############