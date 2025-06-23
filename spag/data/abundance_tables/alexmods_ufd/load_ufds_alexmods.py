def load_ufds_alexmods():
    """
    Load the UFD galaxies from Alexmods, parse abundance values and upper limits.

    Returns:
        pd.DataFrame: A cleaned DataFrame with numerical abundance columns and separate upper limit columns.
    """
    ufd_df = pd.read_csv(data_dir + "abundance_tables/alexmods_ufd/alexmods_ufd_yelland.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

    # Identify abundance columns
    abundance_cols = [col for col in ufd_df.columns if col.startswith("[") and (col.endswith("Fe]") or col.endswith("H]"))]

    # Initialize upper limit columns
    for col in abundance_cols:
        ufd_df["ul" + col] = np.nan

    # Parse string values into numeric + upper limit
    for col in abundance_cols:
        # Strings with '<' are upper limits
        mask = ufd_df[col].astype(str).str.contains("<")
        ufd_df.loc[mask, "ul" + col] = ufd_df.loc[mask, col].astype(str).str.replace("<", "").astype(float)  # Extract upper limit values
        ufd_df.loc[mask, col] = np.nan  # Replace upper limits in main column with NaN

    ## Sort the columns to have the upper limit columns next to the abundance columns
    sorted_cols = []
    for col in abundance_cols:
        sorted_cols.append(col)
        sorted_cols.append("ul" + col)
    # Add other columns that are not abundance columns
    other_cols = [col for col in ufd_df.columns if col not in abundance_cols and not col.startswith("ul")]
    sorted_cols = other_cols + sorted_cols
    ufd_df = ufd_df[sorted_cols]

    ## Fill the NaN values in the RA and DEC columns
    for idx, row in ufd_df.iterrows():
        if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):
            ## pad RA_hms with leading zeros
            if len(row['RA_hms']) == 10:
                row['RA_hms'] = '0' + row['RA_hms']
                ufd_df.at[idx, 'RA_hms'] = row['RA_hms']
            row['RA_deg'] = coord.ra_hms_to_deg(row['RA_hms'], precision=6)
            ufd_df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = coord.dec_dms_to_deg(row['DEC_dms'], precision=2)
            ufd_df.at[idx, 'DEC_deg'] = row['DEC_deg']

        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = coord.ra_deg_to_hms(float(row['RA_deg']), precision=2)
            ufd_df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = coord.dec_deg_to_dms(float(row['DEC_deg']), precision=2)
            ufd_df.at[idx, 'DEC_dms'] = row['DEC_dms']

    return ufd_df