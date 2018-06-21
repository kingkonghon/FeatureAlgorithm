

def renameDF(df, old_col_name, new_col_name):
    rename_dict = {}
    for field_pair in zip(old_col_name, new_col_name):
        rename_dict[field_pair[0]] = field_pair[1]
    new_df = df.rename(columns=rename_dict)
    return new_df


def chgDFDataType(df, col_name, new_data_type):
    for col in col_name:
        df.loc[:, col] = df[col].astype(new_data_type)

    return df