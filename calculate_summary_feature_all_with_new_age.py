import numpy as np
import pandas as pd
import sys

import pandas as pd
import numpy as np

#####################################################
def parse_values(x):
    if isinstance(x, list):
        return [float(i) for i in x if pd.notna(i)]
    if pd.isna(x) or x == "":
        return []
    return [float(i.strip().replace("'", "")) 
            for i in x.strip('[]').split(',') 
            if i.strip() not in ["", "nan"]]

def parse_dates(x):
    if isinstance(x, list):
        return [str(i).strip() for i in x if pd.notna(i)]
    if pd.isna(x) or x == "":
        return []
    return [i.strip().replace("'", "") 
            for i in x.strip('[]').split(',') 
            if i.strip() not in ["", "nan"]]
#####################################################
def calculatePercentageOfNull(df):
    total_rows = len(df)

    # compute null counts and percentages
    null_counts = df.isnull().sum()
    null_percentage = (null_counts / total_rows) * 100

    # combine into a new dataframe
    null_summary = pd.DataFrame({
        'column_name': null_counts.index,
        'null_count': null_counts.values,
        'null_percentage': null_percentage.values
    })

    # optional: sort by highest null percentage
    null_summary = null_summary.sort_values(by='null_percentage', ascending=False)

    return null_summary
#####################################################

file="~/Desktop/data/data/all_features_with_new_age_sorted.csv"
data = pd.read_csv(file)
print(data.shape)
# print(data.columns)

### take year window as input. currently we divide into four windows: 3-year, 2-year, 1-year, 0-year
year_window = int(sys.argv[1])
#### list of all 48 feature names as stated in the data file
features = ['HBA1C', 'HB', 'WBC', 'RBC', 'PLT', 'CAL', 'CREA', 'POTASS', 'PHOS','SODIUM', 'UREA',\
            'ALBMN', 'BILI', 'CHOL', 'PROT', 'ALP', 'ALT', 'TRIG', 'HDL', 'LDL_CAL', 'LDL', 'GLU_FAST', \
           'GLU', 'GLU_RAN', 'EGFR','IRON', 'IRON_SAT', 'HBSAG', 'AHB','AST', 'VB12', 'FERR', 'IRONBIND',\
           'PTIME', 'CAL_ALBMN', 'PROT_U24', 'PROT_USP', 'SODIUM_USP', 'SODIUM_U24', 'CREATININE_U',\
           'FOLATE_RBC', 'AGGLUTININ', 'FOLATE', 'APTT', 'CHOL_NF', 'HDL_NF', 'LDL_CAL_NF', 'TRIG_NF']
### extract cancer_dates once, as this is same for all features
# start_time = time.perf_counter()
cancer_dates = data['CANCER_DATE']


data['_row_id'] = data.index

summary_df = data[[ '_row_id', 'Reference Key', 'ALC','SMOKING','AGE', 'AGE_new','BMI','Sex_n', 'CANCER',\
                'CANCER_DATE','CRC', 'ASPIRIN_BASE', 'Date of Birth (yyyy-mm-dd)']].copy()
print(summary_df.shape)

## for each feature collect those within year window and get summary features for them
for col_name in features:
    print(" ~~~ Working with %s feature ~~~"%col_name)
    col_dates = col_name +"_dates"
    data[col_name] = data[col_name].apply(parse_values)
    data[col_dates] = data[col_dates].apply(parse_dates)
    df_exp = data.explode([col_name, col_dates])

    df_exp[col_dates] = pd.to_datetime(df_exp[col_dates], errors='coerce')
    df_exp['CANCER_DATE'] = pd.to_datetime(df_exp['CANCER_DATE'], errors='coerce')
    df_exp[col_name] = pd.to_numeric(df_exp[col_name], errors='coerce')

    # remove invalid rows
    df_exp = df_exp.dropna(subset=[col_dates, col_name, 'CANCER_DATE'])

    # only keep measurements BEFORE cancer
    df_exp = df_exp[df_exp[col_dates] < df_exp['CANCER_DATE']]

    # compute year difference
    df_exp['year_diff'] = (df_exp['CANCER_DATE'] - df_exp[col_dates]) / pd.Timedelta(days=365)

    # apply year window
    df_exp = df_exp[df_exp['year_diff'] >= year_window]

    agg = df_exp.groupby('_row_id')[col_name].agg(['count', 'mean', 'median', 'std'])
    agg.loc[agg['count'] < 5, ['mean', 'median', 'std']] = None ### use min observation required to calculate summary feature

    summary_df = summary_df.merge(agg[['mean', 'median', 'std']], left_on='_row_id', right_index=True, how='left')

    col_mean = col_name + "_mean"; col_median = col_name + "_median"; col_std = col_name + "_std"
    summary_df.rename(columns={
        'mean': col_mean,
        'median': col_median,
        'std': col_std
    }, inplace=True)

filename = "~/Desktop/data/data/summary_features_two+_year_window_"+str(year_window)+".csv"
summary_df.to_csv(filename, index=False)

### calculate percentage of null values for current min observation value
null_summary = calculatePercentageOfNull(summary_df)
null_summary.to_csv("~/Desktop/data/data/percentage_of_null_all_feature_two+_year_window_"+str(year_window)+".csv", index=False)