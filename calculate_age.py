import pandas as pd
import numpy as np
import ast

def calculateAGEforCancer(data): # calculate based on earliest cancer diagnosis date (CANCER_DATE)
    age_new = list()
    for index, row in data.iterrows():
        dob = row['Date of Birth (yyyy-mm-dd)']
        can_date = row['CANCER_DATE']
        d1 = pd.to_datetime(dob)
        d2 = pd.to_datetime(can_date)
        age = round(abs((d2 - d1).days)/365, 2)
        age_new.append(age)
    return age_new
#############################################
def calculateAGEforNonCancer(data): # calculate based on earliest HBA1C reporting date
    age_new = list()
    for index, row in data.iterrows():
        dob = row['Date of Birth (yyyy-mm-dd)']
        hba1c_date = row['HBA1C_dates'].replace('[','').replace(']','').split(",")[0] # as the dates are sorted, first date is earliest date 
        d1 = pd.to_datetime(dob)
        d2 = pd.to_datetime(hba1c_date)
        age = round(abs((d2 - d1).days)/365, 2)
        age_new.append(age)
    
    return age_new
#############################################

df = pd.read_csv("/Users/kabir.neamul/Desktop/data/all_feature_data_final.csv")
df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)

dob_df = pd.read_csv("~/Desktop/data/date_of_birth.csv")
data = df.merge(dob_df, on='Reference Key', how='left')
print(data.shape)

new_df = data[[ 'Reference Key', 'ALC','SMOKING','AGE', 'BMI','Sex_n', 'CANCER',\
                'CANCER_DATE','CRC', 'ASPIRIN_BASE', 'Date of Birth (yyyy-mm-dd)']].copy()

features = ['HBA1C', 'HB', 'WBC', 'RBC', 'PLT', 'CAL', 'CREA', 'POTASS', 'PHOS','SODIUM', 'UREA',\
            'ALBMN', 'BILI', 'CHOL', 'PROT', 'ALP', 'ALT', 'TRIG', 'HDL', 'LDL_CAL', 'LDL', 'GLU_FAST', \
           'GLU', 'GLU_RAN', 'EGFR','IRON', 'IRON_SAT', 'HBSAG', 'AHB','AST', 'VB12', 'FERR', 'IRONBIND',\
           'PTIME', 'CAL_ALBMN', 'PROT_U24', 'PROT_USP', 'SODIUM_USP', 'SODIUM_U24', 'CREATININE_U',\
           'FOLATE_RBC', 'AGGLUTININ', 'FOLATE', 'APTT', 'CHOL_NF', 'HDL_NF', 'LDL_CAL_NF', 'TRIG_NF']
print(len(features))

#### sort all feature dates in ascending order and sort the values accordingly
for feat in features:
    feat_dates = feat + "_dates"
    data[feat] = data[feat].apply(ast.literal_eval)
    data[feat_dates] = data[feat_dates].apply(ast.literal_eval)

    data['_row_id'] = data.index

    df_exp = data.explode([feat, feat_dates])
    
    df_exp[feat_dates] = df_exp[feat_dates].str.strip()
    df_exp = df_exp.sort_values(['_row_id', feat_dates])
    
    df_sorted = df_exp.groupby('_row_id').agg({
        feat: list,
        feat_dates: list
    })
    new_df[feat] = df_sorted[feat]
    new_df[feat_dates] = df_sorted[feat_dates]

# new_df.to_csv("~/Desktop/data/data/all_features_sorted.csv", index=False)
# new_df = pd.read_csv("~/Desktop/data/data/all_features_sorted.csv")
### calculate age for cancer patients
df_c = new_df[new_df['CANCER']==1][['Reference Key', 'CANCER', 'CANCER_DATE', 'Date of Birth (yyyy-mm-dd)', 'AGE']] # only the cancer patients
df_c['AGE_new'] = calculateAGEforCancer(df_c)

### calculate age for non-cancer patients
df_n = new_df[new_df['CANCER']==0][['Reference Key', 'HBA1C', 'HBA1C_dates', 'Date of Birth (yyyy-mm-dd)', 'AGE']] # only the non-cancer patients
df_n['AGE_new'] = calculateAGEforNonCancer(df_n)

df_c_age = df_c [['Reference Key', 'AGE_new']] ## keep only the age column for cancer
df_n_age = df_n [[ 'Reference Key', 'AGE_new' ]] ## keep only the age column for non-cancer
df_age = pd.concat([df_c_age, df_n_age], axis=0) ### combine cancer and non-cancer

new_df_age = new_df.merge(df_age, on='Reference Key') ### merge back to the original dataframe with all features

new_df_age.to_csv("~/Desktop/data/data/all_features_with_new_age_sorted.csv", index=False)
