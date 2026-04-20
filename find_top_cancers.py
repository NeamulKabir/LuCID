import re
import pandas as pd
import numpy as np
import sys

### use year-window=3 to get top cancers
### this one has the least amount of observations, other year windows have all the same patients plus a few more. 
data3 = pd.read_csv("~/Desktop/data/data_two+/summary_features_two+_year_window_3.csv")

### get cancer icd9 codes and merge with the feature file
cancer_codes = pd.read_csv("~/Desktop/data/cancer_icd_codes.csv")
cancer_codes.drop(['Unnamed: 0', 'Reference Date', 'CANCER'], axis = 1, inplace=True)
data3 = data3.merge(cancer_codes, on='Reference Key', how='left')
print(data3.shape)

feature_set = sys.argv[1] ### get input which feature set to be used, input either 4 or 15

### extract only the four features and drop NA from there
if feature_set == "4":
    feat_count = "four"
    four_features = ["_row_id", "Reference Key", "AGE", "AGE_new", "BMI", "Sex_n", "ALC", "SMOKING",\
        "CANCER", "CANCER_DATE", "CRC", "HBA1C_mean", "HBA1C_median", "HBA1C_std", "TRIG_mean", \
        "TRIG_median", "TRIG_std", "LDL_CAL_mean", "LDL_CAL_median", "LDL_CAL_std", \
        "GLU_FAST_mean", "GLU_FAST_median", "GLU_FAST_std", "All Diagnosis Code (ICD9)"]
    data3_four = data3[four_features].copy()
    print(data3_four.shape)
    data3_clean = data3_four.dropna()
### extract 15 lab mesurements that have <40% missingness
elif feature_set =="15":
    feat_count = "fifteen"
    fifteen_features = ["_row_id", "Reference Key", 'AGE_new', 'Sex_n', 'BMI', 'ALC', 'SMOKING', "CANCER", "CANCER_DATE", "CRC", \
                'HBA1C_mean','HBA1C_median','HBA1C_std', 'TRIG_mean','TRIG_median','TRIG_std',\
                'LDL_CAL_mean','LDL_CAL_median','LDL_CAL_std', 'CREA_mean', 'CREA_median', 'CREA_std', \
                'POTASS_mean', 'POTASS_median', 'POTASS_std', 'SODIUM_mean', 'SODIUM_median', 'SODIUM_std',\
                'UREA_mean', 'UREA_median', 'UREA_std', 'CHOL_mean', 'CHOL_median', 'CHOL_std', \
                'HDL_mean', 'HDL_median', 'HDL_std', 'PROT_mean', 'PROT_median', 'PROT_std', \
                'GLU_FAST_mean','GLU_FAST_median','GLU_FAST_std',\
                'ALBMN_mean', 'ALBMN_median', 'ALBMN_std', 'BILI_mean', 'BILI_median', 'BILI_std', \
                'ALP_mean', 'ALP_median', 'ALP_std','ALT_mean', 'ALT_median', 'ALT_std', "All Diagnosis Code (ICD9)" ]
    data3_fifteen = data3[fifteen_features].copy()
    print(data3_fifteen.shape)
    data3_clean = data3_fifteen.dropna()
### extract 19 lab measurements tha have <40% missingness based on two or more observations
elif feature_set =="19":
    feat_count = "nineteen"
    nineteen_features = ["_row_id", "Reference Key", 'AGE_new', 'Sex_n', 'BMI', 'ALC', 'SMOKING', "CANCER", "CANCER_DATE", "CRC", \
                'HBA1C_mean','HBA1C_median','HBA1C_std', 'TRIG_mean','TRIG_median','TRIG_std',\
                'LDL_CAL_mean','LDL_CAL_median','LDL_CAL_std', 'CREA_mean', 'CREA_median', 'CREA_std', \
                'POTASS_mean', 'POTASS_median', 'POTASS_std', 'SODIUM_mean', 'SODIUM_median', 'SODIUM_std',\
                'UREA_mean', 'UREA_median', 'UREA_std', 'CHOL_mean', 'CHOL_median', 'CHOL_std', \
                'HDL_mean', 'HDL_median', 'HDL_std', 'PROT_mean', 'PROT_median', 'PROT_std', \
                'GLU_FAST_mean','GLU_FAST_median','GLU_FAST_std', 'HB_mean', 'HB_median', 'HB_std',\
                'ALBMN_mean', 'ALBMN_median', 'ALBMN_std', 'BILI_mean', 'BILI_median', 'BILI_std', \
                'ALP_mean', 'ALP_median', 'ALP_std','ALT_mean', 'ALT_median', 'ALT_std', \
                'RBC_mean', 'RBC_median', 'RBC_std', 'WBC_mean', 'WBC_median', 'WBC_std', \
                'PLT_mean', 'PLT_median', 'PLT_std', "All Diagnosis Code (ICD9)" ]
    data3_nineteen = data3[nineteen_features].copy()
    print(data3_nineteen.shape)
    data3_clean = data3_nineteen.dropna()

### get cancer count
cancer_count = np.zeros(70,)
data3_c = data3_clean[data3_clean['CANCER']==1]
print(data3_c.shape)
for i in range(len(data3_c)):
    cur_row = data3_c.iloc[i]
    # print(cur_row)
    icd_code = cur_row['All Diagnosis Code (ICD9)'].split(",")[1:]
    # print(icd_code)
    patient_codes = set()
    for item in icd_code:
        if item[0].isdigit():
            cur_code = int(item.split(".")[0])
            if cur_code >= 140 and cur_code <= 209:
                patient_codes.add(cur_code)
    for code in patient_codes:
        idx = code - 140 # subtract 140 to get array index
        cancer_count[idx] +=1
    
                
sorted_indices = np.argsort(cancer_count)

# 2. Slice the last 10 indices and reverse them for descending order
top_10_indices = sorted_indices[-20:][::-1] 

# 3. Use those indices to get the actual values
top_10_values = cancer_count[top_10_indices]
# print("Top 10 Indices:", top_10_indices)
# print("Top 10 Counts:", top_10_values)
new_df = pd.DataFrame()
new_df['Cancer ICD9'] = top_10_indices + 140 # add 140 back again to convert array index to ICD9 code
new_df['Counts'] = top_10_values

#new_df.to_csv("~/Desktop/data/data/top_cancers_four_features_five+.csv", index=False)
### print top 20 cancer icd9 codes and the count values
print(new_df.head(20))