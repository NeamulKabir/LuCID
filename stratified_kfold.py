import re
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

data3 = pd.read_csv("~/Desktop/data/data_two+/summary_features_two+_year_window_3.csv")

feature_set = sys.argv[1] ### take input if using four features or 21

#### merge cancers icd codes
cancer_codes = pd.read_csv("~/Desktop/data/cancer_icd_codes.csv")
cancer_codes.drop(['Unnamed: 0', 'Reference Date', 'CANCER'], axis = 1, inplace=True)
data3 = data3.merge(cancer_codes, on='Reference Key', how='left')
print(data3.shape)

### extract ref keys for top 10 cancers only
cancer_codes = [197,162,153,174,155,185,154,151,157,188] ## identified top cancers
ref_keys = data3[data3['CANCER']==0]['Reference Key'].tolist()
print(len(ref_keys))
data3_c = data3[data3['CANCER']==1]
count = 0
for i in range(len(data3_c)):
    cur_row = data3_c.iloc[i]
    # print(cur_row)
    icd_code = cur_row['All Diagnosis Code (ICD9)'].split(",")[1:]
    # print(icd_code)
    codes = list()
    for item in icd_code:
        if item[0].isdigit():
            cur_code = int(item.split(".")[0])
            if cur_code >= 140 and cur_code <= 209:
                if cur_code in cancer_codes:
                    count += 1
                    ref_keys.append(cur_row['Reference Key'])
                    break

### keep only top 10 cancers
data3_1 = data3[data3['Reference Key'].isin(ref_keys)].copy()
print(data3_1.shape)

############ drop NA values ##########

### extract only the four features and drop NA from there
if feature_set == "4":
    feat_count = "four"
    four_features = ["_row_id", "Reference Key", "AGE", "AGE_new", "BMI", "Sex_n", "ALC", "SMOKING",\
        "CANCER", "CANCER_DATE", "CRC", "HBA1C_mean", "HBA1C_median", "HBA1C_std", "TRIG_mean", \
        "TRIG_median", "TRIG_std", "LDL_CAL_mean", "LDL_CAL_median", "LDL_CAL_std", \
        "GLU_FAST_mean", "GLU_FAST_median", "GLU_FAST_std", "All Diagnosis Code (ICD9)"]
    data3_four = data3_1[four_features].copy()
    print(data3_four.shape)
    data3_clean = data3_four.dropna()
### extract 16 features that have <40% missingness
elif feature_set =='16':
    feat_count = "sixteen"
    sixteen_features = ["_row_id", "Reference Key", 'AGE_new', 'Sex_n', 'BMI', 'ALC', 'SMOKING', "CANCER", "CANCER_DATE", "CRC", \
                'HBA1C_mean','HBA1C_median','HBA1C_std', 'TRIG_mean','TRIG_median','TRIG_std',\
                'LDL_CAL_mean','LDL_CAL_median','LDL_CAL_std', 'CREA_mean', 'CREA_median', 'CREA_std', \
                'POTASS_mean', 'POTASS_median', 'POTASS_std', 'SODIUM_mean', 'SODIUM_median', 'SODIUM_std',\
                'UREA_mean', 'UREA_median', 'UREA_std', 'CHOL_mean', 'CHOL_median', 'CHOL_std', \
                'HDL_mean', 'HDL_median', 'HDL_std', 'PROT_mean', 'PROT_median', 'PROT_std', \
                 'GLU_FAST_mean','GLU_FAST_median','GLU_FAST_std',\
                'ALBMN_mean', 'ALBMN_median', 'ALBMN_std', 'BILI_mean', 'BILI_median', 'BILI_std', \
                'ALP_mean', 'ALP_median', 'ALP_std','ALT_mean', 'ALT_median', 'ALT_std' ]
    data3_sixteen = data3_1[sixteen_features].copy()
    print(data3_sixteen.shape)
    data3_clean = data3_sixteen.dropna()
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
else: ## use all features
    feat_count = "all"
    data3_clean = data3_1.dropna()

print("After dropping NULL values")
print(data3_clean.shape)
########## ~~~~~ ####################
print(data3_clean.columns)
data3_clean.to_csv("~/Desktop/data/data_two+/cleaned_summary_feature_two+_top_cancer_year_window_3.csv", index=False)

## split into 5-fold cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_folds = []
test_folds = []
idx = 0
for fold_index, (train_index, test_index) in enumerate(skf.split(data3_clean, data3_clean['CANCER'])):
    # Split the data into training and testing sets
    train_df = data3_clean.iloc[train_index]
    test_df = data3_clean.iloc[test_index]
    # Save to a separate DataFrame
    train_folds.append(train_df)
    test_folds.append(test_df)

    fname = "~/Desktop/data/data_two+/k-fold/summary_"+feat_count+"_feature_two+_top_cancers_train_fold_"+str(idx)+"_year_window_3.csv"
    train_df.to_csv(fname, index=False)
    fname = "~/Desktop/data/data_two+/k-fold/summary_"+feat_count+"_feature_two+_top_cancers_test_fold_"+str(idx)+"_year_window_3.csv"
    test_df.to_csv(fname, index=False)
    idx+=1