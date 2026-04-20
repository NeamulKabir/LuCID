import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import shap


def init_model(model):
    if model == 'LogReg':
        clf= LogisticRegression(random_state=42, class_weight={1:0.95, 0:0.05}, max_iter=20000)
    elif model == 'GBM':
        clf = GradientBoostingClassifier(random_state=19)
    elif model == 'XGB':
        clf = XGBClassifier(random_state=19)
    elif model == 'LightGBM':
        clf = HistGradientBoostingClassifier(random_state=19, class_weight={1:0.95, 0:0.05})
    elif model == 'RF':
        clf = RandomForestClassifier(random_state=19, class_weight={1:0.95, 0:0.05})
    elif model == 'SVM_rbf':
        clf = SVC(kernel='rbf', class_weight={1:0.95, 0:0.05}, random_state=19, max_iter=100000, probability=True)
    elif model == 'SVM_linear':
        clf =SVC(kernel='linear', class_weight={1:0.95, 0:0.05}, random_state=19, max_iter=100000, probability=True)
    return clf

#### train a single model on a single window
def train_model(model, data, feature_list, year_window=3):
    clf = init_model(model)
    y_train = data['CANCER']
    # X_train = data3_tr.drop(['Reference Key', 'CANCER'], axis=1)
    X_train = data[feature_list]

    # X_train.drop('Reference Key', axis=1, inplace=True)
    if model == 'XGB' or model == "GBM":
        sample_weights = compute_sample_weight(class_weight={1:0.95, 0:0.05}, y=y_train) 
        clf.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        clf.fit(X_train, y_train)
    # fname = "../models/"+year_window+"/"+model+"_summary_model_"+year_window+"_top_cancers_28092025.pkl"
    # with open(fname,'wb') as f:
    #     pickle.dump(clf,f)
    # print("%s Model on %s window trainning finished!"%(model, year_window))
    return clf
#### get individual model performances
def get_performance(clf, data, feature_list):
    X_test = data[feature_list]
    
    y_pred = clf.predict(X_test)
    y_test = data['CANCER']
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    print(classification_report( y_test, y_pred))
    TPs = matrix[1][1]; FPs = matrix[0][1]; FNs = matrix[1][0]
    jaccard = TPs/(TPs+FPs+FNs)
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = (roc_auc_score(y_test, y_proba))
    print(jaccard, auc)

def calculateAvgPerformance(prob3, prob2, prob1, prob0, ref33, true_label):
    # calculate avg prob
    avg_prob = np.zeros(len(ref33),)

    # use default threshold 0.5 to make predictions
    thre = 0.5
    tp = 0; fp = 0; fn = 0; tn = 0
    for i in range(len(ref33)):
        avg_prob[i] = (prob3[i]+prob2[i]+prob1[i]+prob0[i])/4.0
        # print(avg_prob[i], true_label[i])
        pred_label = 1 if avg_prob[i] > thre else 0
        if true_label[i] ==1:
            if pred_label == true_label[i]:
                tp += 1; 
            else:
                fn +=1;
        else:
            if pred_label == true_label[i]:
                tn += 1
            else:
                fp +=1;
    auc = roc_auc_score(true_label, avg_prob)
    prec = tp/(tp+fp) if (tp+fp) >0 else 0

    return thre, tp, fn, fp, tn, tp/(tp+fn), tn/(tn+fp), prec, tp/(tp+fp+fn), (tp+tn)/(tp+tn+fn+fp), auc

#############################################


model = sys.argv[1] ### get model name
### selected four feature and demographic features
feature_list = ['AGE_new', 'Sex_n', 'BMI', 'ALC', 'SMOKING', 'HBA1C_mean','HBA1C_median','HBA1C_std',\
                'GLU_FAST_mean','GLU_FAST_median','GLU_FAST_std','TRIG_mean','TRIG_median','TRIG_std',\
                'LDL_CAL_mean','LDL_CAL_median','LDL_CAL_std']
### selected 15 feature and demographic features
### these features have less than 40% missing for five or more observations
# feature_list = ['AGE_new', 'Sex_n', 'BMI', 'ALC', 'SMOKING', 'HBA1C_mean','HBA1C_median','HBA1C_std',\
#                 'GLU_FAST_mean','GLU_FAST_median','GLU_FAST_std','TRIG_mean','TRIG_median','TRIG_std',\
#                 'LDL_CAL_mean','LDL_CAL_median','LDL_CAL_std', 'CREA_mean', 'CREA_median', 'CREA_std', \
#                 'POTASS_mean', 'POTASS_median', 'POTASS_std', 'SODIUM_mean', 'SODIUM_median', 'SODIUM_std',\
#                 'UREA_mean', 'UREA_median', 'UREA_std', 'CHOL_mean', 'CHOL_median', 'CHOL_std', \
#                 'HDL_mean', 'HDL_median', 'HDL_std', 'PROT_mean', 'PROT_median', 'PROT_std', \
#                 'ALBMN_mean', 'ALBMN_median', 'ALBMN_std', \
#                 'BILI_mean', 'BILI_median', 'BILI_std', 'ALP_mean', 'ALP_median', 'ALP_std',\
#                 'ALT_mean', 'ALT_median', 'ALT_std' ]


data2 = pd.read_csv("~/Desktop/data/data_two+/summary_features_two+_year_window_2.csv")
data1 = pd.read_csv("~/Desktop/data/data_two+/summary_features_two+_year_window_1.csv")
data0 = pd.read_csv("~/Desktop/data/data_two+/summary_features_two+_year_window_0.csv")


idx = 0
print(f'~~~~~~ Working with fold: {idx}~~~~~')
fname = "~/Desktop/data/data_two+/k-fold/summary_four_feature_two+_top_cancers_train_fold_"+str(idx)+"_year_window_3.csv"
data3_tr = pd.read_csv(fname)
fname = "~/Desktop/data/data_two+/k-fold/summary_four_feature_two+_top_cancers_test_fold_"+str(idx)+"_year_window_3.csv"
data3_ts = pd.read_csv(fname)

# store 3-year reference keys
train_ref = data3_tr['Reference Key'].tolist()
test_ref = data3_ts['Reference Key'].tolist()
# print("Training cancer distribution: ",data3_tr['CANCER'].value_counts())
# print("Test cancer distribution: ",data3_ts['CANCER'].value_counts())

# extract same data points as 3-year reference keys from 2-year, 1-year, 0-year data
data2_tr = data2[data2['Reference Key'].isin(train_ref)]
data2_ts = data2[data2['Reference Key'].isin(test_ref)]

data1_tr = data1[data1['Reference Key'].isin(train_ref)]
data1_ts = data1[data1['Reference Key'].isin(test_ref)]

data0_tr = data0[data0['Reference Key'].isin(train_ref)]
data0_ts = data0[data0['Reference Key'].isin(test_ref)]

### train a model

clf3 = train_model(model, data3_tr, feature_list, year_window=3)
# clf2 = train_model(model, data2_tr, feature_list, year_window=2)
# clf1 = train_model(model, data1_tr, feature_list, year_window=1)
# clf0 = train_model(model, data0_tr, feature_list, year_window=0)
print("~~~~ Model training completed!~~~~")


### add Reference Key to the feature list, as we need it to find it common ref keys in all four year window dataset
feature_list.append('Reference Key')
## prepare data for model prediction
d3 = data3_ts[feature_list].dropna()
d2 = data2_ts[feature_list].dropna()
d1 = data1_ts[feature_list].dropna()
d0 = data0_ts[feature_list].dropna()
# print(d33.shape, d22.shape, d11.shape, d00.shape)
common_ids = set(d3['Reference Key']) & set(d2['Reference Key']) & set(d1['Reference Key']) & set(d0['Reference Key'])

# d33 = d33.reset_index(drop=True)
d33 = d3[d3['Reference Key'].isin(common_ids)]
# print(d33.shape, d22.shape, d11.shape, d00.shape)
# true_label = data3_ts[data3_ts['Reference Key'].isin(common_ids)].set_index('Reference Key').reindex(ref33).reset_index()['CANCER'].tolist()

d33.drop('Reference Key', axis=1, inplace=True)
    
### calculate feature importance

X = data3_tr[feature_list].drop('Reference Key', axis=1, inplace=False)
X_test = data3_ts[feature_list].drop('Reference Key', axis=1, inplace=False)

# X_test.drop('CANCER', axis=1, inplace=True)
# Assuming `hist_gbc` is your trained model, and `X_test` has named columns

# Create the SHAP explainer
explainer = shap.TreeExplainer(clf3)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# --- Global Feature Importance (Mean Absolute SHAP) ---
# `shap.summary_plot` automatically sorts and plots features by importance
# shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
# plt.title("SHAP Global Feature Importance")
# plt.show()

# You can also use this to get the top features:
# Get the mean absolute SHAP value for each feature
mean_shap_values = abs(shap_values).mean(axis=0)

importance = abs(shap_values).mean(axis=0)
shap_relative_imp = importance / importance.sum()
# Create a DataFrame for easy sorting and display
shap_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': shap_relative_imp
}).sort_values('importance', ascending=False)

print("\nTop 10 Feature Importances (SHAP):")
print(shap_importance.head(17))

