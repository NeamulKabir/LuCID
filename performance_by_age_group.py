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
from sklearn import metrics
import sys


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
        clf = SVC(kernel='rbf', class_weight={1:0.95, 0:0.05}, random_state=19, max_iter=50000, probability=True)
    elif model == 'SVM_linear':
        clf = SVC(kernel='linear', class_weight={1:0.95, 0:0.05}, random_state=19, max_iter=50000, probability=True)
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

def getPrediction(prob3, prob2, prob1, prob0, ref33, true_label):
    # calculate avg prob
    avg_prob = np.zeros(len(ref33),)

    # use default threshold 0.5 to make predictions
    thre = 0.5
    pred_labels = list()
    for i in range(len(ref33)):
        avg_prob[i] = (prob3[i]+prob2[i]+prob1[i]+prob0[i])/4.0
        
        cur_pred = 1 if avg_prob[i] > thre else 0
        # print(avg_prob[i], true_label[i], cur_pred)
        pred_labels.append(cur_pred)

    return pred_labels, avg_prob

#############################################


model = sys.argv[1] ### get model name
number_of_age_group = sys.argv[2] ### divide into either 2 or 6 groups
### selected four feature and demographic features
feature_list = ['AGE_new', 'Sex_n', 'BMI', 'ALC', 'SMOKING', 'HBA1C_mean','HBA1C_median','HBA1C_std',\
                'GLU_FAST_mean','GLU_FAST_median','GLU_FAST_std','TRIG_mean','TRIG_median','TRIG_std',\
                'LDL_CAL_mean','LDL_CAL_median','LDL_CAL_std']


data2 = pd.read_csv("~/Desktop/data/data_two+/summary_features_two+_year_window_2.csv")
data1 = pd.read_csv("~/Desktop/data/data_two+/summary_features_two+_year_window_1.csv")
data0 = pd.read_csv("~/Desktop/data/data_two+//summary_features_two+_year_window_0.csv")


for idx in range(1):
    print(f'~~~~~~ Working with fold: {idx} ~~~~~')
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
    clf2 = train_model(model, data2_tr, feature_list, year_window=2)
    clf1 = train_model(model, data1_tr, feature_list, year_window=1)
    clf0 = train_model(model, data0_tr, feature_list, year_window=0)
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
    ref33 = d33['Reference Key'].tolist()

    d22 = d2[d2['Reference Key'].isin(common_ids)].set_index('Reference Key').reindex(ref33).reset_index()
    d11 = d1[d1['Reference Key'].isin(common_ids)].set_index('Reference Key').reindex(ref33).reset_index()
    d00 = d0[d0['Reference Key'].isin(common_ids)].set_index('Reference Key').reindex(ref33).reset_index()
    # print(d33.shape, d22.shape, d11.shape, d00.shape)
    true_label = data3_ts[data3_ts['Reference Key'].isin(common_ids)].set_index('Reference Key').reindex(ref33).reset_index()['CANCER'].tolist()

    d33.drop('Reference Key', axis=1, inplace=True)
    d22.drop('Reference Key', axis=1, inplace=True)
    d11.drop('Reference Key', axis=1, inplace=True)
    d00.drop('Reference Key', axis=1, inplace=True)

    prob3 = (clf3.predict_proba(d33)[:, 1])
    prob2 = (clf2.predict_proba(d22)[:, 1])
    prob1 = (clf1.predict_proba(d11)[:, 1])
    prob0 = (clf0.predict_proba(d00)[:, 1])

    # calculate avg prob and get predicted labels
    pred_labels, avg_prob = getPrediction(prob3, prob2, prob1, prob0, ref33, true_label)

    ## create a result df with Predicted Labels and True Labels
    d33_result = d33.copy()
    d33_result['Pred'] = pred_labels
    d33_result['True_Label'] = true_label
    d33_result['Pred_Probability'] = avg_prob

    print(d33_result['Pred'].value_counts(), d33_result['True_Label'].value_counts())

    # Step 1. Define two or six groups based on input
    if number_of_age_group == "6":
        bins = np.linspace(40, 100, 7)   # 7 edges → 6 bins
        labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
        d33_result["Age_group"] = pd.cut(d33_result["AGE_new"], bins=bins, labels=labels, right=False)
    elif number_of_age_group == "2":
        d33_result["Age_group"] = d33_result["AGE_new"].apply(lambda x: "< 55 years" if x < 55 else "≥ 55 years")
    else:
        print("Sorry! Wrong number of age group. Provide either 2 or 6 as input.")
        continue

    d33_result.to_csv("~/Desktop/data/data_two+/age_group_df_for_plot.csv", index=False)
    # Step 2. Count positives/negatives per group
    group_stats = d33_result.groupby("Age_group")["True_Label"].agg(["sum", "count"]).reset_index()
    group_stats.rename(columns={"sum": "n_positive", "count": "n_total"}, inplace=True)
    group_stats["n_negative"] = group_stats["n_total"] - group_stats["n_positive"]

    # Step 3. Normalize proportions across total positives/negatives
    total_pos = group_stats["n_positive"].sum()
    total_neg = group_stats["n_negative"].sum()
    group_stats["positive_ratio"] = group_stats["n_positive"] / total_pos
    group_stats["negative_ratio"] = group_stats["n_negative"] / total_neg

    print(group_stats)

    # Step 4: calculate performance for each age group
    results = []

    for group, subset in d33_result.groupby("Age_group"):
        y_true = subset["True_Label"]
        y_pred = subset["Pred"]
        y_proba = subset['Pred_Probability']

        recall = metrics.recall_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = np.nan
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        results.append({
            "Age Group": group,
            "Sensitivity": recall,
            "Specificity": specificity,
            "Precision": precision,
            "ROC AUC": auc
        })

    perf_df = pd.DataFrame(results)
    print(perf_df)

    feature_list.remove('Reference Key')
