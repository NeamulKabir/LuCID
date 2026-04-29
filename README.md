# LuCID
LuCID: LongitUdinal Cancer risk prediction In Diabetes using data-centric artificial intelligence

## Data
To execute the code, we use the data file with all available laboratory measurements, their corresponding reference dates, cancer label, cancer incidence date along with demographic information age, sex (as numerical 0: female, 1: male), BMI, alcohol habit, smoking habit and unique reference key for each patient. In separate file, we have date of birth of each patient tagged with reference key which we use to calculate age and in another file we have ICD9 diagnosis code of each patient tagged with reference key which we use to identify individual cancers.
Below are the column names of the data file that we use to process and train our model using the available code in this repo:
| Reference Key | AGE | SEX | BMI | ALC | SMOKING | CANCER | CANCER_DATE | 
| :----------- | :------------: | ------------: | :----------- | :------------: | ------------: |:------------: | ------------: |
| HBA1C | HBA1C_dates | HB | HB_dates | WBC | WBC_dates | RBC | RBC_dates |

## Steps to execute the code

1. **Calculate age**: From the input feature file and date of birth of patients, we calculate age of the patients. As we divide our data using prediction window, we calculate age and other features based on our prediction-window approach.
2. **Calculate summary feature**: Then we calculate summary statistics features from the raw laboratory measurement values. We use a threhsold of at least five or more observations per laboratory measurements available per patient to calculate summary statistics features (mean, median and standard deviation) and also those that fall within our prediction window. If there are fewer than five observations for some patients, we put NULL values as their feature value and remove them before building our model.
3. **Top cancers**: We then find out the top ten prevalant cancers in our processed dataset. In the later part of the model, we use patients with these top ten cancers only.
4. **k-fold**: After dropping patients with null values and filtered for identified top ten cancers, we split the data into five equal folds using startified approach. later we use four folds for our training and one for testing, and repeat the steps five times. Finally, the average of the five-fold results is reported in the manuscript.
5. **Model training**: We train five ML models on the data anc compare their results. these models are Random Forest, XGboost, LightGBM, Logistic Regression and Linear SVM. As there is a huge imbalance between cancer and non-cancer group, we use class-weight parameter for all the models. For class-weight we use the class ratio as our parameter which is about 0.05 vs 0.95 for cancer and non-cancer group respectively. We also tune the decision threshold of the model using ROC AUC curve for a balanced performance. For this purpose, we use 10% of our training data as validation. For each model, we build for sepearate models, one for 3-year prediction window, one for 2-year prediction window, one for 1-year prediction window and one for 0-year prediction window. Finally, we take average of the four models probabibility which is then used as the within 3-year probability for the patient.
6. **Model tuning**:
7. **Model evaluation**:
8. **Visualization Dashboard**:
