import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import scale

train = pd.read_csv('data/train.csv')
X = train.drop(['Response', 'Id'], axis=1)
y = train['Response']
X_numerical = X[[
    'Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 
    'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 
    'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5'
]]
X_categorical = train[[
    'Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 
    'Product_Info_6', 'Product_Info_7', 
    'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 
    'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 
    'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 
    'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 
    'Insurance_History_4', 'Insurance_History_7', 
    'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 
    'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 
    'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 
    'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 
    'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 
    'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 
    'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 
    'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 
    'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 
    'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 
    'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 
    'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 
    'Medical_History_39', 'Medical_History_40', 'Medical_History_41',
    
    'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 
    'Medical_History_24', 'Medical_History_32', 'Response'
]]
mean_encodings = []
for col in X_categorical.columns[:-1]:
    mean_encoding = X_categorical.fillna('Unknown').groupby(col)['Response'].transform('mean').rename(col)
    mean_encodings.append(mean_encoding)
mean_encodings_frame = pd.concat(mean_encodings, axis=1)

median_encodings = []
for col in X_categorical.columns[:-1]:
    median_encoding = (X_categorical
                       .fillna('Unknown')
                       .groupby(col)['Response']
                       .transform('median')
                       .rename(col))
    median_encodings.append(median_encoding)
median_encodings_frame = pd.concat(median_encodings, axis=1)

XX = pd.concat([
    X_numerical.fillna(X_numerical.mean()), 
    mean_encodings_frame
], axis=1)

cross_val_score(RandomForestRegressor(), scale(XX), y, cv=5)
