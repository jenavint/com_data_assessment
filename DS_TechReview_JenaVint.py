
"""
Comcast Data Assessment
Author: Jena Vint
Date: May 10, 2017
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.ensemble as ske
import random
import pickle


# Read in dataset
file_path = r"C:\Users\jenad\Documents"
data_filename = "DS_Tech_Review_Dataset (1).txt"


data = pd.read_csv("{0}\{1}".format(file_path, data_filename), sep="|", header=0)

# preview the data
print data.shape
print data.head(5)

# Examine structure and completeness of the dataset, 
# clean data as needed to prep for modeling

# check for null (i.e. missing) values
print data.isnull().sum()

# look at age demographics
age_cols = [col for col in data.columns.values if "AGE" in col]
print data[age_cols].head()

# look at unique product combinations
for i in list(data["product"].unique()):
    print i
 
# drop columns if all if at least 99% of the values are null    
data.dropna(thresh=len(data)*0.01, axis=1, inplace=True)
print data.shape

print data.head(10)

# check remaining columns for unique values
for col in list(data.columns.values):
    print "{0}: {1}".format(col, data[col].unique())
    
# check how many null values remain
print data.isnull().sum()[0:25]
print data.isnull().sum()[25:50]
print data.isnull().sum()[50:75]
print data.isnull().sum()[75:]

# check data types
print data.dtypes[0:25]
print data.dtypes[25:50]
print data.dtypes[50:75]
print data.dtypes[75:]

# Null handling:
    
# MAJOR_CREDIT_CARD_LIF, convert nulls to 'N'
print "Before:", data.groupby("MAJOR_CREDIT_CARD_LIF")["product"].count()
data["MAJOR_CREDIT_CARD_LIF"] = np.where(data["MAJOR_CREDIT_CARD_LIF"].isnull(),"N", data["MAJOR_CREDIT_CARD_LIF"])
print "After:", data.groupby("MAJOR_CREDIT_CARD_LIF")["product"].count()

# for all other (numeric) columns, set null to 0 based on the assumption that nulls mean not applicable
data.fillna(0,inplace=True)

# drop columns where there is only one unique value (i.e. cannot use it for modeling)
# svod_bollywood_hits: [0], home_dw_takeo: [0], home_system_upgrades: [0], email_flag: [ 0.]
print data.shape
one_value_cols = ["svod_bollywood_hits", "home_dw_takeo", "home_system_upgrades", "email_flag"]
data.drop(one_value_cols, axis=1, inplace=True)
print data.shape

# see how many people are in the target (1), i.e. were cross-sold
print data.groupby("target")["product"].count()
# 7004 cross-sold out of 300k, so about 2.3%

# check for variable correlations against the target. 
corr = data.corr()
print corr["target"].nlargest(25)
print corr["target"].nsmallest(25)

# very low correlations, check for any pattern differences in target group (1) vs. non target group (0)
remove_list = ["target", "product", "MAJOR_CREDIT_CARD_LIF"] + age_cols
hist_var_list = [col for col in data if col not in remove_list]
    
for var in hist_var_list:
    t0 = data[data.target==0][var]
    t1 = data[data.target==1][var]
    n, bins, patches = plt.hist([t0,t1], label=["t0","t1"], bins=10)
    plt.title(var, fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

# video_addon_music_choice
# prev_number_of_products
# video_days_on_books

# create dummy variables for object datatypes

data_mod = pd.get_dummies(data, columns=["product", "MAJOR_CREDIT_CARD_LIF"])
pcols = [col for col in data_mod if "product" in col]
print pcols
print data_mod[pcols].head(10)

# Modeling Process (1st iteration, no parameter tuning yet)

# Split data into training and test, random sample of 70% of customers = train, remaining 30% = test
random.seed = 1234
L = range(0, data_mod.shape[0])
random.shuffle(L)
n = int(0.7*data_mod.shape[0])

# create train as random 70% of customers, test as remaining 30%
train = data_mod.iloc[L[0:n], :].copy() 
test = data_mod.iloc[L[n:], :].copy()
# save train and test datasets to a pickle file
pickle.dump(train, open("{0}\{1}".format(file_path, "train"), "wb"))
pickle.dump(test, open("{0}\{1}".format(file_path, "test"), "wb"))

# uncomment and run below lines if you need to reload train and test data
train = pickle.load(open("{0}\{1}".format(file_path, "train"), "rb"))
test = pickle.load(open("{0}\{1}".format(file_path, "test"), "rb"))

# print shape and % of target in the train and test datasets
print "{0}: {1}".format("Number of rows in training data", train.shape[0])
print "{0}: {1}".format("Number of rows in test data", test.shape[0])
print "{0}: {1}".format("Percent of target in training data", (train[train.target == 1].shape[0]*1.0)/train.shape[0])
print "{0}: {1}".format("Percent of target in test data", (test[test.target == 1].shape[0]*1.0)/test.shape[0])

# create list of features (currently set to all independent variables)
feature_list = [col for col in data_mod if col not in ["target"]]
print feature_list
print "{0}: {1}".format("Number of features", len(feature_list))

# x vars and y var (target) data structures
x_train = train[feature_list].values
y_train = train["target"].values

x_test = test[feature_list].values
y_test = test["target"].values               

# Fit model on training dataset (no tuning)
rfclass = ske.RandomForestClassifier(n_estimators=200, random_state=12345)
rfclass.fit(x_train, y_train)

from sklearn.externals import joblib
joblib.dump(rfclass, "{0}\{1}".format(file_path, "rfclass.pkl"))

# Score model on test dataset and examine model performance metrics
# Accuracy
print "Training Accuracy: ", rfclass.score(x_train, y_train)
print "Test Accuracy: ", rfclass.score(x_test, y_test)

# train and test y predictions
y_train_pred = rfclass.predict(x_train)
y_test_pred = rfclass.predict(x_test)
# train and test predicted probabilities
y_train_pred_prob = rfclass.predict_proba(x_train)
y_test_pred_prob = rfclass.predict_proba(x_test)

# Calculate Area Under Curve (AUC)
from sklearn.metrics import roc_auc_score, auc

print("Training ROC AUC: %.3f" %roc_auc_score(y_train, y_train_pred_prob[:,1]))
print("Test ROC AUC: %.3f" %roc_auc_score(y_test, y_test_pred_prob[:,1]))

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_prob[:,1], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, lw=1, label="ROC (area = %0.2f)" %roc_auc)
plt.plot([0,1 ], [0, 1], linestyle="--", color=(0.6, 0.6, 0.6), label="random guessing")
plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=":", color="black", label="perfect performance")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("Receiver Operator Characteristic Curve")
plt.legend(loc="lower right")
plt.show()

# Recall
from sklearn.metrics import recall_score
print("Training Recall Score: %.4f" %recall_score(y_train, y_train_pred))
print("Test Recall Score: %.4f" %recall_score(y_test, y_test_pred))

# Precision
from sklearn.metrics import precision_score
print("Training Precision Score: %.4f" %precision_score(y_train, y_train_pred))
print("Test Precision Score: %.4f" %precision_score(y_test, y_test_pred))

# confusion matrices

# lift charts

# Look at variable importance
feature_name = np.array(feature_list)
feat_ind = np.argsort(rfclass.feature_importances_)[::-1]
feat_imp = rfclass.feature_importances_[feat_ind]

feature_importance = pd.DataFrame({"Feature_Ind":feat_ind,
                                   "Feature_Name":feature_name,
                                   "Feature_Imp":feat_imp})
    
feature_importance.sort_values(by="Feature_Imp", ascending=False, inplace=True)
feature_importance.reset_index(drop=True, inplace=True)

pd.set_option("display.max_rows", 100)
print feature_importance[0:49][["Feature_Imp", "Feature_Name"]]

# See if we can simplify the model or improve it 



