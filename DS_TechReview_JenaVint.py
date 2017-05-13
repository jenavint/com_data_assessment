
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

"""
# make sure indicator variables have values of 0 or 1
ind_var_list = 
for var in ind_var_list:
    print data.groupby(var)["product"].count()
    if var == "infin_ind":
        data[var] = np.where(data[var] == 5, 1, 0) # convert 5 to 1 and 3 to 0
    data[var] = np.where(data[var] <> 0, 1,0)
    print data.groupby(var)["product"].count()
"""
# see how many people are in the target (1), i.e. were cross-sold
print data.groupby("target")["product"].count()
# 7004 cross-sold out of 300k, so about 2.3%

# check for variable correlations against the target. 
corr = data.corr()
print corr["target"].nlargest(25)
print corr["target"].nsmallest(25)

# very low correlations, check for any pattern differences in target group (1) vs. non target group (0)

non_age_var_list = [col for col in data if col not in [["target"] + age_cols]]
for var in non_age_var_list:
    t0 = data[data.target==0][var]
    t1 = data[data.target==1][var]
    n, bins, patches = plt.hist([t0,t1], label=["t0","t1"])
    plt.title(var, fontsize=14)
    plt.legend(fontsize=12)

"""
var = "tellop_ind"
t0 = data[data.target==0][var]
t1 = data[data.target==1][var]
n, bins, patches = plt.hist([t0,t1], label=["t0","t1"])
plt.title(var, fontsize=14)
plt.legend(fontsize=12)
"""

# Modeling Process (1st iteration, no parameter tuning yet)

# Split data into training and test, random sample of 70% of customers = train, remaining 30% = test
random.seed = 1234
L = range(0, data.shape[0])
random.shuffle(L)
n = int(0.7*data.shape[0])
# create train as random 70% of customers, test as remaining 30%
train = data.iloc[L[0:n], :].copy() 
test = data.iloc[L[n:], :].copy()
# save train and test datasets to a pickle file
pickle.dump(train, open("{0}\{1}".format(file_path, "train"), "Wb"))
pickle.dump(test, open("{0}\{1}".format(file_path, "test"), "Wb"))

# print shape and % of target in the train and test datasets
print "{0}: {1}".format("Number of rows in training data", train.shape[0])
print "{0}: {1}".format("Number of rows in test data", test.shape[0])
print "{0}: {1}".format("Percent of target in training data", (train[train.target == 1].shape[0]*1.0)/train.shape[0])
print "{0}: {1}".format("Percent of target in test data", (test[test.target == 1].shape[0]*1.0)/test.shape[0])

# create list of features (currently set to all independent variables)
feature_list = [col for col in data if col not in ["target"]]
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



# Score model on test dataset and examine model performance metrics

# Look at variable importance

# See if we can simplify the model or improve it 



