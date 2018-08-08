#!/usr/bin/python

import sys
import pickle
import re
import pandas as pd 
import numpy as np
import seaborn as sns
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'bonus', 'total_payments', 'total_stock_value',
    'exercised_stock_options', 'from_poi_to_this_person', 'to_messages',
    'from_messages', 'from_this_person_to_poi', 'long_term_incentive'] 
    #'deferral_payments', 'total_payments', 'loan_advances',
	#'bonus', 'restricted_stock_deferred', 'deferred_income', 
    #'total_stock_value', 'expenses', 'exercised_stock_options', 
    #'other', 'long_term_incentive', 'restricted_stock', 
	#'director_fees', 'to_messages', 'email_address', 
    #'from_poi_to_this_person', 'from_messages',
	#'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi'] 
    # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#converting data to pandas dataframe
data_df = pd.DataFrame.from_dict(data_dict, orient='index')
print(data_df.shape)
print(data_df.head())

#Replace NaN with Numpy's NaN's
data_df.replace(to_replace='NaN', value= np.nan, inplace = True)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)

poi_dataset, n_email, n_sal, total_payments, poi_total_pay =0,0,0,0,0

for i in data_dict:
    if data_dict[i]["poi"]:
        poi_dataset += 1
        if data_dict[i]["total_payments"]=="NaN":
            poi_total_pay += 1
    if data_dict[i]["email_address"]!="NaN":
        n_email += 1
    if data_dict[i]["salary"] != "NaN":
        n_sal +=1
    if data_dict[i]["total_payments"]=="NaN":
        total_payments += 1

print "POI in Dataset: ", poi_dataset

poi_all=0
with open("poi_names.txt") as f:
    content=f.readlines()
for line in content:
    if re.match(r'\((y|n)\)', line):
        poi_all+=1
print "POI ALL: ", poi_all

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

for point in data:
	plt.scatter(point[1], point[2])
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show()
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

"""features = data_df
labels = features.pop('poi')
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
"""
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

"""clf1 = GaussianNB()
clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)
"""
clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


from sklearn.metrics import accuracy_score
#print 'Accuracy of GaussianNB: ', accuracy_score(pred1, labels_test)
print 'Accuracy of DecisionTreeClassifier: ', accuracy_score(pred, labels_test)
#print 'Accuracy of RandomForestClassifier: ', accuracy_score(pred3, labels_test)
#print "Accuracy Adaboost: ", accuracy_score(labels_test, ab_pred)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
    #train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)