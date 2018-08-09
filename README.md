# Identify Enron Fraud
 
## Introduction
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.

## Goal
The goal of this project is to build a person of interest (POI, which means an individual who was indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity) identifier based on financial and email data made public as a result of the Enron scandal. Machine learning is an excellent tool for this kind of classification task as it can use patterns discovered from labeled data to infer the classes of new observations.

Our dataset combines the public record of Enron emails and financial data with a hand-generated list of POI’s in the fraud case.

## Dataset

### Features
The features included in the dataset can be divided in three categories, Salary Features, Stock Features and Email Features.

Financial features: [salary, deferral_payments, total_payments, loan_advances, bonus, restricted_stock_deferred, deferred_income, 

Stock features: total_stock_value, expenses, exercised_stock_options, other, long_term_incentive, restricted_stock, director_fees 

Email features: to_messages, email_address, from_poi_to_this_person, from_messages, from_this_person_to_poi, shared_receipt_with_poi 

POI label: poi 


## Files
1. enron_project.ipynb: Jupyter notebook. Runs final feature selection, feature scaling, various classifiers (optional) and their results. Finally, dumps classifier, dataset and feature list so anyone can check results.

2. poi_id.py: Main file. 

3. tester.py: Functions for validation and evaluation of classifier, dumping and loading of pickle files.

4. my_classifier.pkl: Pickle file for final classifier from poi_id.py.

5. my_dataset.pkl: Pickle file for final dataset from poi_id.py.

6. my_feature_list.pkl: Pickle file for final feature list from poi_id.py.
 
