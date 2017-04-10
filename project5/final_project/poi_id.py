
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary']
# You will need to use more features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

my_dataset = data_dict


# missing value count
total_feature = my_dataset['ALLEN PHILLIP K'].keys()
feature_count_dict = {}
for val in range(0,len(total_feature)):
    feature_count_dict[total_feature[val]] = 0
poi_count = 0
for val in my_dataset:
    for key in my_dataset[val]:
        if key == 'poi':
            if my_dataset[val][key] == True:
                poi_count+=1
        if my_dataset[val][key] == 'NaN':
            feature_count_dict[key]+=1
# total length of data
print len(my_dataset)
# true poi count
print poi_count
# feature nan count

print feature_count_dict

total_feature.remove('deferred_income')
total_feature.remove('director_fees')
total_feature.remove('loan_advances')
total_feature.remove('restricted_stock_deferred')
total_feature.remove('email_address')
total_feature.remove('poi')

del my_dataset['TOTAL']

for name in my_dataset:
    if (all([
        my_dataset[name]['from_messages'] != 'NaN',
        my_dataset[name]['from_this_person_to_poi'] != 'NaN',
        my_dataset[name]['to_messages'] != 'NaN',
        my_dataset[name]['from_poi_to_this_person'] != 'NaN'
    ])):
        my_dataset[name]['from_fraction'] = float(my_dataset[name]['from_poi_to_this_person']) / float(my_dataset[name]['to_messages'])
        my_dataset[name]['to_fraction'] = float(my_dataset[name]['from_this_person_to_poi']) / float(my_dataset[name]['from_messages'])
    else:
        my_dataset[name]['from_fraction'] = 0
        my_dataset[name]['to_fraction'] = 0

total_feature.insert(0,'from_fraction')
total_feature.insert(0,'to_fraction')
my_features = total_feature
data = featureFormat(my_dataset, my_features, sort_keys = True)

labels, features = targetFeatureSplit(data)

my_features = total_feature
k_best = SelectKBest(k=7)
k_best.fit(features, labels)
results_list = zip(k_best.get_support(), my_features[1:],k_best.scores_)
best_features = []
for i in range(0,len(results_list)):
    if results_list[i][0] == True:
        best_features.insert(0,results_list[i][1])
print results_list

data = featureFormat(my_dataset, best_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,classification_report
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import svm
def evaluate_clf(clf,params,feature,lable):
    X_train, X_test, y_train, y_test = train_test_split(feature, lable)
    clf= GridSearchCV(clf, params)
    clf.fit(X_train, y_train)
    print "best estimator"
    print clf.best_estimator_
    y_pred = clf.predict(X_test)
    precision = []
    recall = []
    #print classification_report(y_test,y_pred)
    print 'precision score {}'.format(precision_score(y_test,y_pred,average = 'micro'))
    print 'recall score {}'.format(recall_score(y_test,y_pred,average = 'micro'))
    #print mean(recall)
    #print mean(precision)
best_features.insert(0,'poi')
#clf.fit()
lsvc=svm.LinearSVC(penalty='l2')
# C parameter is used for how much classification curve has to make
#evaluate_clf(lsvc,[{'C':[1,10,100]}],features,labels)
from sklearn.ensemble import RandomForestClassifier
rforest = RandomForestClassifier(n_estimators = 3,min_samples_leaf = 3)
logreg = LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42)
# min samples leaf is used to determine how much sample minimum requrired to go further split
clf = GaussianNB()
evaluate_clf(rforest,[{'min_samples_leaf':[3,5,10,13,15],'min_samples_split':[.1,0.2,.5]}],features,labels)
rforest = RandomForestClassifier(n_estimators = 3,min_samples_leaf = 15, min_samples_split = .2)
print best_features

dump_classifier_and_data(rforest, my_dataset, best_features)
