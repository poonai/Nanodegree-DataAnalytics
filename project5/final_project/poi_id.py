#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

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

### Task 2: Remove outliers
# removing feature which has NaN value more than 100
total_feature.remove('deferred_income')
total_feature.remove('director_fees')
total_feature.remove('loan_advances')
total_feature.remove('restricted_stock_deferred')
total_feature.remove('email_address')
total_feature.remove('poi')

### Task 3: Create new feature(s)


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
### Store to my_dataset for easy export below.
my_features = total_feature
k_best = SelectKBest(k=5)
k_best.fit(features, labels)
results_list = zip(k_best.get_support(), my_features[1:])
best_features = []
for i in range(0,len(results_list)):
    if results_list[i][0] == True:
        best_features.insert(0,results_list[i][1])
print best_features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
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
    X_train, X_test, y_train, y_test = train_test_split(feature, lable,train_size=.7)
    clf= GridSearchCV(clf, params)
    clf.fit(X_train, y_train)
    print "best estimator"
    print clf.best_estimator_
    y_pred = clf.predict(X_test)
    precision = []
    recall = []
    print classification_report(y_test,y_pred)
'''    for i in range(0,len(y_test)):
        #precision.append(precision_score(y_test[i],y_pred[i]))
        #recall.append(recall_score(y_test[i],y_pred[i]))
        print precision_score(y_test[i],y_pred[i])
    print mean(recall)
    print mean(precision)
 '''
best_features.insert(0,'poi')
#clf.fit()
lsvc=svm.LinearSVC(penalty='l2')
evaluate_clf(lsvc,[{'C':[1,10,100]}],features,labels)
from sklearn.ensemble import RandomForestClassifier
rforest = RandomForestClassifier(n_estimators = 10)
evaluate_clf(rforest,[{'min_samples_leaf':[5,10,20]}],features,labels)
rforest = RandomForestClassifier(n_estimators = 10,min_samples_leaf = 5)
dump_classifier_and_data(rforest, my_dataset, best_features)
