# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:32:44 2023

@author: Simon Gebraad
"""

# import some useful libraries
import enum
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import itertools

#import the different classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

#import statistics tests
from scipy.stats import f_oneway
from scipy.stats import kstest
from scipy.stats import kruskal
from scipy.stats import wilcoxon

#import the raw data as a pandas dataframe 
data_path = "C:/Users/Gebra/OneDrive - Delft University of Technology/Master/IER/EyeTracking-data"

participants = 24
conditions = ["Control","Noise","NPC","Second_Task","4_Combined"]

#function that counts fixations and adds them to the dataframe
def blinks(dataframe, duration_threshold):
    window = []
    time = []
    for i in range(dataframe.shape[0]):
        if dataframe['pupil_diam_L'][i] and dataframe['pupil_diam_R'][i] == -1:
            window.append(i)
            time.append(dataframe['timestamp'][i])
            duration = (int(time[-1][-13:-11])- int(time[0][-13:-11]))*3600 + (int(time[-1][-10:-8])- int(time[0][-10:-8]))*60 + float(time[-1][-7:])- float(time[0][-7:]) #convert the different elements of the timestamps (string) to numbers and calculate the duration
        else:
            if not window:
                continue
            elif duration <= duration_threshold:
                window =[]
                time =[]
            elif duration > duration_threshold:
                dataframe['blink'][i-int(0.5*len(window))] = duration
                window =[]
                time = []
                
#function that squares all elements in a list
def square(list):
    return [i ** 2 for i in list]    

#function that calculates the RR intervals. RR intervals are the times between subsequent blinks.
def rr_intervals(data):
    first = True
    indices = []
    RR_interval = []   
    
    for i, count in data['blink'].iteritems():
        if pd.notnull(count):
            if first == True:
                print(count)
                first = False
                indices.append(i)
 
            else:
                previous = indices[-1]
                indices.append(i)
                length = (int(data['timestamp'][i][-13:-11])- int(data['timestamp'][previous][-13:-11]))*3600 + (int(data['timestamp'][i][-10:-8])- int(data['timestamp'][previous][-10:-8]))*60 + (float(data['timestamp'][i][-7:])- float(data['timestamp'][previous][-7:])) #convert the different elements of the timestamps (string) to numbers and calculate the duration
                RR_interval.append(length)        
    return RR_interval

#function that calculates the RMSSD based on a list of RR intervals
def rmssd(RR_interval):
    return np.sqrt(sum(square(RR_interval))/len(RR_interval))

#function that tests a classifier and returns average auc and accuracy scores
def test_classifier(X, y, clf, N):
     #empty lists to store scores
     auc_scores = []
     accuracy_scores = []
     
     #loop through every classifier multiple times to ensure the test/train split is averaged
     for i in range(N):
         # Split the data into training and test sets
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
         
         #fit the classifier
         clf.fit(X_train, y_train)
         
         #predict on test set
         y_pred = clf.predict(X_test)
         accuracy = accuracy_score(y_test, y_pred)
         auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
         #auc = accuracy_score(y_test, y_pred)
         auc_scores.append(auc)
         accuracy_scores.append(accuracy)
     avg_auc = sum(auc_scores)/len(auc_scores)
     avg_accuracy = sum(accuracy_scores)/len(accuracy_scores)
     
     #print for tests
     print(avg_auc)
     print(avg_accuracy)
     return avg_auc, avg_accuracy
#%%

#define empty dictonaries to store the data for plotting
blink_rate = {}
variabilities = {}
blink_rate_un = {} #unnormalized
variabilities_un = {} #unnormalized

#define empty lists to store the data for the classifiers
classifier_br = []
classifier_var = []
classifier_br_un = [] #unnormalized
classifier_var_un = [] #unnormalized
classifier_class = []

#add the keys to the dictonaries
for condition in conditions:
    blink_rate["blinks_" + condition] = []
    variabilities["variabilities_" + condition] = []
    blink_rate_un["blinks_" + condition] = []
    variabilities_un["variabilities_" + condition] = []
    
#loop through all participants   
for p in range(1,(participants+1)):
    
    #define lists used for normalization
    br = []
    variab = []
    
    #loop through all conditions
    for condition in conditions:
        
        #read data
        data = pd.read_csv(data_path + '/P' + str(p) +'/Eyerecording_Test_' + condition + ".csv",sep=';')
        data.columns = ['frame', 'timestamp', 'openness_L', 'openness_R', 'pupil_diam_L', 'pupil_diam_R', 'blink']
        
        #determine blinks and RR_intervals
        blinks(data, 0.)
        RR_interval = rr_intervals(data)
        
        #save the data for later use
        data.to_csv(data_path + '/P' + str(p) +'/Eyerecording_Test_' + condition + '_blinks' + ".csv",sep=';')
        
        #count blinks
        blink_count = data['blink'].count()
        
        #extract experiment duration. Some formatting is required to convert the string time stamp to a number
        experiment_duration = (int(data['timestamp'][data.shape[0]-1][-13:-11])- int(data['timestamp'][0][-13:-11]))*60 + (int(data['timestamp'][data.shape[0]-1][-10:-8])- int(data['timestamp'][0][-10:-8])) + (float(data['timestamp'][data.shape[0]-1][-7:])- float(data['timestamp'][0][-7:]))/60
        
        #calculate RMSSD
        RMSSD = rmssd(RR_interval)
        
        #some printing for checks
        print(condition,p)
        print(experiment_duration)
        print(blink_count)
        print(blink_count/experiment_duration)
        print(RMSSD)
        
        #append lists for normalization
        br.append(blink_count/experiment_duration)
        variab.append(RMSSD)
        
        #append dictonaries with unnormalized data
        blink_rate_un["blinks_" + condition].append(blink_count/experiment_duration)
        variabilities_un["variabilities_" + condition].append(RMSSD)
        classifier_br_un.append(blink_count/experiment_duration)
        classifier_var_un.append(RMSSD)
    
    #loop through all conditions and normalize the data for each participant
    for i, condition in enumerate(conditions):
        norm_br = (br[i]-min(br))/(max(br)-min(br))
        norm_variab = (variab[i]-min(variab))/(max(variab)-min(variab))
        blink_rate["blinks_" + condition].append(norm_br)
        variabilities["variabilities_" + condition].append(norm_variab)
        classifier_br.append(norm_br)
        classifier_var.append(norm_variab)
        classifier_class.append(i)

#%%
#save normalized and unnormalized data for later use
classifier_data = pd.DataFrame({'Blink_rate':classifier_br, 'Variability':classifier_var, 'Class':classifier_class})
classifier_data.to_csv(data_path + '/classes.csv')

classifier_data_un = pd.DataFrame({'Blink_rate':classifier_br_un, 'Variability':classifier_var_un, 'Class':classifier_class})
classifier_data_un.to_csv(data_path + '/classes_un.csv')
#%%
#create binary data, mainly used for some testing
df = pd.read_csv(data_path + '/classes.csv')
df.columns = ['Index','Blink_rate','Variability','Class']
df = df.drop(df[(df.Class == 1) | (df.Class == 2) | (df.Class == 3)].index)
df.to_csv(data_path + '/binary.csv',index=False)

#%%
#read saved data
classifier_data = pd.read_csv(data_path + '/classes.csv')
classifier_data.columns = ['Index','Blink_rate','Variability','Class']

#define classifiers and features
classifiers = {'KNN':KNeighborsClassifier(n_neighbors=1), 'RF':RandomForestClassifier(max_depth=3, random_state=0), 'AB': AdaBoostClassifier(n_estimators=300, random_state=0), 'SVC': svm.SVC(probability=True, random_state=0), 'MLP': MLPClassifier(random_state=0, max_iter=500), 'SL': LogisticRegression(random_state=0), 'LDA': LinearDiscriminantAnalysis(),'GPF': GaussianProcessClassifier(kernel=1.0 * RBF(1.0),random_state=0) }
features = {'BR+BRV':classifier_data.drop(['Class', 'Index'], axis=1), 'BR': np.array(classifier_data['Blink_rate']).reshape(-1,1),'BRV': np.array(classifier_data['Variability']).reshape(-1,1)}

#empty dictionaries to store the scores
aucs = {}
accuracies = {}

# create keys for dictonaries
for name, feature in features.items():
    aucs[name] = []
    accuracies[name] = []

#loop through all features
for name, feature in features.items():
    # Split the data into features (X) and target (y)
    X = feature
    y = classifier_data['Class']
   
    #loop through all classifiers
    for classifier in classifiers.values():
        # define classifier
        clf = classifier
        
        #clf.fit(X, y)
        #y_pred = clf.predict(X)
        #accuracy = accuracy_score(y, y_pred)
        #auc = roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')
        #aucs[name].append(auc)
        
        #test each classifier 15 times to average out test/train split
        avg_auc, avg_accuracy = test_classifier(X, y, clf, 15)
        aucs[name].append(avg_auc)
        accuracies[name].append(avg_accuracy)
        
    # calculate the mean of all classifiers for a certain feature
    mean = sum(aucs[name])/len(aucs[name])
    aucs[name].append(mean)

print(aucs)
#%%
#plot the auc scores of the different classifiers


classifiers['Mean'] = None
x = np.arange(9)  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots()

for attribute, measurement in aucs.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, [round(elem, 2) for elem in measurement], width, label=attribute)
    ax.bar_label(rects, padding=5, rotation = 90)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUC')
ax.set_title('AUC per classifier (binary)')
ax.set_xticks(x + width)
ax.set_xticklabels(classifiers.keys())
ax.legend()
ax.set_ylim(0, 1)

plt.show()

#%%
#create a boxplot of the blinkrate and perform statistical tests

boxplot_data = []

#loop through each condition and check for normality
for condition in conditions:
    boxplot_data.append(blink_rate["blinks_" + condition])
    
    # check for normality
    stat, p = kstest(blink_rate_un["blinks_" + condition], 'norm')
    avg = sum(blink_rate["blinks_" + condition])/len(blink_rate["blinks_" + condition])
    std = np.std(blink_rate["blinks_" + condition])
    
    #print results
    print(condition,'mean: ', avg,' std: ', std,' p of normal: ', p)

fig = plt.figure(figsize =(5, 3))
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
ax.set_ylabel('Normalized blink rate')
ax.set_title('Boxplot of blink rate')
ax.set_xticklabels(conditions)
# Creating plot
bp = ax.boxplot(boxplot_data)
 
# show plot
plt.show()

#print result of kruskal wallis test
print(kruskal(boxplot_data[0],boxplot_data[1],boxplot_data[2],boxplot_data[3],boxplot_data[4]))
#%%
#create a table where we compare each condition of blink rate through a wilcoxon signed rank test

table = []
for i in range(len(boxplot_data)):
    table_row = []
    for j in range(len(boxplot_data)):
        if i == j:
            wilcox = 1.
        else:
            stat, wilcox = wilcoxon(boxplot_data[i],boxplot_data[j])
        table_row.append(wilcox)
    table.append(table_row)

fig, ax = plt.subplots(figsize=(15, 10))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame(table)

ax.table(cellText=df.values, colLabels=df.columns, loc='center')

fig.tight_layout()

plt.show()

print(wilcoxon(boxplot_data[0],boxplot_data[4]))
#%%
#create a boxplot of the blinkrate variability and perform statistical tests

boxplot_data = []

#loop through each condition and check for normality
for condition in conditions:
    boxplot_data.append(variabilities["variabilities_" + condition])
    
    # check for normality
    stat, p = kstest(variabilities_un["variabilities_" + condition], 'norm')
    avg = sum(variabilities["variabilities_" + condition])/len(variabilities["variabilities_" + condition])
    std = np.std(variabilities["variabilities_" + condition])
    
    #print results
    print(condition,'mean: ', avg,' std: ', std,' p of normal: ', p)
fig = plt.figure(figsize =(5, 3))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
ax.set_ylabel('Normalized blink rate variability')
ax.set_title('Boxplot of blink rate variability')
ax.set_xticklabels(conditions)
 
# Creating plot
bp = ax.boxplot(boxplot_data)
 
# show plot

plt.show()

#print result of kruskal wallis test
print(kruskal(boxplot_data[0],boxplot_data[1],boxplot_data[2],boxplot_data[3],boxplot_data[4]))

#%%
#create a table where we compare each condition of blink rate variability through a wilcoxon signed rank test

table = []
for i in range(len(boxplot_data)):
    table_row = []
    for j in range(len(boxplot_data)):
        if i == j:
            wilcox = 1.
        else:
            stat, wilcox = wilcoxon(boxplot_data[i],boxplot_data[j])
        table_row.append(wilcox)
    table.append(table_row)

fig, ax = plt.subplots(figsize=(15, 10))

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame(table)

ax.table(cellText=df.values, colLabels=df.columns, loc='center')

fig.tight_layout()

plt.show()
print(wilcoxon(boxplot_data[0],boxplot_data[4]))
#%%
#histogram used for visual inspection of normality
plt.hist(blink_rate_un["blinks_4_Combined"])

#%%
#unused plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
for condition in conditions:
    ax1.scatter(blink_rate["blinks_" + condition],variabilities["variabilities_" + condition], label = condition)
plt.legend()
plt.show()
#%%
#Poincare plot of RR intervals, not used
fig = plt.figure()
ax1 = fig.add_subplot(111)
for condition in conditions:
    test_data = data = pd.read_csv(data_path + '/P5'  +'/Eyerecording_Test_'+ condition + '_blinks.csv',sep=';')
    data.columns = ['none', 'frame', 'timestamp', 'openness_L', 'openness_R', 'pupil_diam_L', 'pupil_diam_R', 'blink']
    first = True
    indices = []
    timestamps = []
    RR_interval = []
    for i, count in data['blink'].iteritems():
        if pd.notnull(count):
            if first == True:
                print(count)
                first = False
                indices.append(i)
                timestamps.append(data['timestamp'][i])
            else:
                previous = indices[-1]
                indices.append(i)
                length = (int(data['timestamp'][i][-13:-11])- int(data['timestamp'][previous][-13:-11]))*3600 + (int(data['timestamp'][i][-10:-8])- int(data['timestamp'][previous][-10:-8]))*60 + (float(data['timestamp'][i][-7:])- float(data['timestamp'][previous][-7:])) #convert the different elements of the timestamps (string) to numbers and calculate the duration
                RR_interval.append(length)
                timestamps.append(data['timestamp'][i])
    print(RR_interval)
    RMSSD = np.sqrt(sum(square(RR_interval))/len(RR_interval))
    print(RMSSD)
    ax1.scatter(RR_interval[:len(RR_interval)-1],RR_interval[1:], label = condition)

plt.legend()
plt.show()
