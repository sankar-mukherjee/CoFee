# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:02:02 2015

@author: prevot
"""

import sys
#sys.path.append('/Users/prevot/Ubuntu One/Code')

import nltk
import csv
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier 

from numpy import array


from random import shuffle
from pandas import DataFrame
from pandas import Series


WORKING_DIR = 'C:/Users/mukherjee/Desktop/CoFee/analysis/laurent/'

CORPORA_LIST = ['CID','MTR']#,'MTX','DVD']
SPEAKER_LIST = ['AB','CM','AG','YM','EB','SR','NH','LL','AC','MB','BX','MG','LJ','AP','ML','IM','BLX','BLY','ROX','ROY','VEX','VEY','MAX','MAY']
ROLE_LIST = ['F','G','-']
SESSION_LIST = ['1','2','3','4','5','6','7','8']


def readData(filename):
    res = []
    with open(filename, 'r') as filename:
        reader = csv.DictReader(filename, delimiter=',', quotechar='"')
        for row in reader:
            res.append(row)
    return res
    
def filterData(data,filterField,filterValues,positive=True):
    '''
    Example filterData(data,'base0.65',['None'],False)
    Example filterData(data,'corpus',['MTX',MTR'],True)    
    '''
    res = []
    for line in data:
        if positive:
            if line[filterField] in filterValues:
                res.append(line)
        else:
            if line[filterField] not in filterValues:
                res.append(line)
                
    return res

def cutTextfeat(data,feature,threshold):
    
    listValues = [line[feature] for line in data]
    distrib = nltk.FreqDist(listValues)
    frequentValues = [item for item in distrib if distrib[item]>=threshold]
           
    print "size lexicon "+feature+" :"+str(len(frequentValues))
    for line in data:
        if line[feature] not in frequentValues:
            line[feature] = 'hapax'
    return data
    
def buildFilteredLexica(data,featureList,threshold):
    res = {}
    for feat in featureList:
        listValues= [line[feat] for line in data]
        distrib = nltk.FreqDist(listValues)
        frequentValues = [item for item in distrib if distrib[item] >= threshold]
        res[feat] = frequentValues
    return res
    
def cleanData(data,lexica):
    res = []
    
    for line in data:
        lineres = {}
        for item in line.keys():
            if item in ACO_FEAT+POS_FEAT+INTER_ACO_FEAT:
                if line[item] == 'NaN':
                    lineres[item] = 0 #TODO
                elif line[item] == 'Inf':
                    lineres[item] = -1 #TODO
                else:
                    lineres[item] = float(line[item])
            elif item in FUNCTIONS:
                    lineres[item] = line[item]
            elif item in ['sa','osa','osb','pa','pb','opa','opb']:
                if float(line[item]) > 4.0:
                    lineres[item] =4.0
                elif float(line[item]) < 0.0:
                    lineres[item] =0.0
                else: 
                    lineres[item] = float(line[item])
            elif item in NUM_LEX_FEAT + NUM_CTX_FEAT + NUM_INTER_FEAT + CTX_MAN_FEAT:
                lineres[item] = float(line[item])
            elif item in BIN_FEAT:
                lineres[item] = int(line[item])             
            elif item in TXT_FEAT:
                lb = preprocessing.LabelBinarizer()
                lb.fit(lexica[item])
                binSimple = lb.transform([(line[item])])
                for i in range(len(binSimple[0])):
                    lineres[item+'_'+str(i)] = binSimple[0][i]
            elif item in META_FEAT:
                lb = preprocessing.LabelBinarizer()
                lb.fit(lexica[item])
                binSimple = lb.transform([(line[item])])
                for i in range(len(binSimple[0])):
                    lineres[item+'_'+str(i)] = binSimple[0][i]
            lineres['ndo'] = float(line['do'])/float(line['dur'])
        
        res.append(lineres)
    return res
    
def prepdata(rawdata):
    for feature in TOKENS_FEAT:
        tempdata = cutTextfeat(rawdata,feature,30)
    for feature in BIGRAMS_FEAT:
        processedData = cutTextfeat(tempdata,feature,15)   

    lexiUni = buildFilteredLexica(processedData,TOKENS_FEAT,30)
    lexiBi = buildFilteredLexica(processedData,BIGRAMS_FEAT,15)
    lexicaCut = dict(lexiUni.items() + lexiBi.items())

    lexicaCut['corpus']= CORPORA_LIST
    lexicaCut['spk']= SPEAKER_LIST
    lexicaCut['sess']= SESSION_LIST
    lexicaCut['rol']= ROLE_LIST

    cleaneddata = cleanData(processedData,lexicaCut)
    
    return cleaneddata    


def runExpeClassifiers(data,features,target,fileout):

    allres = {}
    shuffle(data)

    # Prep features
    featData = array([[sample[feat] for feat in features] for sample in data])
#    featData_scaled = preprocessing.scale(featData)
    min_max_scaler = preprocessing.MinMaxScaler()
    featData_MinMax = min_max_scaler.fit_transform(featData)
    
    #Prep target
    labelBase = [sample[target] for sample in data]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(featData_MinMax, labelBase, test_size=0.1, random_state=0)
    
    classifiers = [
#        ('ada-10', AdaBoostClassifier(n_estimators=10)),
#        ('ada-50', AdaBoostClassifier(n_estimators=50)),
 #       ('ada-100', AdaBoostClassifier(n_estimators=100)),
#        ('svm-1', svm.SVC(cache_size=1000,C=1.0)),
#        ('svm-05', svm.SVC(cache_size=1000,C=0.5)),
 #       ('forest-10', RandomForestClassifier(n_estimators=10)),
 #       ('forest-20', RandomForestClassifier(n_estimators=20)),
        ('forest-50', RandomForestClassifier(n_estimators=50)),
        ('forest-50-min5', RandomForestClassifier(n_estimators=50,min_samples_leaf=5)),
        ('forest-50-min10', RandomForestClassifier(n_estimators=50,min_samples_leaf=10)),
 #       ('forest-5010', RandomForestClassifier(n_estimators=50,max_features=10)),
 #       ('forest-5020', RandomForestClassifier(n_estimators=50,max_features=20)),
    ]    
    
    for name,clf in classifiers:    
        print name
        clf = clf.fit(X_train, y_train)
        res = clf.predict(X_test)
        scores = cross_validation.cross_val_score(clf,featData_MinMax,labelBase,cv=10)
        print scores.mean()
        print scores.std()
        print(metrics.classification_report(y_test, res))
        allres[name] = scores 

        if name[0:6]=='forest':
            print"Feature Importance"
            showFeatureImportance(clf.feature_importances_,clf.estimators_,features,WORKING_DIR+'importance.png')
                
    cols = [x for (x,y) in classifiers]
    df = pd.DataFrame(allres,columns=cols)
    means = df.mean()
    errors = df.std()

    fig, ax = plt.subplots()
    ax = means.plot(yerr=errors, ax=ax, kind='bar') 
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.4,1)
    fig = ax.get_figure()
    fig.savefig(fileout)
    
    return allres

def showFeatureImportance(importances,estimators,features,fileout):
  
#    std = np.std([tr.feature_importances_ for tr in estimators],
#             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    forsumming = {}

    for f in range(len(features)):
        forsumming[features[indices[f]]] = importances[indices[f]]

    lex = aco = pos = aut = man = met = bid = 0 # oth =
    
    for feat in forsumming.keys():
        if feat in REAL_LEX_FEAT:
            lex = lex + forsumming[feat]
        elif feat in REAL_ACO_FEAT:
            aco = aco + forsumming[feat]
        elif feat in REAL_POS_FEAT:
            pos = pos + forsumming[feat]
        elif feat in REAL_CTX_AUT_FEAT:
            aut = aut + forsumming[feat]
        elif feat in REAL_CTX_MAN_FEAT:
            man = man + forsumming[feat]
        elif feat in REAL_MET_FEAT:
            met = met + forsumming[feat]
#        elif feat in REAL_OTH_FEAT:
#            oth = oth + forsumming[feat]
        else:
            bid = bid + forsumming[feat]
        
    df = pd.Series({'lex':lex,'aco':aco,'pos':pos,'ctx-aut':aut,'ctx-man':man,'met':met,'oth':bid})#inter':oth,

    # Plot the feature importances of the forest
    fig, ax = plt.subplots()
    ax = df.plot(ax=ax, kind='bar') 
    ax.set_ylabel('Accuracy')
    fig = ax.get_figure()
    fig.savefig(fileout)    
        
    return 0
    
def runOneExperiment(data,classifier,features,target):
 
    # Prep data   
    shuffle(data)
    featData = array([[sample[feat] for feat in features] for sample in data])
    min_max_scaler = preprocessing.MinMaxScaler()
    featData_MinMax = min_max_scaler.fit_transform(featData)
    
    #Prep target
    labelBase = [sample[target] for sample in data]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(featData_MinMax, labelBase, test_size=0.1, random_state=0)

    clf = classifier
    clf = clf.fit(X_train, y_train)
    res = clf.predict(X_test)
    scores = cross_validation.cross_val_score(clf,featData_MinMax,labelBase,cv=10)
    print scores.mean()
    print scores.std()
    print(metrics.classification_report(y_test, res))
      
    return scores
    
def runExpeFeatures(rawdata,classifier,target):
    '''
    Featsets is a list of pair name,featset
    '''
    
#    filtereddata = filterData(rawdata,'baseFun0.65','None',False)
#    filtereddata = rawdata    
#    filtereddata = filterData(rawdata,'EvalFun0.49','None',False)
    filtereddata = filterData(rawdata,'EvalFun0.65','None',False)

    corpora = [('cid',prepdata(filterData(filtereddata,'corpus','CID',True))),
               ('mtr',prepdata(filterData(filtereddata,'corpus','MTR',True))),
               ('all',prepdata(filtereddata))
               ]
    
    globalres = {}
    for namec,corpus in corpora:
        print '*******'
        print namec
        print '*******'
        
        allfeatures = corpus[0].keys()
        finalfeatures = set(allfeatures)
        REAL_LEX_FEAT = [item for item in finalfeatures if item[0:5]=='trans'] + NUM_LEX_FEAT + BIN_LEX_FEAT
        REAL_CTX_AUT_FEAT = [item for item in finalfeatures if item[0:7]=='othLast'] + [item for item in finalfeatures if item[0:8]=='prevLast'] + [item for item in finalfeatures if item[0:9]=='prevFirst'] + [item for item in finalfeatures if item[0:9]=='prevfirst'] + [item for item in finalfeatures if item[0:8]=='othFirst'] + [item for item in finalfeatures if item[0:8]=='othfirst']+ NUM_CTX_FEAT + NUM_INTER_FEAT
        REAL_MET_FEAT =  [item for item in finalfeatures if item[0:4]=='spk_'] + [item for item in finalfeatures if item[0:6]=='corpus'] + [item for item in finalfeatures if item[0:4]=='sess'] + [item for item in finalfeatures if item[0:3]=='rol'] 
        featsets = [#('bas',GEST_FEAT),
#                ('lpa+inter',REAL_LEX_FEAT+REAL_POS_FEAT+REAL_ACO_FEAT+REAL_CTX_AUT_FEAT+REAL_INTER_FEAT),
                ('lex',REAL_LEX_FEAT),
                ('pos',REAL_POS_FEAT),
                ('aco',REAL_ACO_FEAT),
                ('lex+pos',REAL_LEX_FEAT+REAL_POS_FEAT),
                ('lex+aco',REAL_LEX_FEAT+REAL_ACO_FEAT),
                ('lex+met',REAL_LEX_FEAT+REAL_MET_FEAT),
                ('aco+pos',REAL_ACO_FEAT+REAL_POS_FEAT),  
                ('lpa',REAL_LEX_FEAT+REAL_POS_FEAT+REAL_ACO_FEAT),
                ('lpa+aut',REAL_LEX_FEAT+REAL_POS_FEAT+REAL_ACO_FEAT+REAL_CTX_AUT_FEAT),
                ('lpa+aut+met',REAL_LEX_FEAT+REAL_POS_FEAT+REAL_ACO_FEAT+REAL_CTX_AUT_FEAT+REAL_MET_FEAT),
                ('lpa+man',REAL_LEX_FEAT+REAL_POS_FEAT+REAL_ACO_FEAT+REAL_CTX_MAN_FEAT),
                ('lpa+man+met',REAL_LEX_FEAT+REAL_POS_FEAT+REAL_ACO_FEAT+REAL_CTX_MAN_FEAT+REAL_MET_FEAT),
#                ('all',REAL_LEX_FEAT+REAL_POS_FEAT+REAL_ACO_FEAT+REAL_CTX_MAN_FEAT+REAL_MET_FEAT+REAL_INTER_FEAT)
                
                ]
        allres = {}
        for name,featset in featsets:
            print name
            allres[name] = runOneExperiment(corpus,classifier,featset,target)
        globalres[namec] = allres

    corpcol = ['cid','mtr', 'all']
    featcol = ['lex', 'pos', 'aco', 'lex+pos','lex+aco','lex+met','aco+pos','lpa', 'lpa+aut','lpa+aut+met','lpa+man','lpa+man+met']
        
    means = DataFrame(columns=corpcol, index=featcol)
    errors = DataFrame(columns=corpcol, index=featcol)
    for corpus in globalres.keys():
        print corpus
        for featset in globalres[corpus].keys():
            print featset
            means[corpus][featset] = Series(globalres[corpus][featset]).mean()
            errors[corpus][featset] = Series(globalres[corpus][featset]).std()

    fig, ax = plt.subplots()
    ax.set_ylim(0.5,0.9)
    means.plot(yerr=errors, ax=ax, kind='bar')
    fig.savefig(WORKING_DIR+'corpus-eval-featset.png',dpi=100)

    return globalres
    
    
##############################
##############################
##############################
##############################
DATA_FILE = WORKING_DIR+'allmerged.csv'
rawdata = readData(DATA_FILE)

# Remove None Values (if desired)
filtereddata = filterData(rawdata,'baseFun0.65','None',False)

TOKENS_FEAT = ['prevLastTok','prevFirstTok','othLastTok','othFirstTok','trans']
BIGRAMS_FEAT = ['prevLastBi','prevfirstBi','othLastBi','othfirstBi']#! prevfirstBi != prevFirstBi, othfirstBi

FUNCTIONS = ['baseFun0.49','baseFun0.65', 'baseFun0.74', 'baseFun1','EvalFun0.49','EvalFun0.65',	'EvalFun0.74','EvalFun1']


###################################
# LEX FEATURES
###################################
NUM_LEX_FEAT = ['nbmh','nbouais','size']
BIN_LEX_FEAT = ['ouais','mh', 'laugh','ah','bon','oui','mais','ok','dac','voila','non','et']
TXT_LEX_FEAT = ['trans']
LEX_FEAT = NUM_LEX_FEAT + BIN_LEX_FEAT + TXT_LEX_FEAT

###################################
# ACO FEATURES
###################################    
PITCH_FEAT = ['slope','f0max','f0min','f0stdev','NanRatio','span','steepness','height']
INTENSITY_FEAT = ['intQ1','intQ3','intQ2']#,'intQ1raw','intQ2raw','intQ3raw'
PHON_FEAT = ['phonSplit','form1','form2','form3']
ACO_FEAT = PITCH_FEAT + INTENSITY_FEAT + PHON_FEAT + ['duration'] + ['aperiodAV']

###################################
# INTER FEATURES
###################################
INTER_ACO_FEAT = ['steepnessInterl','spanInterl','f0maxInterl','intQ2Interl','aperiodAVInterl','f0stdevInterl','heightInterl','f0minInterl','NanRatioInterl',
                    'intQ1Interl', 'intQ2Interl','intQ3Interl','slopeInterl','durationInterl']
                              
###################################
# POS FEATURES
###################################    
POS_FEAT = ['sa','pa','pb','opa','opb','osa','osb','do','ndo','posDial']

###################################
# CTX-AUTO FEATURES
###################################    
NUM_CTX_FEAT = ['prevNbTu','prevNbJe','prevSize','prevNbFeedback','prevRed']
TXT_CTX_FEAT = ['prevLastTok','prevFirstTok','prevLastBi','prevfirstBi'] #! prevfirstBi != prevFirstBi
NUM_INTER_FEAT = ['othNbTu','othNbJe','othNbFeedback','othSize','othRed']
TXT_INTER_FEAT = ['othLastTok','othLastBi','othFirstTok','othfirstBi']#! prevfirstBi != prevFirstBi

###################################
# CTX-MAN FEATURES
###################################
CTX_MAN_FEAT = ['inint','quest','feedback','ordre','essai','assert','incomp']

###################################
# META FEATURES
###################################
META_FEAT = ['spk','sess','rol','corpus']

###################################
# OTHER FEATURES
###################################
SIM_ACOU_FEAT = ['SimScore']
GEST_FEAT = ['gest']    

###################################
# TYPE of FEATURES
###################################
TXT_FEAT = TXT_INTER_FEAT + TXT_CTX_FEAT + TXT_LEX_FEAT
BIN_FEAT = BIN_LEX_FEAT


data = prepdata(filtereddata)
allfeatures = data[0].keys()
finalfeatures = set(allfeatures)

# REAL FEATURES (Binarized,...)
REAL_LEX_FEAT = [item for item in finalfeatures if item[0:5]=='trans'] + NUM_LEX_FEAT + BIN_LEX_FEAT
REAL_ACO_FEAT =  ACO_FEAT  
REAL_POS_FEAT = POS_FEAT          
REAL_CTX_MAN_FEAT = CTX_MAN_FEAT
REAL_CTX_AUT_FEAT = [item for item in finalfeatures if item[0:7]=='othLast'] + [item for item in finalfeatures if item[0:8]=='prevLast'] + [item for item in finalfeatures if item[0:9]=='prevFirst'] + [item for item in finalfeatures if item[0:9]=='prevfirst'] + [item for item in finalfeatures if item[0:8]=='othFirst'] + [item for item in finalfeatures if item[0:8]=='othfirst']+ NUM_CTX_FEAT + NUM_INTER_FEAT
REAL_MET_FEAT = [item for item in finalfeatures if item[0:4]=='spk_'] + [item for item in finalfeatures if item[0:6]=='corpus'] + [item for item in finalfeatures if item[0:4]=='sess'] + [item for item in finalfeatures if item[0:3]=='rol'] 
REAL_INTER_FEAT = INTER_ACO_FEAT
 

#RUN classifier testing experiment
runExpeClassifiers(data,REAL_LEX_FEAT+REAL_POS_FEAT+REAL_ACO_FEAT+REAL_CTX_AUT_FEAT+REAL_CTX_AUT_FEAT+REAL_MET_FEAT,'baseFun0.65',WORKING_DIR+'classifier.png')



runOneExperiment(data,classifier,REAL_POS_FEAT,'baseFun0.65')





