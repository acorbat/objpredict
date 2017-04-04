# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:56:20 2017

@author: Agus
"""
import pickle
from collections import Counter
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.externals import joblib

import process_experiment as exp 

def train_classifier(train_data):
    train_data = equiparate_classes(train_data)
    
    attr_cols = exp.list_attrs()
    X = train_data[attr_cols]
    y = train_data['c']
    
    trf_type = 'PCA'
    # select, train and save transformation method
    trf = choose_transformation(trf_type)
    trf = trf.fit(X)
    X_trf = pd.DataFrame(trf.transform(X), index=train_data.index)
    
    clf_type = 'ExtraTrees'
    clf = choose_classifier(clf_type)
    clf = clf.fit(X_trf, y)
    return {'trf':trf, 'clf':clf}
    

def classify(df, pipeline='PCA_ExtraTrees'):
    cols = exp.list_attrs()
    
    X = df[cols]
    
    pipedict = load_pipeline(pipeline)
    trf = pipedict['trf']
    clf = pipedict['clf']
    
    X = pd.DataFrame(trf.transform(X), index=X.index)
    
    classes = clf.classes_
    predictions = pd.DataFrame(clf.predict_proba(X), columns=classes, index=X.index)
    predictions['predicted'] = clf.predict(X)
    
    df = pd.concat([df, predictions], axis=1)
    
    return df


def save_pipeline(path, pipedict):
    with open(str(path), 'wb') as fo:
        pickle.dump(pipedict, fo)    


def load_pipeline(path):
    with open(str(path), 'rb') as fi:
        return pickle.load(fi)


def choose_transformation(trf_type):
    return PCA()


def choose_classifier(clf_type):
    return ExtraTreesClassifier(1000, bootstrap=True)


def equiparate_classes(df):
    count = Counter(df.c.values)
    mx = np.min(list(count.values()))
    
    selected = []
    for label in count.keys():
        ndxs = df[df['c'] == label].index.tolist()
        np.random.shuffle(ndxs)
        selected.extend(ndxs[:mx])
    
    df = df.loc[selected]
    return df