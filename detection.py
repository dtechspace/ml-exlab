# © Copyright 2019, D-Tech, LLC, All Rights Reserved. 
# Version: 0.5 (initial version), 08/10/2020
# License: The use of this software program is subject to the ML-ExLab 
# license terms and conditions as defined in the LICENSE file.
# Disclaimer: This software is provided "AS IS" without warrantees.  
# D-Tech, LLC has no obligation to provide any maintenence, update 
# or support for this software.  Under no circumstances shall D-Tech,  
# LLC be liable to any parties for direct, indirect, special, incidental,
# or consequential damages, arising out of the use of this software
# and related data and documentation.
#

import pandas as pd
import numpy as np
import math
import random

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.cluster import KMeans

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras import models

from hyperopt import STATUS_OK, tpe, Trials, fmin, hp
from hyperopt.hp import randint, uniform, choice

from time import time
from datetime import timedelta, datetime

import os
import json
import shutil
import warnings

from joblib import dump, load
import sys

from sklearn.pipeline import Pipeline

# returns a sublist of [l] according to [condition]
def interpret_list(condition, l):

    feature_exception = "EXCEPT: "

    if condition == "all": return l
    if condition == "none": return []
    if feature_exception == condition[:len(feature_exception)]: 
        return [x for x in l if not x in eval(condition[len(feature_exception):])]
    if type(eval(condition)) == list: return [x for x in l if x in eval(condition)]
    return list(filter(eval(condition), l))

# Load in Data----------------------------------------------------------------------------------------------

# load filename aliases
sources = json.loads(str(open("filenames.json").read()))

# load config
jsonmap = open(sys.argv[1])
conf = json.load(jsonmap)
jsonmap.close()

# load pipeline
pipe = load("results/models/"+ conf["model"]+ "/pipeline")

# Features Select--------------------------------------------------------------------------------------------  

print("Selecting Features")

# create a dataset by concatenating the datasets from all specified paths
dataset_list = conf["config"]["datasets"]
df = pd.concat(list(map(pd.read_csv, list(map(lambda x: sources[x], dataset_list)))))

# FOR TESTING--
# train, test = train_test_split(df, test_size = .2)
# df = test
# -------------

# strip column names
df = df.rename(columns = (lambda x: x.strip()))

initial_size = len(df)

# remove rows with NaN or +/- infinity
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].reset_index(drop=True)

if(len(df) != initial_size):
    print("<Some rows contained either NaN or inf values are were removed>")
    
# Cat & Label columns------------------------------------------------------------------------------------------

configuration = conf["config"]
processing_configuration = configuration["processing"]
label_column, normal_label = processing_configuration["label_column"], processing_configuration["normal_label"]

# selected features of the dataset 
selected_features = interpret_list(processing_configuration["features"], list(df.columns))
df = df[selected_features]

# categorical features of the dataset 
categorical_configuration = processing_configuration["categorical"]
cat_columns = interpret_list(categorical_configuration["features"], list(df.columns))
nonlabel_columns = [column for column in list(df.columns) if column != label_column]
noncat_columns = [column for column in list(df.columns) if (not column in cat_columns) and column != label_column]

#Index[Ordinal]-----------------------------------------------------------------------------------------------

# index categorical features if specified
print("Indexing")
if categorical_configuration["index"]:
    print("Indexing")

    enc = pipe["index"]
     
    df[cat_columns] = enc.transform(df[cat_columns])
    
#Reduce ------------------------------------------------------------------------------------------------------

print("Categorical Reduciton")
reduce_configuration = categorical_configuration["reduce"]

if reduce_configuration["method"] == "mod":#MOD
    for c in cat_columns:
        df[c] = df[c] % reduce_configuration["num"]

if reduce_configuration["method"] == "cluster":#CLUSTER   
    kmeans = pipe["clustering"]
    center_assignment = pipe["centerAssignment"]
    for c in cat_columns:
        for index, row in df.iterrows():
            if row[c] in list(center_assignment): 
                row[c] = center_assignment[row[c]]
            else: row[c] = kmeans.predict(pd.DataFrame([row[[x for x in nonlabel_columns if x in noncat_columns]]]))[0]
            df.at[index, c] = row[c]

if reduce_configuration["method"] == "hash": pass #HASH --- Remove this

#-if reduce_configuration["method"] == "none": don't do anything

#OneHot -------------------------------------------------------------------------------------------------------

print("OneHot Encoding")
if categorical_configuration["onehot"]:

    hot = pipe["onehot"]

    def onehot_encode(df):
        return pd.DataFrame.sparse.from_spmatrix(hot.transform(df[cat_columns])) \
            .join(df[[c for c in list(df.columns) if c not in cat_columns]])

    df = onehot_encode(df)
    
    #updates nonlabel_columns
    nonlabel_columns = [c for c in list(df.columns) if c != label_column]
    
#Scaling ------------------------------------------------------------------------------------------------------

print("Scaling")
if processing_configuration["scaled"] != "none":
    scaler = pipe["scaler"]
    df[nonlabel_columns] = scaler.transform(df[nonlabel_columns])
    
#Model --------------------------------------------------------------------------------------------------------

def dot(v1, v2):
    total = 0
    for i in range(len(v1)):
        total += v1[i]*v2[i]
    return math.exp(-total)

def dist(v1, v2):
    total = 0
    for i in range(len(v1)): 
        total += (v1[i]-v2[i])**2
    return total ** 0.5

def metric(mode, v1, v2):
    if mode == "dist": return dist(list(v1), list(v2))
    elif mode == "dot": return dot(list(v1), list(v2))
    return dist(list(v1), list(v2)) 
            
df["distances"] = 0

bag_num = conf["config"]["bag_num"]
bag_size = conf["config"]["bag_size"]

model_configuration = conf["config"]["model"]

model = None

#Will iterate for the number of bags specified
for i in range(bag_num):

    if bag_size == "all": bag_size = len(nonlabel_columns)

    features = random.sample(nonlabel_columns, min(bag_size, len(nonlabel_columns)))
    
    limited_df = df[features]

    #If the model was an autoencoder
    if model_configuration["type"] == "autoencoder":
        print("AutoEncoder")
        
        #loads model that was generated in lab.py
        autoencoder = models.load_model("results/models/"+ conf["model"]+ "/autoencoder_model")
        
        predictions = pd.DataFrame(autoencoder.predict(limited_df))\
            .rename(mapper = lambda s: str(s)+"_prediction", axis = "columns")
        
        prediction_columns = list(predictions.columns)
        nonprediction_columns = [x for x in list(limited_df.columns) if x not in prediction_columns]
        limited_df = limited_df.join(predictions)

        df["temp_distances"] = limited_df.apply(func = lambda r: metric(model_configuration["metric"], \
            r[prediction_columns], r[nonprediction_columns]), axis = 1)
        
        
    #If the model was an One Class Support Vector Machine
    elif model_configuration["type"] == "ocsvm":
        print("OCSVM")

        #loads model that was generated in lab.py
        ocsvm = load("results/models/"+ conf["model"]+ "/ocsvm_model")
        
        df["temp_distances"] = ocsvm.predict(limited_df)

    #If the model was K-Means
    elif model_configuration["type"] == "kmeans":
        print("K-Means")
        
        #loads model that was generated in lab.py
        kmeans = load("results/models/"+ conf["model"]+ "/kmeans_model")

        #gets centers from model
        centers = kmeans.cluster_centers_
        noncenter_columns = [x for x in list(limited_df.columns) if x != "centers"]
        
        limited_df["centers"] = kmeans.predict(limited_df)
        
        #computes distances to nearest centroid
        df["temp_distances"] = limited_df.apply(func = lambda r: metric(model_configuration["metric"], \
            centers[int(r["centers"])],r[noncenter_columns]), axis = 1)

    #The [distances] metric is the anomaly score
    df["distances"] = df["distances"] + df["temp_distances"]
    df = df[nonlabel_columns+[label_column, "distances"]]
    
    #df (which now contains a column "distances") can be exported as a CSV file with the line: df.to_csv("/file/path/to/export/to")
    print(df["distances"])
