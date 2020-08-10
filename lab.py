#!/usr/bin/env python
# coding: utf-8

# © Copyright 2020, D-Tech, LLC, All Rights Reserved. 
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

from hyperopt import STATUS_OK, tpe, Trials, fmin, hp
from hyperopt.hp import randint, uniform, choice

from time import time
from datetime import timedelta, datetime

import os
import json
import shutil
import warnings
import sys

from joblib import dump#, load

# do not display python warnings, which are produced by hyperopt
warnings.filterwarnings("ignore")

# [pipe] is a dictionary representing a pipeline containing preprocessing stages
pipe = {}

# wrapper function to run a [config] and return a hyperloss (called for optimization by hyperopt)
def run_wrapper(config):
    
    # load filename aliases
    sources = json.loads(str(open("filenames.json").read()))
    
    # safe division
    def mydiv(a, b):
        if b == 0: return None
        return a/b
    
    # exponentiation that can handle None^x
    def mypow(a,b):
        if a == 0 and b == 0: return 1
        if a is None: return None
        return a ** b
    
    # multiplication that can handle None
    def mytimes(a,b,c,d):
        if None in [a,b,c,d]: return None
        return a*b*c*d
    
    # returns [a] to [n] decimal places
    def clip(a, n = 3):
        if a is None: return None
        return int(a*(10**n))/(10**n)

    # computes and prints metrics based on [confusion_matrix]
    # if [chosen_metric] is specified, that metric is returned
    # [box_weighting] specifies the calculation of the box measure
    def analyze(confusion_matrix, chosen_metric = None, box_weighting = (0.25,0.25,0.25,0.25)):
        
        TN, FP, TP, FN = confusion_matrix

        precision = mydiv(TP, TP + FP)
        coprecision = mydiv(TN, TN + FN)
        recall = mydiv(TP, TP + FN)
        corecall = mydiv(TN, TN + FP)
        
        accuracy = mydiv(TP + TN, TP + TN + FP + FN)
        f = mydiv(TP, TP + (FP + FN)/2)
        
        # normalize box weights
        weight_total = sum(list(box_weighting))
        box_weighting = [w/weight_total for w in box_weighting]
        
        # the box is the weighted geometric mean of the four basic metrics above
        box = mytimes(mypow(precision, box_weighting[0]),\
                      mypow(coprecision, box_weighting[1]),\
                      mypow(recall, box_weighting[2]),\
                      mypow(corecall, box_weighting[3]))

        print(".")
        print(f"  TN: {TN}\tFN: {FN}")
        print(f"  FP: {FP}\tTP: {TP}")
        print(".")
        print(f"  Precision: {clip(precision)}, Recall: {clip(recall)}")
        print(f"  Coprecision: {clip(coprecision)}, Corecall: {clip(corecall)}")
        print(f"  Accuracy: {clip(accuracy)}, F-Measure: {clip(f)}")
        print(f"  Box: {clip(box)}")
        print(".")

        stats = {"TN":TN, "FN":FN, "FP":FP, "TP":TP, \
                 "precision":precision, "recall":recall, \
                 "coprecision":coprecision, "corecall":corecall, \
                 "accuracy":accuracy, "f":f, \
                 "box":box}
        
        # not necessary currently, to be used for a terse log
        short = f"Box: {box},TN: {TN},FP: {FP},TP: {TP},FN: {FN}"
        
        # dictionary result for hyperopt
        dictionary = {"status": STATUS_OK}
        if chosen_metric is not None: dictionary["loss"] = -eval(chosen_metric) # negated because hyperopt is minimizing
        return (dictionary, stats, short)

    # returns a sublist of [l] according to [condition]
    def interpret_list(condition, l):

        #strings beginning with feature_exeption will include all but the columns listed after
        feature_exception = "EXCEPT: "

        if condition == "all": return l
        if condition == "none": return []
        if feature_exception == condition[:len(feature_exception)]:
            return [x for x in l if not x in eval(condition[len(feature_exception):])]
        if type(eval(condition)) == list: return [x for x in l if x in eval(condition)]
        return list(filter(eval(condition), l))

    # runs experiment specified by config, printing intermediate steps and returning results
    def run(config):
        
        global pipe
        
        print("-"*50)
        
        start_time = time()

        configuration = config["run"]

        dataset_list = configuration["datasets"]
        data_management = configuration["data_management"]
        
        # create a dataset by concatenating the datasets from all specified paths
        df = pd.concat(list(map(pd.read_csv, list(map(lambda x: sources[x], dataset_list)))))
        
        # strip column names
        df = df.rename(columns = (lambda x: x.strip()))
        
        initial_size = len(df)

        # remove rows with NaN or +/- infinity
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].reset_index(drop=True)

        if(len(df) != initial_size):
            print("<Some rows contained either NaN or inf values are were removed>")

        processing_configuration = configuration["processing"]
        label_column, normal_label = processing_configuration["label_column"], processing_configuration["normal_label"]

        # selected features of the dataset 
        selected_features = interpret_list(processing_configuration["features"], list(df.columns))
        df = df[selected_features]
        
        categorical_configuration = processing_configuration["categorical"]
        
        # categorical features of the dataset 
        cat_columns = interpret_list(categorical_configuration["features"], list(df.columns))
        nonlabel_columns = [column for column in list(df.columns) if column != label_column]
        noncat_columns = [column for column in list(df.columns) if (not column in cat_columns) and column != label_column]

        # split dataset into training and testing dataset
        trainDf, testDf = train_test_split(df, test_size = data_management["test"])
        
        # only use normal data for training (should be removed for more general machine learning tasks)
        trainDf = trainDf[trainDf[label_column] == normal_label]

        # if specified, move non-normal data from training set into testing set (also should be removed later)
        if data_management["use_all"]: testDf = pd.concat([testDf, df[df[label_column] != normal_label]])

        trainDf = trainDf.reset_index(drop=True)
        testDf = testDf.reset_index(drop=True)

        # index categorical features if specified
        if categorical_configuration["index"]:
            print("Indexing")
            enc = preprocessing.OrdinalEncoder()
            enc.fit(trainDf[cat_columns])
            
            pipe["index"] = enc
            
            trainDf[cat_columns] = enc.transform(trainDf[cat_columns])
            testDf[cat_columns] = enc.transform(testDf[cat_columns])

        reduce_configuration = categorical_configuration["reduce"]
        
        # modulo reduction
        if reduce_configuration["method"] == "mod":
            for c in cat_columns:
                trainDf[c] = trainDf[c] % reduce_configuration["num"]
                testDf[c] = testDf[c] % reduce_configuration["num"]

        # clustering reduction
        if reduce_configuration["method"] == "cluster":

            for c in cat_columns:
                    
                # dataset with a point for each distinct value of the cateogorical variable
                valueDf = trainDf[nonlabel_columns][noncat_columns + [c]]
                # assign each categorical value the averages of the numerical values that appear with it
                valueDf = valueDf.groupby(c).median()
                # populate the cateogrical feature column with the categorical values
                valueDf[c] = valueDf.index
                kmeans = KMeans(n_clusters = reduce_configuration["num"])
                kmeans.fit(valueDf[[x for x in nonlabel_columns if x in noncat_columns]])
                
                pipe["clustering"] = kmeans
                
                # assign each categorical value to its cluster number
                valueDf["center"] = kmeans.predict(valueDf[[x for x in nonlabel_columns if x in noncat_columns]])
                
                # save the map from categorical values to cluster numbers
                center_assignment = {}
                for index, row in valueDf.iterrows():
                    center_assignment[row[c]] = row["center"]

                trainDf[c] = trainDf[c].map(center_assignment)
                
                pipe["centerAssignment"] = center_assignment
                
                # in the test set, map the categorical values to cluster centers, if that map is specified. 
                # otherwise, map to a cluster center based on the numerical features
                for index, row in testDf.iterrows():
                    if row[c] in list(center_assignment): 
                        row[c] = center_assignment[row[c]]
                    else: row[c] = kmeans.predict(pd.DataFrame([row[[x for x in nonlabel_columns if x in noncat_columns]]]))[0]
                    testDf.at[index, c] = row[c]
                    
        if reduce_configuration["method"] == "hash": pass # TODO

        if categorical_configuration["onehot"]:
            print("One hot encoding")
            hot = preprocessing.OneHotEncoder(handle_unknown='ignore')
            hot.fit(trainDf[cat_columns])
            
            pipe["onehot"] = hot
            
            # changes onehot encoded feature from matric to dataframe, and adds it to the rest of the dataframe
            def onehot_encode(df):
                return pd.DataFrame.sparse.from_spmatrix(hot.transform(df[cat_columns])) \
                    .join(df[[c for c in list(df.columns) if c not in cat_columns]])

            trainDf = onehot_encode(trainDf)
            testDf = onehot_encode(testDf)
            
            # update the nonlabel_columns with these newly created dummy variables
            nonlabel_columns = [c for c in list(trainDf.columns) if c != label_column]

        if processing_configuration["scaled"] != "none":
            print("Scaling")
            scaler = None

            if processing_configuration["scaled"] == "SD":
                scaler = preprocessing.StandardScaler()
            elif processing_configuration["scaled"] == "maxabs":
                scaler = preprocessing.MaxAbsScaler()
            elif processing_configuration["scaled"] == "minmax":
                scaler = preprocessing.MinMaxScaler()

            scaler.fit(trainDf[nonlabel_columns])
            
            pipe["scaler"] = scaler
            
            trainDf[nonlabel_columns] = scaler.transform(trainDf[nonlabel_columns])
            testDf[nonlabel_columns] = scaler.transform(testDf[nonlabel_columns])
    
        # --------------------------------------------------------------------------------------------------------------------

        model_configuration = configuration["model"]

        # dot product between iterables (may be replaced with numpy later)
        def dot(v1, v2):
            total = 0
            for i in range(len(v1)):
                total += v1[i]*v2[i]
            return math.exp(-total)

        # Euclidean distance between iterables (may be replaced with numpy later)
        def dist(v1, v2):
            total = 0
            for i in range(len(v1)): 
                total += (v1[i]-v2[i])**2
            return total ** 0.5

        # general metric function between iterables
        def metric(mode, v1, v2):
            if mode == "dist": return dist(list(v1), list(v2))
            elif mode == "dot": return dot(list(v1), list(v2))
            return dist(list(v1), list(v2)) 
            
        # accumulated anomaly scores across all feature bags
        testDf["distances"] = 0
        
        bag_num = configuration["bag_num"]
        bag_size = configuration["bag_size"]
        
        model = None
            
        for i in range(bag_num):
            
            if bag_size == "all": bag_size = len(nonlabel_columns)
                
            # choose random subspace according to bag size
            features = random.sample(nonlabel_columns, min(bag_size, len(nonlabel_columns)))
            
            # project dataset to that random subspace
            limited_trainDf = trainDf[features]
            limited_testDf = testDf[features]

            if model_configuration["type"] == "autoencoder":

                print("Training autoencoder "+str(i))

                # size of middle and outer layers
                middle_size, IO_dimension = model_configuration["encoding_size"], len(list(limited_trainDf.columns))

                # the size of a layer 2 and 4 (if 0, no such layer is included)
                extra_size = model_configuration["extra"]
                
                encoder = None
                decoder = None

                if extra_size > 0:
                    outer_encoder = Sequential([Dense(extra_size, input_shape = [IO_dimension])])
                    middle_encoder = Sequential([Dense(middle_size, input_shape = [extra_size])])
                    middle_decoder = Sequential([Dense(extra_size, input_shape=[middle_size])])
                    outer_decoder = Sequential([Dense(IO_dimension, input_shape=[extra_size])])

                    encoder = Sequential([outer_encoder, middle_encoder])
                    decoder = Sequential([middle_decoder, outer_decoder])

                else:
                    encoder = Sequential([Dense(middle_size, input_shape=[IO_dimension])])
                    decoder = Sequential([Dense(IO_dimension, input_shape=[middle_size])])

                autoencoder = Sequential([encoder,decoder])

                lr = model_configuration["learning_rate"]
                loss = {"dist": "mse"}["dist"]
                optimizer = {"adam": Adam(learning_rate=lr), 
                             "sgd": SGD(learning_rate=lr)
                            }[model_configuration["optimizer"]]

                autoencoder.compile(loss = loss, optimizer = optimizer)

                autoencoder.fit(limited_trainDf, \
                                limited_trainDf, \
                                epochs = model_configuration["epochs"], verbose = 0)
                
                # Save
                print("Persisting")
                n = 0 # number of autoencoders that have been persisted this run
                if not os.path.exists("results/temp_models"): os.mkdir("results/temp_models")
                while(os.path.exists("results/temp_models/autoencoder"+str(n))): n += 1
                if not os.path.exists("results/temp_models/autoencoder"+str(n)): os.mkdir("results/temp_models/autoencoder"+str(n))
                autoencoder.save("results/temp_models/autoencoder"+str(n)+"/autoencoder_model", save_format='h5') # persist model
                dump(pipe,"results/temp_models/autoencoder"+str(n)+ "/pipeline") # persist pipeline
                
                print("Testing")
                # add predictions column to the limited_testDf
                predictions = pd.DataFrame(autoencoder.predict(limited_testDf))\
                    .rename(mapper = lambda s: str(s)+"_prediction", axis = "columns")

                prediction_columns = list(predictions.columns)
                nonprediction_columns = [x for x in list(limited_testDf.columns) if x not in prediction_columns]
                limited_testDf = limited_testDf.join(predictions)
                
                # set the anomalousness scored produced by this feature bag
                testDf["temp_distances"] = limited_testDf.apply(func = lambda r: metric(model_configuration["metric"], \
                    r[prediction_columns], r[nonprediction_columns]), axis = 1)
                
                model = autoencoder

            elif model_configuration["type"] == "ocsvm":

                print("Training OCSVM "+str(i))

                kernel = model_configuration["kernel"]
                nu = model_configuration["nu"]
                gamma = model_configuration["gamma"]
                
                ocsvm = svm.OneClassSVM(nu = nu, kernel = kernel, gamma = gamma)
                ocsvm.fit(limited_trainDf)
                
                # Save
                print("Persisting")
                n = 0 # number of ocsvms that have been persisted this run
                if not os.path.exists("results/temp_models"): os.mkdir("results/temp_models")
                while(os.path.exists("results/temp_models/ocsvm"+str(n)+"/ocsvm_model")): n += 1
                if not os.path.exists("results/temp_models/ocsvm"+str(n)): os.mkdir("results/temp_models/ocsvm"+str(n))
                dump(ocsvm, "results/temp_models/ocsvm"+str(n)+"/ocsvm_model") # persist model
                dump(pipe,"results/temp_models/ocsvm"+str(n)+ "/pipeline") # persist pipeline
                
                print("Testing")
                # set the anomalousness scored produced by this feature bag
                testDf["temp_distances"] = ocsvm.predict(limited_testDf)
                
                model = ocsvm
            
            elif model_configuration["type"] == "kmeans":

                print("Training kmeans "+str(i))

                kmeans = KMeans(n_clusters = model_configuration["k"])
                kmeans.fit(limited_trainDf)
                
                # Save
                print("Persisting")
                n = 0
                # number of kmeans models that have been persisted this run
                if not os.path.exists("results/temp_models"): os.mkdir("results/temp_models")
                while(os.path.exists("results/temp_models/kmeans"+str(n)+"/kmeans_model")): n += 1
                if not os.path.exists("results/temp_models/kmeans"+str(n)): os.mkdir("results/temp_models/kmeans"+str(n))
                dump(kmeans, "results/temp_models/kmeans"+str(n)+"/kmeans_model") # persist model
                dump(pipe,"results/temp_models/kmeans"+str(n)+ "/pipeline") # persist pipeline
                
                centers = kmeans.cluster_centers_
                noncenter_columns = [x for x in list(limited_testDf.columns) if x != "centers"]
                
                print("Testing")
                limited_testDf["centers"] = kmeans.predict(limited_testDf)
                # set the anomalousness scored produced by this feature bag
                testDf["temp_distances"] = limited_testDf.apply(func = lambda r: metric(model_configuration["metric"], \
                    centers[int(r["centers"])],r[noncenter_columns]), axis = 1)
                
                model = kmeans
                    
            # accumulated the anomalousness scores from this bag into the total
            testDf["distances"] = testDf["distances"] + testDf["temp_distances"]
            # remove temporary distance column
            testDf = testDf[nonlabel_columns+[label_column, "distances"]]
            
        # can be used to save anomaly score distribution for further study
        # testDf[["distances",label_column]].to_csv("dist_distro_1.csv", index = False)
                                
        if model_configuration["type"] in ["autoencoder", "kmeans"]:
            # prediction boolean/class is a cutoff of the anomaly score
            testDf["predictions"] = testDf["distances"] > (bag_num * model_configuration["threshold"])
            
        if model_configuration["type"] == "ocsvm":
            
            testDf["predictions"] = testDf["distances"] < 0
        
        # --------------------------------------------------------------------------------------------------------------------
        
        # generate confusion matrix
        TP = int(testDf.loc[(testDf[label_column] != normal_label) & (testDf["predictions"] == True)].shape[0])
        FP = int(testDf.loc[(testDf[label_column] == normal_label) & (testDf["predictions"] == True)].shape[0])
        TN = int(testDf.loc[(testDf[label_column] == normal_label) & (testDf["predictions"] == False)].shape[0])
        FN = int(testDf.loc[(testDf[label_column] != normal_label) & (testDf["predictions"] == False)].shape[0])
        
        print("Analyzing")
        # analyze confusion matrix
        result = analyze((TN, FP, TP, FN), configuration["hyperloss"])
        result[0]["model"] = model # add the model to the hyperopt result dictionary
        
        print(f"Time: {timedelta(seconds = time() - start_time)}")
        
        return result
        
    # in [run_wrapper], run the config
    re = run(config)
    
    print("Logging")
    
    #hours=4 for time zone correction [can be changed to adjust to your time zone]
    date = str(datetime.now() - timedelta(hours=4))[:-7]
    
    # add a log entry to temp.txt
    log_entry = {"date":date, "config":config["run"], "results":re[1], \
           "model":(config["run"]["model"]["type"] + "_" + date.replace(" ", "_"))}

    # clean the log by casting all numpy int64s to python ints (for the purpose of saving to json)
    def cleanlog(ma):
        if(type(ma) == dict):
            newmap = {}
            for i in list(ma):
                newmap[i] = cleanlog(ma[i])
            return newmap
        if(type(ma) == np.int64):
            return int(ma)
        return ma
        
    # write to temp_log
    temp_log = open("results/temp.txt", "a+")
    temp_log.write(json.dumps(cleanlog(log_entry), indent=4))
    temp_log.write("\n\n")
    temp_log.close()
    
    return re[0]

config_filename = sys.argv[1] # take filename as an argument
hyperopt_evals = 1 # number of runs that should take place
if len(sys.argv) > 2: hyperopt_evals = int(sys.argv[2]) # take the number of runs as an optional argument

# remove temporary locations
try: shutil.rmtree("results/temp_models/") 
except: pass
try: os.remove("results/temp.txt")
except: pass
# ensure that the results directory is present
try: os.mkdir("results/")
except: pass

print("-"*50)
   
if hyperopt_evals > 0:
    
    # read in the config
    config_file = open(config_filename)
    config_string = config_file.read()
    config_file.close()
    print(config_string)
    space = eval(config_string) # the config, as a nested python dictionary
    trials = Trials()
    
    open("results/temp.txt", "w").close() # clear the temporary file

    # run hyperopt
    best_run = fmin(fn = run_wrapper,
            space = space,
            algo = tpe.suggest,
            max_evals = int(hyperopt_evals), 
            verbose = False, 
            trials = trials)
    
    # read in the log entries from this run from the temporary log
    temp_file = open('results/temp.txt', newline = '')
    data = list(str(temp_file.read()).split('\n\n')) 
    temp_file.close()
        
    all_log = open('results/all_log.txt', "a", newline = '')
        
    # determine the winning run, and log all runs in the all_log
    best_hyperloss, best_id = 0, 0
    for i in range(len(data)-1):
     
        all_log.write(data[i])
        all_log.write("\n\n")
        
        datum = json.loads(data[i])
        
        hyperloss = datum["results"][datum["config"]["hyperloss"]]
        if hyperloss > best_hyperloss:
            best_hyperloss = hyperloss
            best_id = i
            
    all_log.close()
                
    # log the winning run in the winner_log, and persist the winning model
    best_datum = json.loads(data[best_id])
    
    #moves best to results/models
    shutil.copytree("results/temp_models/" + best_datum["config"]["model"]["type"] + str(best_id), \
                    "results/models/" + best_datum["model"])

    winner_log = open('results/log.txt', "a",  newline='')
    winner_log.write(data[best_id])
    winner_log.write("\n\n")
    winner_log.close()

    #removes the temporary files
    shutil.rmtree("results/temp_models/")
    os.remove("results/temp.txt")
    
    print("-"*50)
