# ML-ExLab
![](MLExLab.png)
## Background

In general, much of the machine learning development process has been of the form: write a piece of code to test a certain machine learning idea, run the code to test the idea, debugging along the way, change or fork the code to try another idea and improve the results, and repeat. This cycle resulted in many pieces of code, and many versions of each program, that varied slightly while being mostly redundant. This led to disorganization, confusion, inefficiency, inaccessibility of old code, and incomparability of recorded results due to slightly varying contexts. This led to the idea to formally specify all of the choices and variables that specify a machine learning experiment, from which results can be more easily organized and compared. This immediately produced the further idea to create a program that can take in such a formally specified "experiment configuration" and run that experiment, eliminating code proliferation, redundancy, and confusion.

## Benefits

The ML-ExLab (stand for ML experiment lab) provides ML designers and data scientists with a convenient and streamlined way to fine-tune ML training parameters, record the experiemental data, and comparing the performance of different models.  It is a Python-based tool for running ML experiments powered by SKLearn (https://scikit-learn.org) and TensorFlow (https://www.tensorflow.org) specified by a configuration data structure.  Those libraries already abstract away most of the technical details, leaving the user to specify only the relevant choices. They integrate with Python, which allows total flexibility and compatibility with other desired behaviors. The experiment lab provides a higher level of abstraction, reducing the customizability of the experiment at the benefit of greatly consolidated user specification. In this sense, Markdown is to HTML as the Ex-Lab is to Keras or SKLearn. A data scientist in a Kaggle competition (https://www.kaggle.com/competitions) can use Keras (https://keras.io/) to converge on an optimal, complex machine learning solution for a particular dataset, but for the rapid development of a simple machine learning solution that must apply across many datasets, it is not preferred to be constantly editing programs, swapping out models, datasets, and preprocessing features. For this rapid variability, the experiment lab is preferred. 

The ML-ExLab is especially suited to standardized logging of experiments.  Instead of identifying and recording all degrees of freedom of an experiment, each time a developer records any results, the experiment lab can automatically log results using the degrees of freedom specified by the configuration. This guarantees that data points can be accurately compared. For example, without the experiment lab, one might have two datapoints with different models running on the same dataset, but unbeknownst to the reviewer of the data, the code was edited to process the data differently between the two runs. This unseen change could account for much of the performance difference between the two runs, which difference would be falsely attributed to the difference in model. If all of the degrees of freedom are specified and logged with the experiment lab, any two data points that appear to be comparable will actually be comparable.

This system also allows easier integration with hyperopt, since the space of hyperparameters can be more narrowly integrated into the program, in the configuration, rather than distributed all throughout a program file. It also allows easy hyperoptimization of higher level variables, like the type of model and the nature of the data processing. In this same vein, the experiment lab allows ML experiments to be more easily specified from an external customizer, like a network request or a graphical interface.

## Usage

To use the program, run from the command line: `python3 lab.py configuration_file num-trials`

`configuration-file` is the name of a file, the contents of which are formatted as a python dictionary as described below. The values of the dictionary may be hyperopt spaces such as:  
        
    uniform("label", a, b),  
    randint("label", a, b),  
    choice("label", ["option1","option2"]),  
    normal("label", m, s)  
  
`num-trials` may be omitted, and a default value of 1 will be used. It is an integer that specifying the number of runs to make.
There also need to be a file called filenames.json in the same folder as the program, with key value pairs for dataset aliases and dataset paths, like so:  
    
    {  
    "Dataset 1": "/directory/directory/dataset_a.csv",  
    "Dataset 2": "/directory/directory/dataset_b.csv",  
    ...  
    } 
    
In the same folder as the program, there will be a folder called `results`, in which the program will produce a `models` folder, a `log.txt`, and a `terse_log.txt`. It will also produce a `temp_models` folder and a `temp.txt` temporarily while running. 

### Loading Models

There is also an option to run already trained models. Once you have run `python3 lab.py configuration_file num_trials`, you may run from the command line:

    python3 detection.py configuration_file

To run experiment again, with the already trained model.

### GUI for Generating Configs

There is also a graphical user interface for generating config files. To use it, you can run the `server.py` file to start a local server, and open localhost:8000 in a browser to view the GUI. In the current version, when the "generate" button is clicked, a config structure will be displayed. You can copy this into a text file and continue from there. However, in the `cgi-bin` folder, the `handle_run.py` file can be modified to accomplish arbitrary computation with the submitted config data. 

## Configuration Format

The configuration is a Python dictionary with the following fields (the '|' denotes a choice between mutliple options; each configuration includes only one of the choices). 

    { "run": {
        "datasets": list of dataset aliases to comprise the dataset used,
        "processing": {
            "features": sublist specification string, specifying which features of the dataset should be used,
            "label_column": name of column indicating which data is normal,
            "normal_label": value in label_column indicating normal data,
            "categorical": {
                "features": sublist specification string, specifying which features are categorical,
                "index": boolean, specifying whether categorical features should be indexed,
                "reduce": {
                    "method": "mod" | "cluster" | "hash",
                    "num": integer, specifying the parameter for the reduction method }
                    | {
                    "method": "none" },
                "onehot": boolean, specifying whether categorical features should be one-hot encoded },
            "scaled": "none" | "SD" | "minmax" | "maxabs" },
        "model": {
            "type": "kmeans",
            "k": int hyperparameter,
            "threshold": float hyperparameter,
            "metric": "dot" | "dist" | "cos" specifying the metric used to compute anomalousness }
            | {
            "type": "ocsvm",
            "kernel": "rbf" | "linear" | "poly" | "sigmoid" specifying the kernel function,
            "nu": float hyperparameter,
            "gamma": float hyperparameter }
            | {
            "type": "autoencoder",
            "encoding_size": int hyperparameter, specifying the width of the encoded layer,
            "extra": int hyperparameter, specifying the width of an extra layer
                     (on either side of the encoded layer), 0 for no such extra layer,
            "epochs": int hyperparameter
            "optimizer": "sgd" | "adam"
            "learning_rate": float hyperparameter
            "metric": "dot" | "dist" | "cos" specifying the metric used to compute anomalousness,
            "threshold" float hyperparameter },
        "data_management": {
            "test": float, specifying the proportion of the data used for testing,
            "use_all": boolean, specifying whether the abnormal data in the training portion
                       set should be added to the test set instead of discarded },
        "hyperloss": either none, if hyperopt is not being used,
                     or the name of the metric used for hyperopt's loss function,
        "bag_num": int, specifying the number of bags used (1 for normal run),
        "bag_size": int or "all", specifying the number of features per bag ("all" for normal run) } }
    
## Warnings and Flaws 

- Hyperopt performs each run independently, even when the models and data would be the same, and the only variation is in, say, the threshold. This theoretically harms performance, so a future, rather difficult refactoring might be worth the effort, depending on the use cases. 

- When using feature bagging, only the last model is persisted and logged.

- The autoencoder's loss is not configurable at the moment, so it will train to optimize mean squared error, even when the anomalousness metric is not Euclidean distance.

- The cosine metric and the hashing feature reduction options are currently not implemented. Standard scaling gives lots of NaN entries in the dataset we are currently using.

- The `lab.py` and the `detection.py` files currently contain a large overlap in code, which makes future edits very difficult since they would have to be implemented twice (violating the Don't Repeat Yourself principle).

- The gui currently cannot produce config files with hyperopt spaces.

- The within the analyze function, the box measure is customizable with different weightings. But that customization is not available to the user from the config file.

## Logging and Persisting

There are four output artifacts of a run in the experiment lab. First, for every single model that is constructed, information about its stages and metrics is printed to the terminal. Second, there is a `models` folder containing a directory for each "winner" model: the model with the lowest hyperloss found during one run of the script. Third, there is a `log.txt` file containing a json formatted string for each "winner" model, separated by blank lines. Fourth, there is an `all_log.txt` with a log entry for each model, not just the winners. 

The models folder contains a folder for each model, with the naming format: `type_of_model`YYYY-MM-DD_HH:MM:SS, e.g. `autoencoder_2020-07-29_14:47:18`. It is a parquet formatted folder for keras models, and for sklearn models it is just a folder containing a file named "`type_of_model`model" serialized using joblib. 

## Dependencies

The dependencies are listed in `requirements.txt`, and can be installed using:

    pip install -r requirements.txt

## Examples

Here is an example filenames.json:

    {
    "Jan": "/data/dataset_collected_january.csv",
    "Feb": "/data/dataset_collected_February.csv",
    "Mar 1": "/data/dataset_collected_march1-15.csv",
    "Mar 2": "/data/dataset_collected_March16-31.csv",
    "Apr": "/data/dataset_collected_april.csv",
    }

Here is an example configuration file that work with it:

    { "run": {
            "datasets": ["Mar 1", "Mar 2"],
            "processing": {
                "features": "all",
                "label_column": "Label",
                "normal_label": "BENIGN",
                "categorical": {
                    "features": "none",
                    "index": True,
                    "reduce": {
                        "num": 20, 
                        "method": "mod"},
                    "onehot": True},
                "scaled": "minmax"},
            "model": {
                "type": "kmeans",
                "k": randint("k",4,20),
                "metric": "dist",
                "threshold": uniform("t",0.5,2.5) },
            "data_management": {
                "test": 0.2,
                "use_all": False},
            "hyperloss": "box",
            "bag_num": 1,
            "bag_size": "all" } }
            
More can be found in the `examples` file.

## Extensionality

The ExLab initially contains only three models, but its architecture is suited to other machine learning tasks. To add more anomaly detection models, simply add more if statements of the form: `elif model_configuration["type"] == "...": `. Anyone can design a custom configuration structure, and acccess the fields in model_configuration within the code. Simple analogy with the other model type conditions will show any extensor of the code what essential actions need to be taken in that conditional: initializing a model according to hyperparameters, training a model on `limited_trainDf`, persisting the model in the `results/temp_models` folder, and assigning the `testDf["temp_distances"]` column to be the anomaly scores produced by the model on the limited_testDf.

Updating the code to allow for arbitrary classifiers will not require much, other than a conditional treatment of `trainDf` around line 150. To include regressors, the bagging procedure must be updated, and `testDf["predictions"]` should be defined and analyzed slightly differently. This fork needs only happen once; it will not be too hard, and from then on arbitrary regressors may be included. 
