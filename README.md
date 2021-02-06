# Million Songs Decade Prediction

## Project Description:
------------------------------------
This project aims to apply various machine learning algorithms to classify the songs by the decade in which they were released. Multi-layer neural network (NN), support vector machine (SVM), and AdaBoost schemes were employed. For the detailed report of this project, please refer to the attached [file](report/report.pdf). 

## To use this software:
------------------------------------
1. Install the libraries in [requirements.txt](requirements.txt) to be able to run the scripts. This can be done by: 
    ```bash
    pip install -r requirements.txt 
    ```  
2. Go to the `Source` directory.
3. Run the following command to perform **Neural Network**, **SVM** and **Adaboost** classifiers to predict the decade of the songs:

    ```bash
	python3 NN_master.py <input_file> <entries_to_skip> 
    python3 svm.py <input_file> <entries_to_skip> 
    python3 adaboost.py <input_file> <entries_to_skip> 
    ```  

* For example:

    ```bash
	python3 NN_master.py "reduced.txt" 0
    ```  

A demo for a reduced 515 datasets is provided inside the `Testing` directory: 

1. Go to the `Testing` directory
2. Run the command below:
    ```bash
	python3 demo.py
    ```  

Sample input file is provided in `reduced.txt` in the `Testing` directory. Output for the neural network, SVM and adaboost analyses are logged in `NN.log`, `SVM.log` and `Adaboost.log`. Optimal number of neurons, layers and hyperparameters are also plotted and outputted in [`neuron_layers_contour.pdf`](test/neuron_layers_contour.pdf) and [`hyper_test.pdf`](test/hyper_test.pdf). 

## Data analysis steps:
------------------------------------
Database used is from [Columbia Labrosa Laboratory](http://millionsongdataset.com) million song [dataset](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd). Instructions for getting the full dataset can be found in [here](https://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset). Data set is provided with a series of `.hd5` files, which need to be cleaned and processed before starting the analysis. A series of scripts provided in the `lib` directory to extract feature information as well to manipulate the data [here](lib/). 

Data preprocessing and cleaning are as follows: 
1. **Aggregation**: Data for each song is provided by a unique `.hd5` file and grouped based on the alphabetic order. First, we compike all the `.hd5` files into a single file.  This process can be done by modifying `maindir` variable where the raw `.hd5` data resides and the `output` variable which dumps the aggregated file in [`create_aggregate_file.py`](lib/hd5_aggregation/create_aggregate_file.py) from `lib` directory. 

2. **Data extraction to a Numpy array**: Next, the output data is transformed to an `.npy` format to be read by `numpy` package for further processing. This is being done by using [`hdf5_getters_mod.py`](lib/hd5_aggregation/hdf5_getters_mod.py) routine from `lib` directory. The songs for which the release year is not specified are not considered in the analyses.

3. Features used for each entry is listed in [here](features.md). These song features include sample rate, digital id, latitude, danceability, etc. 


## Feedback, bugs, questions 
-------------------------------
Please reach out to me by email to shahriarhoushmand@gmail.com for any inquiry. Comments and feedbacks are greatly appreciated. 
