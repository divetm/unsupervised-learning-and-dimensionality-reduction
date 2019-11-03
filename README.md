# Unsupervised Learning and Dimensionality Reduction
*Assignment #3 - CS 7641 Machine Learning course - Charles Isbell & Michael Littman - Georgia Tech*

Please clone this git to a local project if you want to replicate the experiments reported in the assignment paper.

Virtual Environment
----
This project contains a virtual environment folder ```venv``` (the folder is too heavy for Github - click on the link [here](https://drive.google.com/drive/folders/1rMRTMel6CY66KMoxxhyu1D7r75wXKR3z?usp=sharing) to download it from my Google Drive). This folder contains all the files needed to create a virtual environment in which the project is supposed to run.

requirements.txt
----
This file contains all the necessary packages for this project. (Running ```pip install -r requirements.txt``` will install all the packages in your project's environment - should not be necessary if you are using the given ```venv```folder here)

The dataset
----
This datasets (```tumor_classification_data.csv```) is one of the datasets described in the assignment paper. It can be downloaded from its original source:
* https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

digits_problem.py and tumor_problem.py
----
These Python files are the implementations of the 2 clustering algorithms (K-means, EM) and 4 dimensionality reduction algorithms (PCA, ICA, RCA, LDA) studied in this assignment over the two datasets that we use for this project. They are almost identical but the way the data is prepared for the algorithms and the hyperparameters of each algorithm differ from one script to the other because of the differences in the datasets.
What they do is:
- load the dataset
- prepare the data before running it through the algorithms (we use the ```scikit-learn``` algorithms in this project: it is important to follow the requirements of these algorithms for the script to work)
- plot different graphs that allow to choose the best hyperparameters for the K-means, EM, PCA, ICA, RCA and LDA algorithms
- plot 2D representations of the clustering and reduction results on both datasets (reduction to 2 dimensions done by PCA/ICA/RCA/LDA and then clustering done by K-means/EM)
- compute the accuracy of the clustering compared to the labels given in the original datasets.

In ```tumor_problem.py```, a Neural Network learner (the one defined in the first assignment) is tested on the reduced/clustered tumor classification dataset. To do so, the script:
- splits the data into training and testing sets
- reduces the training dataset (with PCA/ICA/RCA/LDA), trains the Neural Network on the reduced data and tests it with the testing reduced dataset
- clusters the training dataset (with K-means/EM), trains the Neural Network on the data using the clusters as an added feature and tests it with the testing dataset
