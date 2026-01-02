# EVALUATION-OF-TREE-BASED-CLASSIFIERS-AND-THEIR-ENSEMBLES

## PROJECT OVERVIEW

This project looks at tree-based classifiers and their ensemble methods. It uses synthetic CNF-generated datasets and the MNIST image dataset. The aim is to compare the performance of Decision Trees, Bagging with Decision Trees, Random Forests, and Gradient Boosting across different data conditions. It will also identify which models generalize the best. The classifiers used in the project are:
- Decision Trees
- Bagging with Decision Trees
- Random Forests
- Gradient Boosting
The primary metrics used in the project for measuring performance are: Accuracy and F1 score.

```text

cnf-classification-project/
│
├── all_data/ # Dataset directory
│ ├── train_c300_d100.csv
│ ├── valid_c300_d100.csv
│ ├── test_c300_d100.csv
│ ├── train_c300_d1000.csv
│ ├── valid_c300_d1000.csv
│ ├── test_c300_d1000.csv
│ ├── train_c300_d5000.csv
│ ├── valid_c300_d5000.csv
│ ├── test_c300_d5000.csv
│ ├── train_c500_d100.csv
│ ├── valid_c500_d100.csv
│ ├── test_c500_d100.csv
│ ├── train_c500_d1000.csv
│ ├── valid_c500_d1000.csv
│ ├── test_c500_d5000.csv
│ ├── train_c1000_d100.csv
│ ├── valid_c1000_d100.csv
│ ├── test_c1000_d100.csv
│ ├── train_c1000_d1000.csv
│ ├── valid_c1000_d1000.csv
│ ├── test_c1000_d1000.csv
│ ├── train_c1000_d5000.csv
│ ├── valid_c1000_d5000.csv
│ ├── test_c1000_d5000.csv
│ ├── train_c1500_d100.csv
│ ├── valid_c1500_d100.csv
│ ├── test_c1500_d100.csv
│ ├── train_c1500_d1000.csv
│ ├── valid_c1500_d1000.csv
│ ├── test_c1500_d1000.csv
│ ├── train_c1500_d5000.csv
│ ├── valid_c1500_d5000.csv
│ ├── test_c1500_d5000.csv
│ ├── train_c1800_d100.csv
│ ├── valid_c1800_d100.csv
│ ├── test_c1800_d100.csv
│ ├── train_c1800_d1000.csv
│ ├── valid_c1800_d1000.csv
│ ├── test_c1800_d1000.csv
│ ├── train_c1800_d5000.csv
│ ├── valid_c1800_d5000.csv
│ └── test_c1800_d5000.csv
│
├── decisionTree.ipynb # Main code file
│
├── gradient_boosting_results.json # Results output
│
├── dt_results.json # Results output │
├── bagging_results.json # Results output
│
├── mnist_results.json # Results output
│
├── random_forest_results.json # Results output
│
├── README.md # Project documentation
│
├── REPORT.md # Experimental report
│
└── AI_TRANSCRIPT.md # AI conversation transcript

```

## **DATASETS**

1. CNF Synthetic Datasets: Five CNF formulas of size 500 variables and with varying clause sizes {300, 500, 1000, 1500, 1800} were created. For every formula, positive and negative examples of sizes 100, 1000, and 5000 were randomly selected. The datasets are divided into train, validation, and test splits. All the features are binary (0/1), and the label is 1 if the assignment to the variable is a model of the CNF formula and 0 otherwise.
2. MNIST Dataset: The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (28×28 grayscale). It is utilized to test the classifiers on a standard real-world image classification task.

## **DEPENDENCIES**

This project is written in Python 3.9+ and the requirements are:
● Python 3.9+
● Python library numpy for numerical computing
● Python library pandas for cleaning and transforming data
● Python library scikit-learn
The dependencies can be used by installing them using this code:
pip install numpy pandas scikit-learn matplotlib tqdm


## **INSTALLATION AND SETUP**

1. Download the project files into a local folder.
2. Ensure all dependencies are installed using the command above.
3. Keep your CNF datasets (if provided) in a ‘datasets/’ folder following the naming convention mentioned in the project file.
The filenames should follow this convention:
train_c[i]_d[j].csv
valid_c[i]_d[j].csv
test_c[i]_d[j].csv
For MNIST, the dataset will be automatically downloaded.

## **HOW TO RUN THE PROJECT**
To run the experiment, use: python decisionTree.py

## **STEPS USED IN THIS PROJECT - MNIST**

1. Data Loading & Preprocessing
- MNIST data loaded with fetch_openml("mnist_784")
- Pixel intensities normalized to interval [0, 1]
- Split into training (60,000) and test (10,000)

2. Model Training
Four classifiers trained sequentially:
- DeciSion Tree
- Bagging Classifier
- Random Forest
- Gradient Boosting

3. Hyperparameter Tuning
- Custom parameter grids defined for each model
- Iterative search using combinations of hyperparameters
- Best combination chosen based on test accuracy

4. Evaluation
- Compare models on the test set with Accuracy as the top metric
- Store best hyperparameters, accuracy, and training time

5. Result Summary
Print out a table summarizing the results
Identify the highest performing classifier
Save results to mnist_results.json


## **METRICS USED**

- Accuracy - provides fraction of correctly classified samples
- F1 Score - provides harmonic mean of precision and recall, useful for imbalanced data
- Training Time - provides total time (in seconds/minutes) taken to train and evaluate each model

## **FUTURE IMPROVEMENTS**
- Implement automated hyperparameter search using GridSearchCV or RandomizedSearchCV
- Add visualizations for confusion matrices and accuracy comparisons
- Extend analysis to include precision, recall, and ROC-AUC metrics
