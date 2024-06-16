# Bank Marketing Classifier Comparison

This project aims to compare the performance of different classifiers, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines, using a dataset related to marketing bank products over the telephone. The goal is to predict whether a client will subscribe to a term deposit.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Understanding and Preprocessing](#data-understanding-and-preprocessing)
- [Visualizations](#visualizations)
- [Model Comparison](#model-comparison)
- [Outcomes](#outcomes)
- [Contributing](#contributing)

## Dataset

The dataset used in this project is the "Bank Marketing Dataset" (https://archive.ics.uci.edu/dataset/222/bank+marketing), which is available from the UCI Machine Learning Repository. The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (variable y).

## Installation

1. Clone the repository:
    git clone https://github.com/harpreetsohal1/bank-marketing-classifier.git

2. Change into the project directory:
    cd bank-marketing-classifier


## Usage

1. Load the dataset and understand its structure.
    import pandas as pd
    df = pd.read_csv('data/bank-additional/bank-additional-full.csv', sep=';')
    df.head()

2. Follow the steps outlined in the Jupyter Notebook (`Bank_Marketing_Classifier_Comparison.ipynb`) to preprocess the data, build and compare different classifiers, and visualize the results.

3. To run the entire analysis, execute the cells in the provided Jupyter Notebook.

## Data Understanding and Preprocessing

1. **Feature Engineering**:
   - Handle missing values.
   - Encode categorical variables.
   - Scale numerical variables.

2. **Train/Test Split**:
   - Split the dataset into training and testing sets.

3. **Baseline Model**:
   - Establish a baseline accuracy using the most frequent class.

4. **Logistic Regression Model**:
   - Build and evaluate a basic Logistic Regression model.

5. **Model Comparisons**:
   - Compare the performance of Logistic Regression, KNN, Decision Tree, and SVM models using accuracy and training time.

## Visualizations

Appropriate visualizations are created to understand the data distribution and relationships between features. Some of the visualizations include:
- Age Distribution
- Box Plot for Age by Subscription Status
- Correlation Matrix for Continuous Variables
- Job Distribution
- Marital Status Distribution
- Education Level Distribution
- Subscription Status by Job, Marital Status, and Education Level

## Model Comparison

The models compared in this project include:
- K Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)

Each model's performance is evaluated based on training time, training accuracy, and test accuracy.


## Outcomes

**Baseline Accuracy**: 0.8876

### Logistic Regression Model

- Train Accuracy: 0.9013
- Test Accuracy: 0.8971

### Model Comparison

| Model                 | Train Time (s) | Train Accuracy | Test Accuracy |
|-----------------------|----------------|----------------|---------------|
| KNN                   | 0.071579       | 0.914264       | 0.888080      |
| Logistic Regression   | 0.194977       | 0.901275       | 0.897062      |
| Decision Tree         | 0.235515       | 0.995357       | 0.836125      |
| SVM                   | 34.341545      | 0.904765       | 0.896941      |

### Improving the Model

- **Best parameters found**:  {'classifier__max_depth': 5, 'classifier__min_samples_split': 2}
- **Best cross-validation score**:  0.9012443095599393
- **Best Decision Tree Test Accuracy**: 0.8967

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you have any suggestions or improvements.
