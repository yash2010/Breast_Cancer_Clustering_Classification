# Breast Cancer Clustering and Classification 🦠
This repository contains a project that uses clustering and classification techniques to analyze the breast cancer dataset. The project involves the following steps:

1. Data visualization and preprocessing
2. KMeans clustering
3. Hyperparameter tuning for RandomForestClassifier using GridSearchCV
4. Decision tree classification
5. Visualization of the decision tree and prediction results

## Dataset 📃
The dataset used for this project is the Breast Cancer dataset from sklearn.datasets. The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, including:

+ Mean radius
+ Mean texture
+ Mean perimeter
+ Mean area
+ Mean smoothness
+ ...and many other features (30 in total)

## Files 📁
Breast_Cancer_Clustering_Classification.py: Python script containing the implementation of the clustering and classification model.

## Requirements
+ Python 3.6 or higher
+ Pandas
+ NumPy
+ Scikit-learn
+ Seaborn
+ Matplotlib

You can install the required libraries using the following command:

```sh
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Usage
1. Clone the repository:
```sh
git clone https://github.com/yash2010/Breast_Cancer_Clustering_Classification.git
```
2. Navigate to the project directory:
```sh
cd Breast_Cancer_Clustering_Classification
```
3. Run the Python script:
```sh
python Breast_Cancer_Clustering_Classification.py
```
## Steps

### Data Visualization and Preprocessing
+ The script loads the Breast Cancer dataset using load_breast_cancer from sklearn.datasets.
+ The dataset is converted to a Pandas DataFrame and a heatmap of feature correlations is plotted.
+ Features are scaled using StandardScaler to standardize the data for clustering.

### KMeans Clustering
+ KMeans clustering is performed with a range of cluster values (1 to 10) to determine the optimal number of clusters.
+ The sum of squared errors (SSE) for each k is calculated and plotted.
+ Data points are plotted with cluster labels assigned by KMeans.
### Hyperparameter Tuning for RandomForestClassifier
+ The labeled dataset is split into training and test sets.
+ GridSearchCV is used to find the best hyperparameters for the RandomForestClassifier.
+ The best random state and accuracy are determined by fitting a DecisionTreeClassifier with different random states.
### Decision Tree Classification
+ A DecisionTreeClassifier is trained using the best hyperparameters and random state.
+ Predictions are made on the test set and a classification report is generated.
+ The decision tree is visualized using tree.plot_tree.
### Prediction Results Visualization
Actual vs Predicted labels are plotted to visualize the model's performance.

## Example
After running the script, the system will:
  + Display a heatmap of feature correlations.
  + Perform KMeans clustering and plot the results.
  + Conduct hyperparameter tuning for RandomForestClassifier and determine the best random state.
  + Train a DecisionTreeClassifier and generate a classification report.
  + Visualize the decision tree and plot the actual vs predicted labels.
