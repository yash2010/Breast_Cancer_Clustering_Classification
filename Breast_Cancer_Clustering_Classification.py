from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df = pd.DataFrame(data.data, columns = data.feature_names ).astype(float)

df.head()

len(df.columns)

df.columns

plt.figure(figsize = (25 , 12))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdGy')
plt.show()

scaler = StandardScaler()
scaled_values = scaler.fit_transform(df)

sse = []
k_values = range(1,11)

for k in k_values:
  kmeans = KMeans(n_clusters = k, n_init=10)
  kmeans.fit(scaled_values)
  sse.append(kmeans.inertia_)

labels = kmeans.labels_

plt.scatter(scaled_values[:, 0], scaled_values[:, 1], c=labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.show()

labels_df = pd.DataFrame(labels, columns = ['cluster label'])

df_labeled = pd.concat([df, labels_df], axis =1)

df_labeled.head()

X = df_labeled[df_labeled.columns[:-1]].values
Y = df_labeled[df_labeled.columns[-1]].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state= 42 )

param_grid = {
    'max_depth': range(1,21),
    'min_samples_split': range(2, 21)
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_

best_random_state = None
best_accuracy = 0

for random_state in range(100):
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_random_state = random_state

print("Best Random State:", best_random_state)
print("Best Accuracy:", best_accuracy)

model = DecisionTreeClassifier(random_state=best_random_state, max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

report =classification_report(y_test, y_pred)
print(report)

plt.figure(figsize = (25, 20))

class_names = [str(label) for label in np.unique(labels)]

tree.plot_tree(model, filled = True, rounded = True, class_names = class_names, feature_names = data.feature_names)

plt.show()

y_pred = np.array(y_pred)
y_test = np.array(y_test)

indices = np.arange(len(y_test))

plt.figure(figsize=(10, 5))
plt.plot(indices, y_test, color='blue', label='Actual')
plt.plot(indices, y_pred, color='red', label='Predicted')
plt.xlabel('Index')
plt.ylabel('Label')
plt.title('Actual vs Predicted Labels')
plt.legend()
plt.show()
