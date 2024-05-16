# importing the necessary libraries

import pandas
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# loading the dataset
path = os.path.join("..", "dataFiles", "ObesityDataSet_raw_and_data_sinthetic.csv")
data = pandas.read_csv(path)

# converting strings to numerical values

mapping = {"Male": 1, "Female": 0}
mapping2 = {"yes": 1, "no": 0}
data["Gender"] = data["Gender"].map(mapping)
data["SMOKE"] = data["SMOKE"].map(mapping2)
# extract the input and output
# print(data) nvm this
# X = data.filter(items= ["Age","Gender","Height","Weight","SMOKE","FAF","TUE"])
X = data[["Age", "Gender", "Height", "Weight", "SMOKE", "FAF", "TUE"]]
# print(X) nvm
y = data["NObeyesdad"]
# y = data.filter(items= ["NObeyesdad"])
# print(y) nvm

# spliting data into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# creat the svc model
model = SVC(kernel="linear")

# Training or fitting
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("accuracy:", accuracy * 100, "%")
