import pandas as pd
from sklearn.linear_model import LogisticRegression

iris_data = pd.read_csv('iris.csv')
# print(iris_data.head())

X = iris_data.drop(columns=['Id','Species'])
Y = iris_data['Species']

model = LogisticRegression()
model.fit(X.values,Y)

predictions = model.predict([[5.7,3.0,4.2,1.2]])
print(predictions)