import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#reading data
df=pd.read_csv("/home/shubham/Desktop/iris/Iris.csv")

#transforming the data into array 2D
x=df.iloc[:, 1:5].values.reshape(-4,4)
y=df.iloc[:, 5].values.reshape(-1,1)

#spliting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

#classification
clf=KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train,y_train.ravel())

#prediction
y_pred = clf.predict(x_test)

#accuracy
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

while True:

    try:
        sepal_length = float(input("Sepal Length in Cm: "))
        sepal_width = float(input("Sepal Width in Cm: "))
        petal_length = float(input("Petal Length in Cm: "))
        petal_width = float(input("Petal Width in Cm: "))

        flower_value = [sepal_length, sepal_width, petal_length, petal_width]
        print(flower_value)

        expected_class=clf.predict([flower_value])
        print(expected_class[0])
    except:
        print("Data Maybe Inaccurate")
    else:
        pass