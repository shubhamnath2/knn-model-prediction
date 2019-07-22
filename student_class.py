import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#reading data
df=pd.read_csv("/home/shubham/Desktop/iris/student_data.csv")
df['Result']=np.where((df['Physics']+df['Chemistry']+df['Maths'])/3>=4,'P','F')

print(df)

#transforming the data into array 2D
x=df.iloc[:, 1:4].values.reshape(-3,3)
y=df.iloc[:, 4].values.reshape(-1,1)

print(x)
print(y)

#spliting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=4)

print(x_train)
print(y_train)

#classification
clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train.ravel())

#prediction
y_pred = clf.predict(x_test)

#accuracy
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

while True:

    try:
        name = input("Name: ")
        physics = float(input("Physics Marks: "))
        chemistry = float(input("Chemistry Marks: "))
        maths = float(input("Maths Marks: "))

        assert physics>=0 and physics<=10 \
            and chemistry>=0 and chemistry <=10 \
            and maths>=0 and maths <=10

        subject_value = [physics, chemistry, maths,]
        print(subject_value)

        expected_class=clf.predict([subject_value])
        print(expected_class[0])

    except:
        print("Data is Inaccurate\n")
    else:
        pass