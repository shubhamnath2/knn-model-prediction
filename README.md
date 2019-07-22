# knn-model-prediction
Using KNN classification to predict the class of given input

This model first trained using KNN to classify the class and tested for accuracy. Then asks the user input to classify the given input. Let's say the data to train and test is-
[4,2,3]->A
[1,6,4]->A
[10,11,31]->B

Give user input like [45,99,19] results B. Here 1 digit results A and 2 digit results B. The above example is for understanding the classification process in general. KNN uses euclieden distance to determine the class.

Warning: The data used here is 100% accurate so it will predict class perfectly, which is almost impossible in real life data.

Configuration: You need virtualenv, PyCharm or any other Python GUI to run this model, as this model requires virtualenv to use. PyCharm is prefered with Python3.6 with all the required modules/libraries in venv folder.

References-
https://www.edureka.co/blog/k-nearest-neighbors-algorithm/
