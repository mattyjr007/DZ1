import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13 ].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X =X[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# building the ann
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense

#adding input layers and first hifdden layer
model = keras.Sequential([
        Dense(6,activation =tf.nn.relu,input_shape = [11]),
        Dense(6,activation =tf.nn.relu),
        Dense(1,activation =tf.nn.sigmoid)
        ])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss = 'binary_crossentropy',
              metrics = ['accuracy']
              )

model.fit(X_train,y_train,batch_size=10,epochs=100)

test_loss , test_acc  = model.evaluate(X_test,y_test)
print(test_acc)
# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred =(y_pred >0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)