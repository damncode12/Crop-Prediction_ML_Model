# import the required libraries

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
 
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# read the dataset
data = pd.read_csv('Crop_recommendation.csv')

# check the first 5 rows of the dataset
data.head()

data.info()

data.describe()

data['label'].unique()

data['label'].value_counts()

# over all distribution
 
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 60
 
features = ['N', 'P', 'K', 'temperature',
            'humidity', 'ph', 'rainfall']
 
for i, feat in enumerate(features):
    plt.subplot(4, 2, i + 1)
    sns.distplot(data[feat], color='greenyellow')
    if i < 3:
        plt.title(f'Ratio of {feat}', fontsize=12)
    else:
        plt.title(f'Distribution of {feat}', fontsize=12)
    plt.tight_layout()
    plt.grid()
plt.show()

sns.pairplot(data, hue='label')
plt.show()

'''
# Correlation matrix
fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(data.corr(),annot=True,cmap='viridis')
ax.set(xlabel='features')
ax.set(ylabel='features')

plt.title('Correlation between different features',
		fontsize=15,
		c='black')
plt.show()
'''
# Put all the input variables into features vector
features = data[['N', 'P', 'K', 'temperature',
				'humidity', 'ph', 'rainfall']]

# Put all the output into labels array
labels = data['label']

X_train, X_test,\
	Y_train, Y_test = train_test_split(features,labels,test_size=0.2,random_state=42)

# Pass the training set into the
# LogisticRegression model from Sklearn
LogReg = LogisticRegression(random_state=42)\
.fit(X_train, Y_train)

# Predict the values for the test dataset
predicted_values = LogReg.predict(X_test)

# Measure the accuracy of the test
# set using accuracy_score metric
accuracy = metrics.accuracy_score(Y_test,predicted_values)

# Find the accuracy of the model
print("Logistic Regression accuracy: ", accuracy)

# Get detail metrics
print(metrics.classification_report(Y_test,predicted_values))

filename = 'LogisticRegression.pkl'
MODELS = '/home/pegasus/Desktop/'
# Use pickle to save ML model
pickle.dump(LogReg, open(MODELS + filename, 'wb'))

#Predicting the crop for the given input
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Load the saved model
    model = pickle.load(open(MODELS + filename, 'rb'))
    # Predict the output class using the model loaded
    # and the features given as the function arguments
    prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    # Return the predicted class to the user
    return prediction[0]

# Predict the crop for the given input
N=int(input("Enter the value of Nitrogen:"))
P=int(input("Enter the value of Phosphorous:"))
K=int(input("Enter the value of Potassium:"))
temperature=float(input("Enter the value of temperature:"))
humidity=float(input("Enter the value of humidity:"))
ph=float(input("Enter the value of ph:"))
rainfall=float(input("Enter the value of rainfall:"))
print(predict_crop(N, P, K, temperature, humidity, ph, rainfall))

#print(predict_crop(90, 42, 43, 20.8, 82, 6.5, 202.3))
