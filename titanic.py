import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb  

# print(sb.get_dataset_names()) # seaborn dataset list; CSV files can be obtained from https://github.com/mwaskom/seaborn-data

# Titanic data set
data = sb.load_dataset('titanic')
# print(data)
# print(data.columns.values)

# Dependent variables (X) : pclass, sex, age, sibsp, parch, fare, who, adult_male, alone
# Independent variable (Y) : survived

# =========================================================================================================
# Prediction of survival in Titanic tragedy

# 1 Labelling
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

data['sex'] = label.fit_transform(data['sex'])
# print(label.classes_)           # to know which one represents which value: ['female' 'male']

data['who'] = label.fit_transform(data['who'])
# print(label.classes_)    # ['child' 'man' 'woman']

data['adult_male'] = label.fit_transform(data['adult_male'])        # actually 'adult_male' and 'alone' not necessarily need one hot encoder since it contains only boolean
# print(label.classes_)   # [False  True]

data['alone'] = label.fit_transform(data['alone'])
# print(label.classes_)   # [False  True]

data = data.drop(
    ['embarked', 'class', 'deck', 'embark_town', 'alive'],
    axis=1
)

# print(data.head(5))
# print(data['age'].isnull())         # checking which age is a NaN data --> drop it!
data = data.dropna(subset = ['age'])
# print(len(data))

# Split: feature X & target Y
x = data.drop(['survived'], axis=1)
# print(x)
# print(x.iloc[0])
y = data['survived']
# print(y)

# 2. One Hot Encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

coltrans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1, 6])],            # 'adult_male' and 'alone' columns doesn't need to be transformed bcs the data is already in boolean mode (True and False) and python had understood that 1 == True and 0 == False
    remainder = 'passthrough'                               # ^ is the indexes of columns that's going to be transformed: sex, who
)                                                         

x = np.array(coltrans.fit_transform(x))
# print(x[0])         # [ 0.    1.    0.    1.    0.    3.   22.    1.    0.    7.25  1.    0.  ]   --> the first number (0 1) is the sex column, meanwhile the next numbers ( 0 1 0 ) is the who columns transformed
# read:                fm    male  chld  man   wmn  pclass age  sibsp  parch  fare adultM alone


# Splitting
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    x,
    y,
    test_size = .1
)

# print(xtrain[0])
# print(ytrain.iloc[0])
# print(xtest[0])
# print(ytest.iloc[0])

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'liblinear')
model.fit(xtrain, ytrain)

print(round(model.score(xtest, ytest) * 100, 2), '%')
print(xtest[0])
print(ytest.iloc[0])
print(model.predict(xtest[0].reshape(1, -1)))

# Self prediction
# fm    male  chld  man   wmn  pclass age  sibsp  parch  fare adultM alone
print(model.predict([[0, 1, 0, 1, 0, 2, 23, 1, 2, 200, 1, 0]]))


from sklearn.externals import joblib
joblib.dump(model,'modelTitanic')