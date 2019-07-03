import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

# print(sb.get_dataset_names())
data=sb.load_dataset('titanic')
df=pd.DataFrame(data)
df=df.fillna({
    'age':20
})
print(df)
print(df.columns)

# Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
    #    'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town',
    #    'alive', 'alone'],

# 2. sklearn one hot encoding
print(df.head(5))
print(df['fare'].head(5))
print(df['embarked'].head(5))


# 2a. labelling
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['sex'] = label.fit_transform(df['sex'])
# print(data['sex'])
print(label.classes_)
print(label.transform(['male', 'female', 'male']))

dfX=df[['pclass', 'sex', 'age', 'fare' ]].values
print(dfX)


dfY=df['survived'].values
print(dfY)

df=df.drop(
    ['sibsp','parch','class', 'who', 'adult_male', 'deck', 'embark_town','alive', 'alone','embarked'],
    axis=1
)
print(df.head(5))

# 2b. one hot encoder       => similar with pandas get dummies
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

coltrans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[2])],        #=>[2]= kolom 'sex' yang mau diubah
    remainder='passthrough'
)
print()

dfX=np.array(coltrans.fit_transform(dfX))
print('dfX')
print(dfX[0])


# linear regression
from sklearn.linear_model import LinearRegression
modelLR = LinearRegression()
modelLR.fit(dfX, dfY)

# print(modelLR.predict([[3,1,20,5.2500]]))
