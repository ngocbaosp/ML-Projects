#!flask/bin/python
# Web server
from flask import Flask
# Get request parameters
from flask import request
# This is needed for logistic regression
# from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
# Save and load models to/from disk
import pickle

# Output is the probability that the given
# input (ex. email) belongs to a certain
# class (ex. spam or not)
clf = linear_model.LogisticRegression()

# Samples (your features, they should be normalized
# and standardized). Normalization scales the values
# into a range of [0,1]. Standardization scales data
# to have a mean of 0 and standard deviation of 1
# Note that we are using fake data here just to
# demonstrate the concept
X = [[1.0, 1.0, 2.1], [2.0, 2.2, 3.3], [3.0, 1.1, 3.0]]

# Labeled data (Spam or not)
Y = [1, 0, 1]

# Build the model
clf.fit(X, Y)

# Save it to disk
pickle.dump(clf, open('randomForestClassifier.pkl', 'wb'))


#----------------------------------

import pandas as pd
import numpy as np
from sklearn.externals import joblib


from sklearn.decomposition import PCA
from sklearn import preprocessing




RawData = pd.read_csv('anime.csv')


# MyPCA
def myPCA(data,n):
    pca = PCA(n_components=n)
    pca.fit(data)
    df = pca.transform(data)
    PCA_Data = pd.DataFrame(df)
    return PCA_Data

# myNormalize
def myNormalize(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    Normalized_Data = min_max_scaler.fit_transform(data)
    Normalized_Data = pd.DataFrame(Normalized_Data)
    return Normalized_Data

# myEncode
def myEncode(data,col):
    NewData_Encode = data.copy()
    NewData_Encode = pd.get_dummies(NewData_Encode, columns=col, prefix = col)
    return NewData_Encode


def myCleanAndTransformData(data):
    # Drop null rows
    NewData = data.dropna()
    # Remove unknown ata
    NewData = NewData[NewData['episodes'] != 'Unknown']

    # Add a new column rating class
    NewData['Class'] = 1
    # 1: High
    # or 0: Low based on rating
    NewData.loc[NewData['rating'] >= NewData['rating'].mean(), 'Class'] = 1
    NewData.loc[NewData['rating'] < NewData['rating'].mean(), 'Class'] = 0

    # Split genre values into rows
    NewData = pd.DataFrame(NewData.genre.str.split(',').tolist(),
                           index=[NewData.anime_id, NewData.type, NewData.episodes, NewData.rating, NewData.members,
                                  NewData.Class]).stack()
    NewData = NewData.reset_index([0, 'anime_id', 'type', 'episodes', 'rating', 'members', 'Class'])
    NewData.columns = ['anime_id', 'type', 'episodes', 'rating', 'members', 'Class', 'genre']

    # Encode type feature: 6 unique values
    NewData = myEncode(NewData, ['type'])

    # Encode genre feature: 82 unique values
    NewData = myEncode(NewData, ['genre'])

    TestData = NewData[NewData['anime_id'] == "ID_TEST"]

    # Drop anmie_id,rating,Class
    NewData = NewData.drop(['rating'], axis=1)
    NewData = NewData.drop(columns=['anime_id'])

    NewData = NewData.drop(columns=['episodes'])

    return NewData,TestData

def getPredictData(params,n,model):

    df_predict = pd.DataFrame(
        {"anime_id": ["ID_TEST"], "name": ["name"], "genre": [params['genre']], "type": [params["type"]], "episodes": [params["episode"]],
         "rating": [10], "members": [params['members']]})

    print(df_predict)

    newDT = RawData.append(df_predict)
    newDT,TestData = myCleanAndTransformData(newDT)
    newDT = myPCA(newDT, n)
    Row = TestData.anime_id.count()
    newDT = newDT.tail(Row)

    print(newDT)

    y_pred_dt = model.predict_proba(newDT)[:, 1]
    # y_pred_dt = model.predict(newDT)

    print(y_pred_dt)

    print(y_pred_dt.mean())

    return y_pred_dt.mean()


#----------------------------------

# API server
app = Flask(__name__)


# Define end point
@app.route('/project1/api/v1.0/predict', methods=['GET'])
def get_prediction():
    # We are using 3 features. For example:
    # subject line, word frequency, etc
    print("params ", request.args)
    genre = request.args.getlist('genre')
    genre = ', '.join(genre)
    mtype = request.args.get('type')
    episode = int(request.args.get('episode'))
    members = int(request.args.get('members'))

    # Load model from disk
    # model = pickle.load(open('randomForestClassifier.pkl', 'rb'))

    # Load model from disk
    model = joblib.load('my_dt_model.pkl')

    # Predict
    # pred = model.predict([[genre, mtype, episode, members]])[0]

    result = {
        'genre': genre,
        'type': mtype,
        'episode': episode,
        'members': members
    }

    pred = getPredictData(result, 40, model)

    if pred < 0.5:
        result['rating'] = "Low rating"
    else:
        result['rating'] = "High rating"

    return result


# Main app
if __name__ == '__main__':
    app.run(port=7777, host='0.0.0.0', debug=True)
