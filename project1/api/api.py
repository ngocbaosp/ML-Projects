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

# API server
app = Flask(__name__)


# Define end point
@app.route('/project1/api/v1.0/predict', methods=['GET'])
def get_prediction():
    # We are using 3 features. For example:
    # subject line, word frequency, etc
    print("params ", request.args)
    genre = request.args.get('genre')
    mtype = request.args.get('type')
    episode = int(request.args.get('episode'))
    members = int(request.args.get('members'))

    # Load model from disk
    # model = pickle.load(open('randomForestClassifier.pkl', 'rb'))

    # Predict
    # pred = model.predict([[genre, mtype, episode, members]])[0]
    pred = 0
    result = {
        'genre': genre,
        'type': mtype,
        'episode': episode,
        'members': members
    }
    if pred == 0:
        result['rating'] = "Low rating"
    else:
        result['rating'] = "High rating"
    return result


# Main app
if __name__ == '__main__':
    app.run(port=7777, host='0.0.0.0', debug=True)
