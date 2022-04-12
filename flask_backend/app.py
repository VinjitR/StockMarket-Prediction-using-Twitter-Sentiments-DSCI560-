import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model

# instantiate flask 
app = flask.Flask(__name__)


# define a predict function as an endpoint 
@app.route("AAPL/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
    # return a response in json format 
    return flask.jsonify(data)    

# start the flask app, allow remote connections 
app.run(host='0.0.0.0')