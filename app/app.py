from flask import Flask, request
import logging
import xgboost as xgb

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/',methods=['POST'])
def predict():
    data = request.form['message']
    my_prediction = model.predict(data)

    return my_prediction


if __name__ == '__main__':
	model_xgb = xgb.Booster()
	model_xgb.load_model("XGB.json")
	logging.info('Model loaded')
	logging.info('App starting')
	app.run()	