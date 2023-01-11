from flask import Flask, request, jsonify
import logging
import xgboost as xgb

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/',methods=['POST'])
def predict():
	try:
	    data = request.form['message']
	    my_prediction = model.predict(data)
	    return my_prediction
	except Exception:
		return jsonify('error')

if __name__ == '__main__':
	try:
		model_xgb = xgb.Booster()
		model_xgb.load_model("XGB.json")
		logging.info('Model loaded')
		logging.info('App starting')
		app.run()
	except Exception as e:
		logging.info(str(e))	