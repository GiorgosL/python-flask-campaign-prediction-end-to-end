from utils import *
import configparser
import json

config = configparser.ConfigParser()
config.read('data/config.ini')

df_campaign = config['data']['campaign']
df_mortgage = config['data']['mortgage']
parameters = {'max_depth': json.loads(config.get("hyperparams","max_depth")),
			'learning_rate':json.loads(config.get("hyperparams","learning_rate")),
			'min_child_weight':json.loads(config.get("hyperparams","min_child_weight")),
			'subsample':json.loads(config.get("hyperparams","subsample"))}
n_folds = config['booting_params']['n_folds']
cv_metric = config['booting_params']['cv_metric']
model_name = config['booting_params']['model_name']


def main(df_campaign,df_mortgage,params,_nlarg,metric,model_id):
	d = DataLoader(df_campaign,df_mortgage)
	final_df = d.preprocess_merge_serve()
	impute = Imputer(final_df)
	le, df = impute.impute_data()
	opt = OptimizeData(df)
	opt.feature_importance(_nlarg)
	df_new = opt.add_features()
	df_new2 = drop_outliers(df_new)
	data_drift_retrain()
	m=Model(df_new2)
	m.create_model(parameters,metric,model_id)

if __name__ == "__main__":
	main(df_campaign,df_mortgage,parameters,n_folds,cv_metric,model_name)