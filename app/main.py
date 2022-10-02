from utils import *

df_campaign = pd.read_csv('data/Campaign.csv')
df_mortgage = pd.read_csv('data/Mortgage.csv')

parameters = {'max_depth': range(2,5),
             'learning_rate':[0.1,0.01],
             'min_child_weight':[0.5,1.5],
             'subsample':[0.5,0.6,1]}


def main(df_campaign,df_mortgage,_nlarg,params,metric,model_id):
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
	m.create_model(params,metric,model_id)

if __name__ == "__main__":
	main(df_campaign,df_mortgage,parameters,5,'f1','XGB.json')