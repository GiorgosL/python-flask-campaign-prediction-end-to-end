"""
Usage: main.py

Contains all the steps for the end to end modelling pipeline written into clean, reusable modular code.
All the functions are being kept within a few lines of code for easier debugging and apprehending.
Contains classes and fucntions to be imported to main.py.
"""
import pandas as pd
import pickle
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import re
from scipy.stats import chi2
import numpy as np
import logging
import warnings
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import (
    DataDriftTab,
    CatTargetDriftTab
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

class DataLoader:
    def __init__(self,df_camp,df_mort):
        try:
            self.df_camp = df_camp
            self.df_mort = df_mort
            logging.info('Data loaded')
            logging.info('%s %s','len df_campaign :',len(self.df_camp))
            logging.info('%s %s','len df_mortgage :',len(self.df_mort))
        except Exception as e:
            logging.info(str(e))
    
    def find_fill_nan(self):
        try:
            nan_camp = [i for i in self.df_camp.columns if self.df_camp[i].isnull().any()]
            nan_mort = [i for i in self.df_mort.columns if self.df_mort[i].isnull().any()]
            logging.info('%s %s','NaN columns for df_campaign :',nan_camp)
            logging.info('%s %s','NaN columns for df_mortgage :',nan_mort)
            self.df_camp['created_account'].fillna('N/A',inplace=True)
            self.df_camp['name_title'].fillna('',inplace=True)
            logging.info('NaN filled in created_acount')
        except Exception as e:
            logging.info(str(e))
    
    def find_duplicates(self):
        try:
            logging.info('%s %s', 'Total number of duplicates for df_campaign :', self.df_camp.duplicated().sum())
            logging.info('%s %s', 'Total number of duplicated for df_mortgage :', self.df_mort.duplicated().sum())
        except Exception as e:
            logging.info(str(e))
    
    def get_full_name(self):
        try:
            self.df_camp['full_name'] = self.df_camp['name_title'] + ' ' + self.df_camp['first_name'] + ' ' + self.df_camp['last_name']
            self.df_camp['full_name'] = self.df_camp['full_name'].str.strip()
            logging.info('Full name created')
        except Exception as e:
            logging.info(str(e))
    
    def make_bins(self,df):
        try:
            bins = [16, 29, 39, 49, 59, 69, 79, 99]
            label_names = ['20s','30s','40s','50s','60s','70s','above 80s']
            df['age_bracket'] = pd.cut(df['age'], bins, labels=label_names)
        except Exception as e:
            logging.info(str(e))
    
    def drop_columns(self):
        try:
            self.df_camp.drop(['participant_id', 'postcode','company_email',
                              'name_title','first_name','last_name'],axis=1,inplace=True)
            self.df_mort.drop(['paye','new_mortgage','dob','birth_year'],axis=1,inplace=True)
            logging.info('%s %s', 'Number of campaign columns dropped: ', len(['participant_id', 'postcode','company_email',
                              'name_title','first_name','last_name']))
            logging.info('%s %s', 'Number of mortgage columns dropped: ', len(['paye','new_mortgage','dob','birth_year']))
        except Exception as e:
            logging.info(str(e))
    
    def find_mix_types(self,df):
        try:
            for col in df.columns:
                weird = (df[[col]].applymap(type) != df[[col]].iloc[0].apply(type)).any(axis=1)
                if len(df[weird]) > 0:
                    logging.info(col)
        except Exception as e:
            logging.info(str(e))
            
    def check_mix_types(self):
        try:
            logging.info('%s %s', 'Mixed type in campaign :', self.find_mix_types(self.df_camp))
            logging.info('%s %s', 'Mixed type in campaign :', self.find_mix_types(self.df_mort))
        except Exception as e:
            logging.info(str(e))
    
    def create_age(self):
        try:
            self.df_mort['dob'] = pd.to_datetime(self.df_mort['dob'])
            self.df_mort['birth_year'] = self.df_mort['dob'].dt.year
            self.df_mort['age'] = 2022 - self.df_mort['birth_year']
            logging.info('Age created')
        except Exception as e:
            logging.info(str(e))
            
    
    def get_age_bins(self):
        try:
            self.make_bins(self.df_camp)
            self.make_bins(self.df_mort)
            logging.info('Age bins created for both dataframes')
        except Exception as e:
            logging.info(str(e))
    
    def process_salary_band(self):
        try:
            self.df_mort['salary_band'] = self.df_mort['salary_band'].apply(lambda x: float(x.split()[0].replace('£','')) if 'yearly' in str(x) else x)
            self.df_mort['salary_band'] = self.df_mort['salary_band'].apply(lambda x: float(x.split()[0].replace('£',''))*12 if 'month' in str(x) else x)
            self.df_mort['salary_band'] = self.df_mort['salary_band'].apply(lambda x: float(x.split()[0].replace('£',''))*52 if 'pw' in str(x) else x)
            self.df_mort['salary_band'] = self.df_mort['salary_band'].apply(lambda x: float(x.split()[0].replace('£',''))+float(x.split()[2])/2 if 'range' in str(x) else x)
            self.df_mort['salary_band'] = self.df_mort['salary_band'].apply(lambda x: 'crypto' if len(str(x))<=5 else x)
            self.df_mort['salary_band'] = self.df_mort['salary_band'].apply(lambda x: '.'.join(re.findall("\d+", x)) if type(x)==str and len(str(x))>6 else x)
            logging.info('Salary bands are normalised')
        except Exception as e:
            logging.info(str(e))
    
    def merge_dfs(self):
        try:
            self.df_mort['full_name'] = self.df_mort['full_name'].astype(str)
            self.df_camp['full_name'] = self.df_camp['full_name'].astype(str)
            self.df_total = pd.merge(self.df_mort, self.df_camp,'inner','full_name')
            logging.info('Dataframes succesfully marged')
        except Exception as e:
            logging.info(str(e))
            
    def get_code(self):
        try:
            code = []
            for item in self.df_mort['salary_band'].tolist():
                if type(item) ==str and item!='crypto':
                    code.append(re.sub(r'[^a-zA-Z]', '', item))
                else:
                    code.append('GBP')
            self.df_mort['country_code'] = code
        except Exception as e:
            logging.info(str(e))

    def adjust_final_df(self):
        try:
            self.df_total.rename({'age_bracket_y':'age_bracket'},axis=1,inplace=True)
            self.df_total['age_bracket'] = self.df_total['age_bracket'].astype(str)
            self.df_total.drop(['full_name'],axis=1,inplace=True)
            self.df_adjusted = self.df_total.loc[self.df_total['created_account']!='N/A']
            self.df_adjusted.loc[self.df_adjusted['country_code']!='GBP','salary_band']=np.nan
            self.df_adjusted.loc[self.df_adjusted['salary_band']=='crypto','salary_band']=np.nan
            self.df_adjusted.drop(['country_code','dob'],axis=1,inplace=True)
            logging.info('Final datafrane adjusted')
            logging.info('%s %s', 'Length of final df :', len(self.df_adjusted))
            return self.df_adjusted
        except Exception as e:
            logging.info(str(e))
    
    def preprocess_merge_serve(self):
        self.find_fill_nan()
        self.find_duplicates()
        self.get_full_name()
        self.drop_columns()
        self.check_mix_types()
        self.create_age()
        self.get_code()
        self.get_age_bins()
        self.process_salary_band()
        self.merge_dfs()
        df_final = self.adjust_final_df()
        return df_final

class Imputer:
    def __init__(self,df):
        try:
            self.df = df
            logging.info('Data loaded')
            logging.info('%s %s','Dataframe length :',len(self.df))
        except Exception as e:
            logging.info(str(e))
    
    def find_categoricals(self):
        try:
            categoricals = [col for col in self.df.columns if self.df[col].dtype=='O']
            logging.info('%s %s','Categorical columns:',categoricals)
        except Exception as e:
            logging.info(str(e))
    
    def encode_categoricals(self):
        try:
            label_encoder = preprocessing.LabelEncoder()
            self.df = self.df.apply(label_encoder.fit_transform)
            self.X = self.df.drop(['created_account'],axis=1)
            self.y = self.df['created_account'].tolist()
            logging.info('Encoded finished')
            return label_encoder
        except Exception as e:
            logging.info(str(e))
    
    def scale_data(self):
        try:
            scaler = StandardScaler()
            model = scaler.fit(self.X)
            self.scaled_data = model.transform(self.X)
            logging.info('Scaling finished')
        except Exception as e:
            logging.info(str(e))
    
    def impute_salary(self):
        try:
            imputer = KNNImputer(n_neighbors=2)
            self.df_imputed = pd.DataFrame(imputer.fit_transform(self.scaled_data),columns=self.X.columns.tolist())
            self.df_imputed.drop(['age_bracket_x'],axis=1,inplace=True)
            self.df_imputed['created_account'] = self.y
            logging.info('Imputing finished')
            return self.df_imputed
        except Exception as e:
            logging.info(str(e))
            
    def impute_data(self):
        self.find_categoricals()
        label_enc = self.encode_categoricals()
        self.scale_data()
        df_imputed = self.impute_salary()
        return label_enc, df_imputed


class OptimizeData:
    
    def __init__(self,df):
        try:
            self.df = df
            self.X = self.df.drop(['created_account'],axis=1)
            self.y = self.df['created_account']
            logging.info('X, y created')
        except Exception as e:
            logging.info(str(e))
            
    def feature_importance(self,n_larg):
        try:
            model = ExtraTreesClassifier(n_estimators = 100, random_state = 42)
            model.fit(self.X, self.y)
            feat_importances = pd.Series(model.feature_importances_, index = self.X.columns)
            self.features = feat_importances.nlargest(n_larg).index.tolist()
        except Exception as e:
            logging.info(str(e))
            
    def add_features(self):
        try:
            self.features.append('created_account')
            self.df_new = self.df[self.features]
            logging.info('Top features added')
            return self.df_new
        except Exception as e:
            logging.info(str(e))

def mahalanobis(X=None,data=None,cov=None):
    try:
        x_mu = X-np.mean(data)
        if not cov:
            cov = np.cov(data.values.T)
        inv_covmat = np.linalg.inv(cov)
        left = np.dot(x_mu, inv_covmat)
        mahal = np.dot(left,x_mu.T)
        return mahal.diagonal()
    except Exception as e:
        logging.info(str(e))
        

def drop_outliers(df):
    try:
        df_mahal = df.copy()
        df_mahal['Mahalanobis_Dis']=mahalanobis(df.select_dtypes(['float']),df.select_dtypes(['float']))
        df_mahal['p_value'] = 1-chi2.cdf(df_mahal['Mahalanobis_Dis'], df_mahal.shape[1]-3)
        out_idx = df_mahal.loc[df_mahal['p_value']<0.001]['created_account'].index
        df_final = df.drop(index=out_idx)
        del df_mahal
        logging.info('Outliers dropped')
        return df_final
    except Exception as e:
        logging.info(str(e))


class Model:
    def __init__(self,df):
        try:
            self.df = df
            logging.info('%s %s','df len :',len(self.df))
        except Exception as e:
            logging.info(str(e))
        
    def split_data(self):
        try:
            self.X = self.df.drop(['created_account'],axis=1)
            self.y = self.df['created_account']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=43)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=43)
            logging.info('Data split to train, test, validation succesfully')
        except Exception as e:
            logging.info(str(e))
    
    def oversample_data(self):
        try:
            sm = SMOTE(random_state=43)
            self.X_train_sm, self.y_train_sm = sm.fit_resample(self.X_train, self.y_train)
            logging.info('Data resampled succesfully')
        except Exception as e:
            logging.info(str(e))
            
    def initiate_model(self,params=None):
        try:
            self.xgb = XGBClassifier(random_state=43,params=params)
            logging.info('XGB initiated succesfully')
        except Exception as e:
            logging.info(str(e))

    def grid_search(self, params,eval_metric):
        try:
            grid = GridSearchCV(estimator=self.xgb,
                                 param_grid=params,
                                 scoring=eval_metric,
                                 cv=StratifiedKFold(),
                                 verbose=1)
            grid.fit(self.X_train_sm,self.y_train_sm)
            logging.info('%s %s','Best params are: ', grid.best_params_)
            return grid.best_params_
        except Exception as e:
            logging.info(str(e))
            
    def fit_predict(self):
        try:
            self.xgb.fit(self.X_train,self.y_train, eval_metric=['error'],eval_set=[(self.X_train,self.y_train),
                                                                                    (self.X_test,self.y_test)])
            self.y_pred = self.xgb.predict(self.X_test)
            logging.info('Fit predict successful')
        except Exception as e:
            logging.info(str(e))
        
    def binary_classification_performance(self):
        return print(classification_report(self.y_test, self.y_pred))
    
    def plot_loss(self):
        results = self.xgb.evals_result()
        epochs = len(results['validation_0']['error'])
        x_axis = range(0,epochs)
        fig,ax = plt.subplots(figsize=(6,6))
        ax.plot(x_axis,results['validation_0']['error'],label='train')
        ax.plot(x_axis,results['validation_1']['error'],label='test')
        ax.legend()
        plt.ylabel('Classification Error')
        plt.title('Model error')
        plt.show()
    
    def save_model(self,model_id):
        try:
            return self.xgb.save_model('data/'+model_id)
            logging.info('%s %s %s','Model saved at: ', 'data/',model_ide)
        except Exception as e:
            logging.info(str(e))
    
    def create_model(self, params, metric, model_id):
        self.split_data()
        self.oversample_data()
        self.initiate_model()
        best_params = self.grid_search(params, metric)
        self.initiate_model(best_params)
        self.fit_predict()
        self.binary_classification_performance()
        self.plot_loss()
        self.save_model(model_id)

def data_drift_retrain(retrain_flag=None,df_old_=None,df_new=None):
    if retrain_flag !=None:
        X_old = df_old.drop(['created_account'],axis=1)
        X_new = df_new.drop(['created_account'],axis=1)
        drift_report = Dashboard(tabs=[DataDriftTab()])
        drift_report.calculate(X_old, X_new, column_mapping = None)
        drift_report.save("drift_report.html")
        logging.info('Drift report generated')
    else:
        logging.info('Not a retrain run')
