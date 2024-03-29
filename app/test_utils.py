from utils import *

df_campaign = config['data']['campaign']
df_mortgage = config['data']['mortgage']

dl = DataLoader(df_campaign,df_mortgage)

def test_fill_na_camp():
	dl.find_fill_nan()
	nan_camp = [i for i in df_campaign.columns if df_campaign[i].isnull().any()]
	assert len(nan_camp) == 0

def test_make_bins():
	dl.get_age_bins()
	assert len(df_campaign['age_bracket']) > 0

def test_create_age():
	dl.create_age()
	assert len(df_mortgage['age']) > 0

def test_get_code():
	dl.get_code()
	assert len(df_mortgage['country_code'])>0


def test_drop_columns_campaign():
	dl.drop_columns()
	assert json.loads(config.get("ETL","drop_camp")) not in df_campaign.columns.tolist()


def test_drop_columns_mortgage():
	dl.drop_columns()
	assert json.loads(config.get("ETL","drop_mort")) not in df_mortgage.columns.tolist()


def test_process_salary_band():
	dl.process_salary_band()
	assert 'crypto' in df_mortgage['salary_band'].tolist()

def test_process_salary_band_2():
	dl.process_salary_band()
	assert json.loads(config.get("test","salary_test_2"))  not in df_mortgage['salary_band'].tolist()


def test_drop_outliers():
	df_out = drop_outliers(df_campaign)
	assert len(df_out) >1