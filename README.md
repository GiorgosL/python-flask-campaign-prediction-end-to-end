<h1 align="center">Data Science Assignment</h1>

## 1. Components 

1. **assignment_notebook.ipynb** - initial notebook for data cleaning, modeling and business insights
2. **utils.py** - classes and functions for the end to end modeling pipeline
3. **main.py** - call and run the functionality from utils.py to produce the model and its evaluation components
4. **app.py** - Flask app that exposes an endpoint to productionise the model
5. **test_utils.py** - pytests for utils.py
6. **Dockerfile** - build the Flask app image that hosts the model and distribute it to any system to run in production
7. **requirements.txt** 
8. **campaign_report.html / mortgage_report.html** - pandas profiling report for the two dataframes

## 2. How to


1. Install requirements.txt
2. Run main.py to create the model.json in data folder. Evaluation metrics (loss function/learning curves) will appear on screen once done.
3. If retraining the mode with new data enable the retrain_flag of the data_drift_retrain() function in main.py to produce the data drift report against the old data.
4. A Flask application (app.py) is included to expose the model in a dedicated endpoint.
5. Once the model is created build the app.py Docker image using the Dockerfile.
6. Once the Docker image is built, it can be distributed and productionised on any system.

## 3. Optional steps

1. Check the pandas profilling report for the two dataframes found in the data folder.
2. Run the pytests to debug utils.py.
3. Specify custom hyperparameters and evaluation metrics for the Grid as well as the number of features for feature selection, within main.py.
