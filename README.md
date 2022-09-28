------------------------ COMPONENTS ---------------------------------------------------------------------

1. assignment_notebook.ipynb - initial notebook for data cleaning modeling etc
2. utils.py - classes and functions for end to end modeling pipeline
3. main.py - run the pipeline and produce the model with its evaluation metrics
4. app.y - Flask app that exposes an endpoint to productionise the model
5. test_utils.py - pytests for utils.py
6. Dockerfile - build the Flask app container that hosts the model
7. requirements.txt 

---------------------- USAGE -----------------------------------------------------------------------------

1. assignment_notebook contains the initial workings from data to modelling, in notebook format

2. install requirements.txt

4. optionally run the pytests to debug utils.py

5. run the main.py which essentially calls the functionalities within utils.py. 

6. running main.py will create the model. Evaluation metrics (loss function/training loss curves) will appear at the end

7. if you want to retrain the model with different data switch the retrain_flag of the data_drift_retrain() function found within the main.py to produce the data drift report

8. the model will be saved in the data folder

9. once the model is produced build a Docker image using the Dockerfile

10. a Flask application (app.py) is being included so once the Docker image is built the model is exposed on a Flask endpoint thus in can run on produdction in any system