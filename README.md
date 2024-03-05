# Disaster Response Pipeline Project

## Project summary
This Disaster Reponse project contains the following elements:

	- ETL process (process_data.py)
			This script reads in data from two CSV files, cleans the data and stores it in a SQLite DB

	- ML training (train_classifier.py)
			This script takes the data from the SQLite DB and splits it into training and test data
			It then uses Natural Language processing and features from scikit-learn to train a classifier model
			The finished model is saved to a pickle file and is capable of reading and categorising text messages

	- Web application (run.py; go.html; master.html)
			The web app uses the classifier model created in the previous step to read messages and assign them
			to one or more of 36 designated categories to assist in guiding the message to the appropriate group

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

	- To run ETL pipeline that cleans data and stores in database
			`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
	- To run ML pipeline that trains classifier and saves
			`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
	`python run.py`

3. Go to http://0.0.0.0:3001/
