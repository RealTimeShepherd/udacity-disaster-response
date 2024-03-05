import re
import sys
import joblib
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
	"""
	Load data from a SQLite database and return features, targets, and category names.

	Parameters:
	- database_filepath: str
			Filepath of the SQLite database.

	Returns:
	- X: numpy.array
			Feature variable (message values).
	- y: pandas.DataFrame
			Target variables (category values).
	- category_names: list
			List of category names.
	"""
	engine = create_engine(f'sqlite:///{database_filepath}')
	df = pd.read_sql_table('messages_categorised', engine)
	X = df.message.values
	y = df.iloc[:, -36:]
	category_names = df.columns.tolist()[4:]
	return X, y, category_names


def tokenize(text):
	"""
	Tokenize and preprocess text data.

	Parameters:
	- text: str
			Input text to tokenize.

	Returns:
	- clean_tokens: list
			List of clean tokens after tokenization and preprocessing.
	"""
	detected_urls = re.findall(url_regex, text)
	for url in detected_urls:
		text = text.replace(url, "urlplaceholder")

	tokens = word_tokenize(text)
	lemmatizer = WordNetLemmatizer()

	clean_tokens = []
	for tok in tokens:
		clean_tok = lemmatizer.lemmatize(tok).lower().strip()
		clean_tokens.append(clean_tok)

	return clean_tokens


def build_model():
	"""
	Build and configure the machine learning pipeline.

	Returns:
	- model: GridSearchCV
			GridSearchCV model containing the pipeline and parameter grid.
	"""
	pipeline = Pipeline([
		('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
		('tfidf', TfidfTransformer()),
		('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=500)))
	])
	parameters = {
		'clf__estimator__max_features': ['sqrt', 'log2'],
	}
	return GridSearchCV(pipeline, param_grid=parameters)


def evaluate_model(model, X_test, y_test, category_names):
	"""
	Evaluate the trained model and print classification report for each category.

	Parameters:
	- model: GridSearchCV
			Trained model.
	- X_test: numpy.array
			Test feature variables.
	- y_test: pandas.DataFrame
			Test target variables.
	- category_names: list
			List of category names.

	Returns:
	- None
	"""
	y_pred = model.predict(X_test)
	for idx, column_name in enumerate(category_names):
		print('Results for ' + column_name)
		print(classification_report(y_test.iloc[:, idx], (pd.DataFrame({column_name: y_pred[:, idx]})).iloc[:, 0]))


def save_model(model, model_filepath):
	"""
	Save the trained model as a serialized file.

	Parameters:
	- model: GridSearchCV
			Trained model to be saved.
	- model_filepath: str
			Filepath to save the model.

	Returns:
	- None
	"""
	joblib.dump(model, f'{model_filepath}')


def main():
	"""
	Main script to execute model training steps.
	"""
	if len(sys.argv) == 3:
		database_filepath, model_filepath = sys.argv[1:]
		print('Loading data...\n	DATABASE: {}'.format(database_filepath))
		X, Y, category_names = load_data(database_filepath)
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
		
		print('Building model...')
		model = build_model()
		
		print('Training model...')
		model.fit(X_train, Y_train)
		
		print('Evaluating model...')
		evaluate_model(model, X_test, Y_test, category_names)

		print('Saving model...\n	MODEL: {}'.format(model_filepath))
		save_model(model, model_filepath)

		print('Trained model saved!')

	else:
		print('Please provide the filepath of the disaster messages database '\
			  'as the first argument and the filepath of the pickle file to '\
			  'save the model to as the second argument. \n\nExample: python '\
			  'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
	main()