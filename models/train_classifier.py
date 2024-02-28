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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
	engine = create_engine(f'sqlite:///{database_filepath}')
	df = pd.read_sql_table('messages_categorised', engine)
	#df = df.drop('child_alone', axis=1) # All values are the same in this column, if we don't remove it the training will fail
	X = df.message.values
	y = df.iloc[:, -36:]
	category_names = df.columns.tolist()[4:]
	return X, y, category_names


def tokenize(text):
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
	pipeline = Pipeline([
		('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
		('tfidf', TfidfTransformer()),
		('clf', MultiOutputClassifier(estimator=LogisticRegression()))
	])
	parameters = {
		'vect__ngram_range': ((1, 1), (1, 2)),
		'clf__estimator__max_iter': [100, 200, 300],
		'clf__estimator__intercept_scaling': [1, 2, 3]
	}
	return GridSearchCV(pipeline, param_grid=parameters)


def evaluate_model(model, X_test, y_test, category_names):
	y_pred = model.predict(X_test)
	for idx, column_name in enumerate(category_names):
		print('Results for ' + column_name)
		print(classification_report(y_test.iloc[:, idx], (pd.DataFrame({column_name: y_pred[:, idx]})).iloc[:, 0]))


def save_model(model, model_filepath):
	joblib.dump(model, f'{model_filepath}')


def main():
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