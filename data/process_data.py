import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	"""
	Load and merge message and categories data from two CSV files.

	Parameters:
	- messages_filepath: str
		Filepath of the CSV file containing the messages data.
	- categories_filepath: str
		Filepath of the CSV file containing the categories data.

	Returns:
	- DataFrame
		Merged DataFrame containing messages and categories data.
	"""
	messages = pd.read_csv(messages_filepath, dtype=str)
	categories = pd.read_csv(categories_filepath, dtype=str)
	return messages.merge(categories, on='id')

def clean_data(df):
	"""
	Clean and preprocess the merged DataFrame.

	Parameters:
	- df: DataFrame
		Input DataFrame to be cleaned.

	Returns:
	- DataFrame
		Cleaned DataFrame after preprocessing.
	"""
	categories = df['categories'].str.split(';', expand=True)
	row = categories.head(1)
	category_colnames = list(map(lambda x: x[:-2], row.values.tolist()[0]))
	categories.columns = category_colnames
	for column in categories:
		categories[column] = categories[column].apply(lambda x: x[-1])
		categories[column] = pd.to_numeric(categories[column]).astype('bool').astype(int)
	df = df.drop(['categories'], axis=1)
	df = df.join(categories)
	return df.drop_duplicates()

def save_data(df, database_filename):
	"""
	Save the cleaned DataFrame to a SQLite database.

	Parameters:
	- df: DataFrame
		Cleaned DataFrame to be saved.
	- database_filename: str
		Filename of the SQLite database.

	Returns:
	- None
	"""
	engine = create_engine(f'sqlite:///{database_filename}')
	df.to_sql('messages_categorised', engine, index=False)

def main():
	"""
	Main script to execute data processing steps.
	"""
	if len(sys.argv) == 4:
		messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

		print('Loading data...\n	MESSAGES: {}\n	CATEGORIES: {}'.format(
			messages_filepath, categories_filepath))
		df = load_data(messages_filepath, categories_filepath)

		print('Cleaning data...')
		df = clean_data(df)

		print('Saving data...\n	DATABASE: {}'.format(database_filepath))
		save_data(df, database_filepath)

		print('Cleaned data saved to database!')
	else:
		print('Please provide the filepaths of the messages and categories '
			  'datasets as the first and second argument respectively, as '
			  'well as the filepath of the database to save the cleaned data '
			  'to as the third argument. \n\nExample: python process_data.py '
			  'disaster_messages.csv disaster_categories.csv '
			  'DisasterResponse.db')

if __name__ == '__main__':
	main()