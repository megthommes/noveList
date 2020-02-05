import streamlit as st
import os, sys, joblib, time, surprise
import importlib.util
import pandas as pd
import numpy as np

if len(sys.argv) > 1: # parse command-line argument if supplied (folder path)
    folder = os.path.abspath(sys.argv[1])
else: # otherwise use the current working directory
    folder = os.path.abspath(os.getcwd())

# get filenames for all python files in this path, excluding this script
thisFile = os.path.abspath(__file__)
fileNames = []
for baseName in os.listdir(folder):
	fileName = os.path.join(folder, baseName)
	if (fileName.endswith(".py")) and (fileName != thisFile):
		fileNames.append(fileName)

# filename formatter to display a nicer url (instead of the whole github path)
def format_url(s):
	els = s.split("/")[-1].split(".")[0].split("_")
	return " ".join(el for el in els).capitalize()

# function to load book_id mappings and reviews
# I would cach this, but it doesn't seem to work
def load_data():
	# book map
	map_df = pd.read_csv(os.path.join(folder, 'data', 'book_id_map.csv'), dtype={'book_id_csv':int, 'book_id':int}, skipinitialspace=True)
	book_map = dict([(v,k) for k,v in map_df.values]) # create mapping between book_id_csv and book_id
	# reviews
	reviews = joblib.load(os.path.join(folder, 'data', 'reviews.joblib'))
	return book_map, reviews
# load book_id mappings, user_id mappings, and reviews
book_map, reviews = load_data()

# function to upload Goodreads library export csv file
def read_library_csv(file_name):
	user_data = pd.read_csv(file_name, usecols=['Book Id','Title','Author','My Rating'])
	return user_data

# function to convert user input to read/to-read
def parse_user_input(user_data, book_map):
	# rename columns
	user_data = user_data.rename(columns={'Book Id':'book_id', 'My Rating':'rating'})
	# convert book_id to book_id_csv
	user_data['book_id'] = user_data['book_id'].map(book_map)
	# remove books not in mapping
	user_data.drop(user_data[user_data['book_id'].isnull()].index, inplace=True)
	# split into read and to-read
	toread_list = user_data[user_data['rating'] == 0]
	read_list = user_data[user_data['rating'] != 0]
	return toread_list, read_list

# function to train model
def train_model(reviews_df):
	# define rating scale
	reader = surprise.Reader(rating_scale=(1, 5))
	# column names to use in building the collaborative filtering models
	col_names = ['book_id', 'user_id', 'rating']
	# convert to dataset
	reviews = surprise.Dataset.load_from_df(reviews_df[col_names], reader)
	# build training set
	trainingSet = reviews.build_full_trainset()
	# baseline configuration
	bsl_options = {'method': 'als',
				   'reg_i': 5, # regularization term for item bias
				   'reg_u': 10, # regularization item for user bias
				   'n_epochs': 5} # number of iterations
	# fit
	model = surprise.prediction_algorithms.BaselineOnly(bsl_options=bsl_options, verbose=False)
	model.fit(trainingSet);
	return model

# function to predict book ratings
def pred_ratings(model, reviews, toread_list, user_id=876145, k=10):
	pred = dict()
	for book_id in np.intersect1d(np.array(reviews['book_id']), np.array(toread_list['book_id'])):
		pred[book_id] = model.predict(user_id, book_id).est
	# convert to dataframe
	pred_df = pd.DataFrame(pred.items(), columns=['book_id','est_rating'])
	# only return top k
	pred_df = pred_df[:k]
	return pred_df

# function to predict top 10 books
def top_ten(user_data, reviews, book_map, user_id=876145, k=10):
	# add user_id
	user_data['user_id'] = user_id
	# split into read and to-read
	toread_list, read_list = parse_user_input(user_data, book_map)
	# add read books to reviews
	reviews = reviews.append(read_list, sort=False)
	# train model
	model = train_model(reviews)
	# predict book ratings
	pred = pred_ratings(model, reviews, toread_list, user_id, k)
	# convert book_id to title and author
	top_ten = pd.merge(pred,toread_list, how='inner', on='book_id')
	return top_ten

# title and tagline
st.markdown('<span style="font-family:verdana; font-size:36pt; font-style:bold;">NoveList</span><br><span style="font-family:verdana; font-size:24pt; font-style:italic;">Find your next page turner</span>', unsafe_allow_html=True)

# explain what this app does
st.markdown('<span style="font-family:verdana; font-size:14pt;">This app predicts how you will rate your books in your "To-Read" list on [Goodreads](https://www.goodreads.com/)</span>', unsafe_allow_html=True)

# inputs
st.markdown('<span style="font-family:verdana; font-size:20pt;">Would you like to upload a CSV of your exported Goodreads library?</span>', unsafe_allow_html=True)
upload_flag = st.radio('Upload a Goodreads Library Export CSV file?',
					  ('Yes','No'), index=0)

if upload_flag == 'Yes': # upload file
	st.markdown('<span style="font-family:verdana; font-size:16pt; font-style:bold;">To export your Goodreads library:</span>', unsafe_allow_html=True)
	# instructions on how to export Goodreads library
	st.markdown('<span style="font-family:verdana; font-size:14pt;">1. Go to [My Books](https://www.goodreads.com/review/list), then click on [Import and Export](https://www.goodreads.com/review/import) under **Tools** on the bottom left.<br>2. Click on the **Export Library** button at the top of the Import/Export screen below the Export heading.<br>3. Wait for the file to generate (this may take some time if you have a large library). If successful, you will see a **Your export from (date) - (time)** note below the button. Click on that text to download the csv file.</span>', unsafe_allow_html=True)
	# upload csv file
	csv_file = st.file_uploader(label='Upload a Goodreads Library Export CSV file',
								type=['csv'], encoding='utf-8')

	if csv_file is not None:
		# read csv file
		user_data = read_library_csv(csv_file)
		# display file
		st.markdown('<span style="font-family:verdana; font-size:12pt;">Goodreads Library Export CSV File:</span>', unsafe_allow_html=True)
		st.dataframe(user_data)
		# predict top 10 books
		top10_books = top_ten(user_data, reviews, book_map)
		# show predictions
		st.table(top10_books[['Title','Author']])

elif upload_flag == 'No': # use saved file
	st.markdown('<span style="font-family:verdana; font-size:16pt; font-style:bold;">Use a pre-generated Goodreads Library Export CSV file</span>', unsafe_allow_html=True)
	# read csv file
	user_data = read_library_csv(os.path.join(folder, 'data', 'goodreads_library_export.csv'))
	# display file
	st.markdown('<span style="font-family:verdana; font-size:12pt;">Goodreads Library Export CSV File:</span>', unsafe_allow_html=True)
	st.dataframe(user_data)
	# predict top 10 books
	top10_books = top_ten(user_data, reviews, book_map)
	# show predictions
	st.table(top10_books[['Title','Author']])
