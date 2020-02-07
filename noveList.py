import streamlit as st
import os, sys, joblib, surprise
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
def load_data():
	# book map
	map_df = pd.read_csv(os.path.join(folder, 'data', 'book_id_map.csv'), dtype={'book_id_csv':int, 'book_id':int}, skipinitialspace=True)
	book_map = dict([(v,k) for k,v in map_df.values]) # create mapping between book_id_csv and book_id
	# reviews
	reviews = joblib.load(os.path.join(folder, 'data', 'ratings.joblib'))
	return book_map, reviews
# load book_id mappings, user_id mappings, and reviews
book_map, reviews = load_data()

# function to upload Goodreads library export csv file
def read_library_csv(file_name):
	user_data = pd.read_csv(file_name, usecols=['Book Id','Title','Author','My Rating','ISBN13','Exclusive Shelf'])
	# rename columns
	user_data = user_data.rename(columns={'Exclusive Shelf':'Shelf', 'My Rating':'Rating', 'Book Id':'Book ID'})
	return user_data

# function to convert user input to read/to-read
def parse_user_input(user_data, book_map, user_id=876145):
	# add user_id
	user_data['user_id'] = user_id
	# rename columns
	user_data = user_data.rename(columns={'Book ID':'book_id', 'Rating':'rating'})
	# convert book_id to book_id_csv
	user_data['book_id'] = user_data['book_id'].map(book_map)
	# remove books not in mapping
	user_data.drop(user_data[user_data['book_id'].isnull()].index, inplace=True)
	# split into read and to-read
	toread_list = user_data[user_data['Shelf'] == 'to-read']
	read_list = user_data[user_data['Shelf'] == 'read']
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
	bsl_options = {'method': 'als', # use alternating least squares to estimate values
				   'reg_i': 5, # regularization term for item bias
				   'reg_u': 10, # regularization item for user bias
				   'n_epochs': 5} # number of iterations
	# fit
	model = surprise.prediction_algorithms.BaselineOnly(bsl_options=bsl_options, verbose=False)
	model.fit(trainingSet);
	return model

# function to predict book ratings
def pred_ratings(model, reviews, toread_list, user_id=876145, k=10):
	pred = pd.DataFrame([model.predict(user_id, book_id).est for book_id in np.intersect1d(np.array(reviews['book_id']), np.array(toread_list['book_id']))])
	pred['book_id'] = np.intersect1d(np.array(reviews['book_id']), np.array(toread_list['book_id']))
	pred = pred.rename(columns={'0':'est_rating'})
	return pred

# function to predict top k and bottom k books
def ranked_books(toread_list, read_list, reviews, book_map, user_id=876145, k=10):
	with st.spinner('Ranking books...'):
		# add read books to reviews
		reviews = reviews.append(read_list, sort=False)
		# train model
		model = train_model(reviews)
		# predict book ratings
		pred = pred_ratings(model, reviews, toread_list, user_id, k)
		# top k
		top_k = pred[:k]
		# bottom k
		n_bottom = len(toread_list) - k
		if (n_bottom > 0) and (n_bottom >= k):
			bottom_k = pred[-k:]
		elif (n_bottom > 0) and (n_bottom < k):
			bottom_k = pred[-n_bottom:]
		else:
			bottom_k = []
		# convert book_id to title and author
		top_k = pd.merge(top_k,toread_list, how='inner', on='book_id')
		bottom_k = pd.merge(bottom_k,toread_list, how='inner', on='book_id')
	st.success('Done!')
	return top_k, bottom_k

# title and tagline
st.markdown('<span style="font-size:36pt;">**NoveList**</span><br><span style="font-size:24pt;">*Find your next page turner*</span>', unsafe_allow_html=True)

# explain what this app does
st.markdown('<span style="font-size:16pt;">This app ranks your "To-Read" books on [Goodreads](https://www.goodreads.com/)</span><br><br>', unsafe_allow_html=True)

# sidebar
# instructions on how to export Goodreads library
st.sidebar.markdown('<span style="font-size:18pt; font-style:bold;">To export your Goodreads library:</span>', unsafe_allow_html=True)
st.sidebar.markdown('<span style="font-size:16pt;">1. Go to [My Books](https://www.goodreads.com/review/list), then click on [Import and Export](https://www.goodreads.com/review/import) under **Tools** on the bottom left.<br>2. Click on the **Export Library** button at the top of the Import/Export screen below the Export heading.<br>3. Wait for the file to generate (this may take some time if you have a large library). If successful, you will see a **Your export from (date) - (time)** note below the button. Click on that text to download the csv file.</span>', unsafe_allow_html=True)

# input
st.markdown('<span style="font-size:20pt;">Would you like to upload a CSV of your exported Goodreads library?</span>', unsafe_allow_html=True)
upload_flag = st.radio('Upload your Goodreads data?',
					  ('Yes, upload my own Goodreads data','No, use pre-loaded data'), index=0)

k = 10 # number of books to rank
if upload_flag == 'Yes, upload my own Goodreads data': # upload file
	# upload csv file
	st.markdown('<br><span style="font-size:20pt; font-style:bold;">Upload your Goodreads library:</span>', unsafe_allow_html=True)
	csv_file = st.file_uploader(label='Upload your exported Goodreads Library CSV file',
								type=['csv'], encoding='utf-8')

	if csv_file is not None:
		# read csv file
		user_data = read_library_csv(csv_file)
		# display file
		st.markdown('<span style="font-size:16pt;">Your Goodreads Library:</span>', unsafe_allow_html=True)
		st.dataframe(user_data[['Title','Author','Rating','Shelf']].style.set_properties(**{'text-align': 'left'}))
		
		# check input
		# split into read and to-read
		toread_list, read_list = parse_user_input(user_data, book_map)
		# check how many books they've read and how many books they want to read
		if (len(read_list) < 10):
			st.error('Must have read at least 10 books to rank')
		elif (len(toread_list) < 2):
			st.error('Must want to read at least two books to rank')
		else:
			# click submit to run
			if st.button('Submit'):
				# if the number of to-read books is less than k, only sort those books
				if (len(toread_list) < k):
					k = len(toread_list)
				# predict top k books
				topk_books, bottomk_books, k = ranked_books(toread_list, read_list, reviews, book_map, k=k)
				# show predictions
				st.markdown('<span style="font-size:20pt; font-style:bold;">Your top ' + str(len(topk_books)) + ' ranked books are:</span>', unsafe_allow_html=True)
				st.table(topk_books[['Title','Author']].style.set_properties(**{'text-align': 'left'}))
				st.markdown('<span style="font-size:20pt; font-style:bold;">Your bottom ' + str(len(bottomk_books)) + ' ranked books are:</span>', unsafe_allow_html=True)
				st.table(bottomk_books[['Title','Author']].style.set_properties(**{'text-align': 'left'}))

elif upload_flag == 'No, use pre-loaded data': # use saved file
	# choose a profile
	st.markdown('<br><span style="font-size:20pt; font-style:bold;">Choose a user profile:</span>', unsafe_allow_html=True)
	user = st.selectbox('User profile',('User A', 'User B', 'User C'))
	# read csv file
	if user=='User A':
		user_data = read_library_csv(os.path.join(folder, 'goodreads_library_export', 'user0.csv'))
	elif user=='User B':
		user_data = read_library_csv(os.path.join(folder, 'goodreads_library_export', 'user1.csv'))
	elif user=='User C':
		user_data = read_library_csv(os.path.join(folder, 'goodreads_library_export', 'user2.csv'))
	# display file
	st.markdown('<span style="font-size:16pt;">Pre-Loaded Goodreads Library:</span>', unsafe_allow_html=True)
	st.dataframe(user_data[['Title','Author','Rating','Shelf']].style.set_properties(**{'text-align': 'left'}))

	# split into read and to-read
	toread_list, read_list = parse_user_input(user_data, book_map)

	# click submit to run
	if st.button('Submit'):
		# if the number of to-read books is less than k, only sort those books
		if (len(toread_list) < k):
			k = len(toread_list)
		# predict top k books
		topk_books, bottomk_books = ranked_books(toread_list, read_list, reviews, book_map, k=k)
		# show predictions
		st.markdown('<span style="font-size:20pt; font-style:bold;">Your top ' + str(len(topk_books)) + ' ranked books are:</span>', unsafe_allow_html=True)
		st.table(topk_books[['Title','Author']].style.set_properties(**{'text-align': 'left'}))
		st.markdown('<span style="font-size:20pt; font-style:bold;">Your bottom ' + str(len(bottomk_books)) + ' ranked books are:</span>', unsafe_allow_html=True)
		st.table(bottomk_books[['Title','Author']].style.set_properties(**{'text-align': 'left'}))

#
st.markdown('<br><br><span style="font-size:12pt">Created by Meghan Thommes<br>Health Data Science Fellow, Insight Data Science | Boston, MA</style>', unsafe_allow_html=True)
