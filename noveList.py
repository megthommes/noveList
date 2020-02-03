import streamlit as st
import os, sys, joblib, time
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

# function to convert user input to read/to-read
def parse_user_input(user_data):
	# get book_id and rating
	tmp_df = user_data[['Book Id','My Rating']].copy()
	# rename columns
	tmp_df = tmp_df.rename(columns={'Book Id':'book_id', 'My Rating':'rating'})
	# convert book_id to book_id_csv
	map_df = pd.read_csv(os.path.join(folder, 'data', 'book_id_map.csv'),
						 dtype={'book_id_csv':int, 'book_id':int},
						 skipinitialspace=True)
	mapping = dict([(v,k) for k,v in map_df.values]) # create mapping between book_id_csv and book_id
	tmp_df['book_id'] = tmp_df['book_id'].map(mapping)
	# remove books not in mapping
	tmp_df.drop(tmp_df[tmp_df['book_id'].isnull()].index, inplace=True)
	# split into read and to-read
	toread_df = tmp_df[tmp_df['rating'] == 0]
	read_df = tmp_df[tmp_df['rating'] != 0]
	return toread_df, read_df

# function to load model
@st.cache
def load_model():
	# SVD Parameters
	item_biases = joblib.load(os.path.join(folder, 'svd-params', 'item_biases.joblib'))
	user_biases = joblib.load(os.path.join(folder, 'svd-params', 'user_biases.joblib'))
	item_factors = joblib.load(os.path.join(folder, 'svd-params', 'item_factors.joblib'))
	user_factors = joblib.load(os.path.join(folder, 'svd-params', 'user_factors.joblib'))
	return global_mean, item_biases, user_biases, item_factors, user_factors

# function to create model
@st.cache
def make_model():
	global_mean, item_biases, user_biases, item_factors, user_factors = load_model()
	new_user_biases = np.append(user_biases,0)
	new_user_factors = np.vstack((user_factors,np.zeros(200)))
	qTp = item_factors.dot(np.transpose(new_user_factors)) + global_mean
	return item_bases, new_user_biases, qTp

# function to predict book ratings
#def pred_ratings(toread_df):
	
	#book_pred

# function for progress bar
def progress_bar():
	progress_bar = st.progress(0)
	status_text = st.text('Working...')
	for i in range(11):
		progress_bar.progress(i*10)
		time.sleep(0.1)
	status_text.text('Done!')

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
		user_data = pd.read_csv(csv_file)
		# display uploaded file
		st.dataframe(user_data)
		# parse uploaded file
		toread_df, read_df = parse_user_input(user_data)

elif upload_flag == 'No': # use saved file
	st.markdown('<span style="font-family:verdana; font-size:16pt; font-style:bold;">Use a pre-generated Goodreads Library Export CSV file</span>', unsafe_allow_html=True)
	# read csv file
	user_data = pd.read_csv(os.path.join(folder, 'data', 'goodreads_library_export.csv'))
	# display file
	st.markdown('<span style="font-family:verdana; font-size:12pt;">Goodreads Library Export CSV File:</span>', unsafe_allow_html=True)
	st.dataframe(user_data)
	# parse uploaded file
	toread_df, read_df = parse_user_input(user_data)
	

if st.button('Submit'): # run model
	pred_ratings = user_data[['Title','Author','Average Rating']].copy()
	pred_ratings = pred_ratings.rename(columns={'Average Rating':'Predicted Rating'})
	st.dataframe(pred_ratings)

	progress_bar()