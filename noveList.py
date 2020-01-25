import streamlit as st
import os, sys, pickle
import importlib.util
import pandas as pd

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

# function to load model
@st.cache # cache
def load_model(modelName):
	model = pd.read_pickle(os.path.join(folder, 'models', modelName + '.sav'))
	return model

# title and tagline
st.markdown('<span style="font-family:verdana; font-size:32pt; font-style:bold;">NoveList</span><br><span style="font-family:verdana; font-size:20pt; font-style:italic;">Find your next page turner</span>', unsafe_allow_html=True)

# explain what this app does
st.markdown('<span style="font-family:verdana; font-size:14pt;">This app predicts how you will rate your books on your Want-to-Read list in [Goodreads](https://www.goodreads.com/)</span>', unsafe_allow_html=True)

# load model
modelName = 'model_collab_KNNWithMeans'
model = load_model(modelName)

# get user inputs
st.markdown('<span style="font-family:verdana; font-size:20pt; font-style:bold;">Inputs</span>', unsafe_allow_html=True)
# goodreads id
user_id = st.text_input('Goodreads Numeric User ID:', '8842281e1d1347389f2ab93d60773d4d')
st.markdown('<span style="font-family:verdana; font-size:14pt;">Your Goodreads ID can be found in your URL when on your profile page</span>', unsafe_allow_html=True)
st.markdown('![goodreads id](https://github.com/megthommes/noveList/blob/master/goodreads_id.png?raw=true)')
# book id
book_id = st.text_input('Goodreads Numeric Book ID:', '24375664')
st.markdown('<span style="font-family:verdana; font-size:14pt;">The Goodreads book ID can be found in the book URL</span>', unsafe_allow_html=True)

# make predictions
st.markdown('<span style="font-family:verdana; font-size:20pt; font-style:bold;">Estimated Rating(s)</span>', unsafe_allow_html=True)

# Make Predictions
if (user_id is not None) & (book_id is not None):
	# Make Prediction
	pred = model.predict(book_id, user_id, r_ui=None, verbose=False)
	# Convert to Dataframe
	pred_col_names = ['book_id', 'user_id', 'true_rating', 'est_rating', 'details']
	pred_df = pd.DataFrame(columns=pred_col_names)
	pred_sr = pd.Series(pred, index=pred_col_names)
	pred_df = pred_df.append(pred_sr, ignore_index=True)
	st.dataframe(pred_df['est_rating'])
