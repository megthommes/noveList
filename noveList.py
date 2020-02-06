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

st.write(os.path.join(folder, 'data', 'ratings.joblib'))
reviews = joblib.load(os.path.join(folder, 'data', 'ratings.joblib'))
st.write(reviews.head(5))
