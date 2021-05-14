"""
Description
This is a NLP proof of concept app to showcase models. 
"""

# Core Pkgs

import os
from pathlib import Path 
import json

import requests
import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import plotly.express as px

import streamlit as st 
import matplotlib.pyplot as plt

# make sure you specify .env
ML_INFERENCE_SERVER_ENDPOINT = os.environ['ML_INFERENCE_SERVER_ENDPOINT']
ML_ZEROSHOT_INFERENCE_SERVER_ENDPOINT = os.environ['ML_ZEROSHOT_INFERENCE_SERVER_ENDPOINT']

st.set_option('deprecation.showPyplotGlobalUse', False)

demos = {
	"DBpedia14":{
		"title":"DBpedia entity classifier",
		"description":'The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes from DBpedia 2014. Click [here](https://huggingface.co/datasets/dbpedia_14) for more info.',
		"samples":[
			"Costas Evangelatos. Costas Evangelatos is a Greek artist and poet born in Argostoli in 1957 but he is originated by his father side from Lixouri Kefalonia. He studied law at Athens University painting and aesthetic theory of modern art in Manhattan (New York City). From 1986 until 1993 he was the artistic director of the DADA Gallery in Athens and in 1990 founded the art group ART STUDIO EST.Along with painting has worked internationally in the fields of Performance Body Art Happening and Mail art.",
	"Leffingwell Inn. The Leffingwell Inn is a historic inn in the Norwichtown section of Norwich Connecticut.The building is architecturally important for having been built around some of the best seventeenth century remains left in Connecticut.It was listed on the National Register of Historic Places in 1970.It was photographed in the Historic American Buildings Survey program in 1961.",
	"Ramadasu (1964). Ramadasu is a 1964 Telugu devotional biographical film. The ensemble cast film was directed by veteran Chittor V. Nagaiah who has also enacted the role Kancharla Gopanna. The blockbuster film has garnered the National Film Award for Best Feature Film in Telugu and has garnered several state awards.",
	"Volno. Volno is a village in Petrich Municipality in Blagoevgrad Province Bulgaria.",
	"Rosenberg Library. Rosenberg Library a public library located at 2310 Sealy Street in Galveston Texas United States is the oldest continuously operating library in Texas. It serves as headquarters of the Galveston County Library System and its librarian also functions as the Galveston County Librarian."	
		],
		"labels":["Company", "Educational Institution", "Artist", "Athlete", "Office Holder", "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal", "Plant", "Album", "Film", "Written Work"],
		"available_models":["dbpedia_baseline_bert", "dbpedia_distilbert", "Zero-shot classification"]
	},
	"Toxicity":{
		"title":"",
		"description":"Trained models to predict toxic comments. Task and datasets were provided by The Conversation AI team, a research initiative founded by Jigsaw and Google (both a part of Alphabet). Click [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) for more info.",
		"samples":[
			"You are studid and ugly as f*",
            "I hate you and you suck",
            "This is bullshit!",
            "Another lovely comment",
            "Go back to your country"
		],
		"labels":['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
		"available_models":["toxic_comments_baseline_bert", "Zero-shot classification"]
	}
}

# Title
st.sidebar.subheader("Text classification labs for FSDL")
st.sidebar.markdown("This is app showcases experiments on text classification.")

selected_demo = st.sidebar.selectbox("Select demo", list(demos.keys()), 0)

def radar_chart_plot(df):  
    
    fig = px.line_polar(df, r='scores', theta='labels', line_close=True)
    st.write(fig)

def bar_chart_plot(prediction_result):
	plt.plot()

	height = prediction_result["scores"]
	bars = prediction_result["labels"]
	y_pos = np.arange(len(bars))
	
	# Create bars
	plt.bar(y_pos, height)
	
	# Create names on the x-axis
	plt.xticks(y_pos, bars, rotation='vertical')

	# Show graphic
	st.pyplot()

from contextlib import contextmanager

def remote_zeroshot_inference_request(input_text, candidate_labels, multi_class=False):
	
	loading = st.info(f"Running prediction request ...")

	payload = {
		"inputs": [input_text],
		"parameters": {
			"candidate_labels": candidate_labels,
			"multi_class": multi_class
		}
	}
	
	print(payload)
	response = requests.request("POST", ML_ZEROSHOT_INFERENCE_SERVER_ENDPOINT, json=payload, 
					headers={"Content-Type": "application/json"})

	loading.empty()

	return json.loads(response.text)

def remote_inference_request(input_text, model_name):
	loading = st.info(f"Running prediction request ...")

	payload = {
		"text": input_text,
		"model_name":model_name
		}

	headers = {"Content-Type": "application/json"}

	response = requests.request("POST", ML_INFERENCE_SERVER_ENDPOINT, json=payload, headers=headers)

	loading.empty()
	return json.loads(response.text)

def main():
	
	if(selected_demo=="DBpedia14"):
		st.header(demos[selected_demo]["title"])

		st.sidebar.write(demos[selected_demo]["description"])	

		selected_input_sample = st.selectbox("Predefined examples", demos[selected_demo]["samples"])
		input_text = st.text_area("Type your message here.", selected_input_sample, height=250)
		selected_model = st.selectbox("Select model", demos[selected_demo]["available_models"], 0)
		
		if st.button("Classify text"):
			st.subheader("Results")
			if(selected_model == "Zero-shot classification" ):

				prediction_result = remote_zeroshot_inference_request(input_text, demos[selected_demo]["labels"], multi_class=False)
				bar_chart_plot(prediction_result)

				with st.beta_expander("See detailed scores", expanded=True):
					st.write(prediction_result)

			else:
				prediction_result = remote_inference_request(input_text, selected_model)
		
				st.markdown(f"**{prediction_result['prediction'][0]['class']}** is the most likely category.")
				with st.beta_expander("See detailed scores", expanded=True):
					st.write(prediction_result)					
				
		
	elif selected_demo == "Toxicity":

		st.header(demos[selected_demo]["title"])
		st.sidebar.write(demos[selected_demo]["description"])	

		selected_input_sample = st.selectbox("Select example", demos[selected_demo]["samples"]	, index=0)
		input_text = st.text_area("Or edit here your comment",value=selected_input_sample)

		# view_mode = st.radio("Select view mode", ["Chart", "Table"], index=1)
		selected_model = st.selectbox("Select model", demos[selected_demo]["available_models"], 0)
		
		if st.button("Classify comment"):
			st.subheader("Results")

			if(selected_model == "Zero-shot classification" ):
				prediction_result = remote_zeroshot_inference_request(input_text, demos[selected_demo]["labels"], multi_class=True)

				score_threshold = 0.6
				
				predicted_scores = np.array(prediction_result["scores"])
				predicted_labels = prediction_result["labels"]

				if(len(predicted_scores[predicted_scores >= score_threshold]) == 0):
					
					st.markdown(f"Model didn't detect any toxic content :)")
				else:
					st.markdown(f"Model detected potential toxic content.")

					df = pd.DataFrame.from_dict(dict(scores = predicted_scores, 
													labels = predicted_labels))\
													.sort_values(by="scores", axis=0, ascending=False)
					radar_chart_plot(df)	

					with st.beta_expander("See detailed scores", expanded=True):
						st.write(prediction_result)					
			else:
				
				prediction_result = remote_inference_request(input_text, selected_model)

				predicted_scores = prediction_result["prediction"][0]["confidence"]
				predicted_labels = prediction_result["prediction"][0]["class"] 

				if(len(predicted_labels) == 0):
					st.markdown(f"Model didn't detect any toxic content :)")
				else:
					st.markdown(f"Model detected potential toxic content.")

					result_table = {}
					for label in demos[selected_demo]["labels"]:
						result_table[label] = 0

					for idx, predicted_label in enumerate(predicted_labels):
						result_table[predicted_label] = predicted_scores[idx]

					
					df = pd.DataFrame.from_dict(dict(scores = result_table.values(), 
													labels = result_table.keys()))\
													.sort_values(by="scores", axis=0, ascending=False)
					
					radar_chart_plot(df)

					with st.beta_expander("See detailed scores", expanded=True):
						st.write(result_table)

if __name__ == '__main__':
	main()
	
