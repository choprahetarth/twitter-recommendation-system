import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import random

# set a title #
st.title('Tweet Recommendation Engine')

# create a function where the model is reading the data and doing the embedding storage #
@st.experimental_singleton
def load_data_and_store_embeddings():
    twitter_dataset = pd.read_csv('tweet.csv')
    twitter_dataset = twitter_dataset['clean_text']
    twitter_dataset = twitter_dataset.head(10000)
    twitter_dataset = twitter_dataset.to_list()
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = embedder.encode(twitter_dataset, convert_to_tensor=True)
    return twitter_dataset, corpus_embeddings, embedder

def show_text(dataset):
    return random.choice(dataset)

def semantic_s(query, corpus_embeddings, dataset,embedder):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=6)
    hits = hits[0]      #Get the hits for the first query
    st.write("Recommended Tweets are - ")
    for i,hit in enumerate(hits): 
        if i>0:
            st.write(dataset[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

st.text('Please select a random tweet and get it
s recommended tweets')



twitter_dataset, corpus_embeddings, embedder= load_data_and_store_embeddings()

if st.button('Get Random Tweet'):
    last_retrieved_tweet = show_text(twitter_dataset)
    st.write(last_retrieved_tweet)
    semantic_s(last_retrieved_tweet, corpus_embeddings, twitter_dataset, embedder)

# if st.button('Get Suggestions'):

