import streamlit as st
import tensorflow as tf
from transformers import AutoProcessor, TFCLIPModel
import numpy as np
import pandas as pd
from PIL import Image
import requests

intro = """
Welcome to NLP-based image search engine using OPENAI's CLIP Model(Contrastive Language–Image Pre-training). The database contains 25k images from the Unsplash Dataset. You can search them:
-using a natural language description (e.g., animals in jungle)
-using Image URL.
The algorithm will return the nine most relevant images.

"""

getting_started = "To get started, simply enter your query type and text/url in the text box inside the sidebar and hit the search button. Our search engine will then scan unslash database of 25k images and return the most relevant results."


model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


photo_ids = pd.read_csv('/content/unsplash-dataset/lite/features/photo_ids.csv')
photo_ids = list(photo_ids['photo_id'])
photo = np.load('/content/unsplash-dataset/lite/features/features.npy')
print(f'Photos Loaded: {len(photo_ids)}')
photo_features = tf.convert_to_tensor(
    photo
)

def encode_text_query(query):
  encoded_text = processor(query, return_tensors = 'tf')
  encoded_text = model.get_text_features(**encoded_text)
  encoded_text /= tf.norm(encoded_text, axis = -1,  keepdims = True)
  return encoded_text

def encode_image_query(query):
  image = Image.open(requests.get(query, stream = True).raw)
  encoded_image = processor(images = image, return_tensors = 'tf')
  encoded_image = model.get_image_features(**encoded_image)
  return encoded_image

def find_best_match(features, photo_features, photo_ids, count):
  similarities = tf.squeeze(photo_features @ tf.cast(tf.transpose(features), tf.float16))
  best_photo_idx = tf.argsort(-similarities)
  return [photo_ids[i] for i in best_photo_idx[:count]]

def display_photo(photo_id):
  photo_image_url = f"https://unsplash.com/photos/{photo_id}/download?w=240"
  st.image(photo_image_url)

def search_unslash(query, query_type, photo_features, photo_ids, results_count):
  if query_type == 'Image URL':
    features = encode_image_query(query)
  else:
    features = encode_text_query(query)
  best_photo_ids = find_best_match(features, photo_features, photo_ids, results_count)
  col = st.columns(3)
  for i, photo_id in enumerate(best_photo_ids):
    with col[i%3]:
      display_photo(photo_id)



st.sidebar.header("Search Images")

query_type = st.sidebar.radio("Select query type:", ('Text', 'Image URL',))
query = st.sidebar.text_input("Enter text/url here:", 'two dogs playing in the snow')
search = st.sidebar.button('Search')

st.title("CLIP Image Search Engine")
if search:

  if query_type == 'Text':
    st.subheader(f'Search Results: {query}')
  else:
    st.subheader(f'Search Results:')
    st.sidebar.image(query)

  with st.spinner('Wait for it...'):
    search_unslash(query, query_type, photo_features, photo_ids, 9)
  st.success('Done!')

else:
  st.write(intro)
  st.subheader('Getting Started')
  st.write(getting_started)
  st.subheader("OPENAI's CLIP(Contrastive Language-Image Pre-Training)")
  st.image('https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png')
