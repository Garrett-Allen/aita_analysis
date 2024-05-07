#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bertopic import BERTopic
import pandas as pd
import nltk


# In[2]:


reddit_data = pd.read_parquet("labeled_post.parquet")

model_data = reddit_data[["link_flair_text","selftext","title"]]

titles = model_data["title"]
body_texts = model_data["selftext"]
label = reddit_data["link_flair_text"]


# In[3]:


from nltk.tokenize import sent_tokenize, word_tokenize

sentences = [sent_tokenize(body_text) for body_text in body_texts]


# In[4]:


# embedding
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(body_texts, show_progress_bar=True)


# In[5]:


# dimensionality reduction
from umap import UMAP

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)


# In[31]:


# clustering

from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', prediction_data=True)


# In[7]:


# tokenize topics

from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 3))


# In[9]:


# topic representation
from bertopic.vectorizers import ClassTfidfTransformer
ctfidf_model = ClassTfidfTransformer()


# In[19]:


# representation model (for tine tuning representation)
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
representation_model = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=.5)]


# In[32]:


# all steps

topic_model = BERTopic(
  embedding_model=embedding_model,          # Step 1 - Extract embeddings
  umap_model=umap_model,                    # Step 2 - Reduce dimensionality
  hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
  representation_model=representation_model, # Step 6 - (Optional) Fine-tune topic represenations
  verbose = True, 
  calculate_probabilities = True
)
   


# In[33]:


# fitting model
topics, prob = topic_model.fit_transform(body_texts,embeddings)


# In[1]:


topic_model.get_topic_info()[0:3]


# In[17]:


topic_model.visualize_hierarchy()


# In[23]:


topic_model.visualize_topics()


# In[24]:


topic_model.visualize_heatmap()


# In[30]:


reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

topic_model.visualize_documents(docs = body_texts, embeddings=reduced_embeddings)

