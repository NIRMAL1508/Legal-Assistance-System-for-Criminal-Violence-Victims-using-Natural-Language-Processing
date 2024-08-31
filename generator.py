import torch
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import pandas as pd
nltk.download('stopwords')
import numpy as np
 
# Load pre-trained BERT tokenizer and model
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = DistilBertModel.from_pretrained('distilbert-base-uncased')


# Load English stopwords
stop_words = set(stopwords.words('english'))

# Function to tokenize, remove stop words, and encode sentences
class suggestor:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased',num_hidden_layers=24 , num_attention_heads=24 )
    data = pd.read_csv(".\embeddings\sectionsdataset_withoutillustrations - Sheet1.xls.csv")
    embeddings = pd.read_csv(".\embeddings\section_embeddings_noillustrations_24l24h.csv")
    sections = data["Description"].values
    sectionnames = data["Section"].values
    sectionnumber = data["Number"].values
    
    def encode_sentences(sentences):
        cleaned_sentences = []
        for sentence in sentences:
            # Remove stop words
            cleaned_sentence = ' '.join([word for word in sentence.split() if word.lower() not in stop_words])
            cleaned_sentences.append(cleaned_sentence)

        # Tokenize and encode the cleaned sentences
        encoded_sentences = suggestor.tokenizer(cleaned_sentences, return_tensors='pt', padding=True, truncation=True)

        return encoded_sentences
    
    def calculate_similarity(self,input_text,sentence_vectors=embeddings):
        # Encode the input text
        encoded_input = suggestor.encode_sentences([input_text])

        # Get BERT embeddings for the tokenized input text
        with torch.no_grad():
            input_embedding = suggestor.model(**encoded_input).last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
            input_embedding = torch.nn.functional.normalize(input_embedding, p=2, dim=-1)  # L2 normalize

        similarity_scores = cosine_similarity(input_embedding, sentence_vectors)

        # Rank sentences based on similarity scores
        ranked_sentences = sorted(zip(suggestor.sectionnames, similarity_scores[0]), key=lambda x: x[1], reverse=True)
        print("DONE")
        return ranked_sentences