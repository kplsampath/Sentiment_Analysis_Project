import numpy as np
import pandas as pd
import re
import string
import pickle

# Import the PorterStemmer class from the nltk.stem module
from nltk.stem import PorterStemmer
# Create an instance of the PorterStemmer class
ps = PorterStemmer()


# Load  model -------------------  
# File path is '../static/model/model', opened in binary read mode ('rb')
with open('static/model/model.pickle', 'rb') as f:
    # Unpickle (deserialize) the model and store it in the variable 'model'
    model = pickle.load(f)
    
# Load Stopwords -------------------    
# Remove Stopwords open
with open('static/model/corpora/stopwords/english', 'r') as file:
# Read the contents of the file and split it into a list of lines (each line contains a stopword)
    sw = file.read().splitlines()
  
# Load tokens -------------------      
# Load the vocabulary file into a DataFrame
# File path is '../static/model/vocabulary.txt', and it has no header
vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
# Extract the first column of the DataFrame and convert it into a list
tokens = vocab[0].tolist()
    
#Remove Punctuations
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    return text

# Define the preprocessing function
def preprocessing(text):
    
    # Remove punctuations
    data["tweet"] = data["tweet"].apply(remove_punctuations)

    # Remove numbers
    data["tweet"] = data["tweet"].str.replace(r'\d+', '', regex=True)

    # Remove stopwords
    data["tweet"] = data["tweet"].apply(
        lambda x: " ".join(
            word for word in x.split() if word not in sw
        )
    )

    # Apply stemming
    data["tweet"] = data["tweet"].apply(
        lambda x: " ".join(
            ps.stem(word) for word in x.split()
        )
    )

    return data["tweet"]

def preprocessing(text):
    # Convert text input into a pandas DataFrame
    data = pd.DataFrame([text], columns=['tweet'])  # Wrap text in a list
    
    # Convert uppercase to lowercase
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    return data["tweet"]


# Define a function to vectorize sentences based on a given vocabulary
def vectorizer(ds):
    # Initialize an empty list to store the vectorized sentences
    vectorized_lst = []
    
    # Iterate over each sentence in the dataset
    for sentence in ds:
        # Create a zero vector of length equal to the vocabulary size
        sentence_lst = np.zeros(len(tokens))
        
        # Iterate through each word in the vocabulary
        for i in range(len(tokens)):
            # Check if the vocabulary word exists in the current sentence
            if tokens[i] in sentence.split():
                # Set the corresponding index in the sentence vector to 1
                sentence_lst[i] = 1
        
        # Append the vectorized sentence to the list
        vectorized_lst.append(sentence_lst)
    
    # Convert the list of vectors into a NumPy array with type float32
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    
    # Return the vectorized dataset
    return vectorized_lst_new


def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'