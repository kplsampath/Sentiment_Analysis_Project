from flask import Flask, render_template,request, redirect
from helper import preprocessing, vectorizer, get_prediction

app = Flask(__name__)

data = dict()
#reviews = ['i love this product ' , 'Bad product' , 'i like it']
reviews = []
positive = 0
negative = 0


@app.route("/")
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative
    return render_template('index.html' , data=data)
    
    
 # Define a route for the root URL and allow only POST requests
@app.route("/", methods=['POST'])
def my_post():
    # Extract the 'text' field from the form data submitted with the POST request
    text = request.form['text']
    
    # Preprocess the input text (e.g., clean, tokenize, remove stop words)
    preprocessed_txt = preprocessing(text)
    
    # Convert the preprocessed text into a numerical vector  
    vectorized_txt = vectorizer(preprocessed_txt)
    
    # Use the vectorized text to get a prediction  
    prediction = get_prediction(vectorized_txt)
    
    # Initialize counters for negative and positive predictions (if not already initialized)
    global negative, positive
    if prediction == 'negative':  # Check if the prediction indicates negative sentiment
        global negative
        negative += 1  # Increment the negative counter
    else:  # Otherwise, assume the sentiment is positive
        global positive
        positive += 1  # Increment the positive counter
    
    # Add the original text to the beginning of the 'reviews' list for display or logging
    reviews.insert(0, text)   
    return redirect(request.url)



if __name__ == "__main__":
    app.run()
