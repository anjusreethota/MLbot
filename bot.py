import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
file_path = 'C:/Users/anjus/Desktop/Chatbot/d1.csv' 
data = pd.read_csv(file_path)

# Prepare the data
questions = data['question'].tolist()
answers = data['answer'].tolist()

# Vectorize the questions using TF-IDF
vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

# Define a function to get the best matching answer for a user query
def get_answer(user_query):
    user_query_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_query_vector, question_vectors).flatten()
    best_match_index = np.argmax(similarities)
    return answers[best_match_index], similarities[best_match_index]

# Function to simulate a conversation
def chatbot_conversation():
    print("Welcome to the chatbot. Ask me anything!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        answer, similarity = get_answer(user_input)
        if similarity > 0.1:  # Threshold to ensure some relevance
            print(f"Chatbot: {answer}")
        else:
            print("Chatbot: I'm sorry, I don't understand the question.")

# Start the conversation
chatbot_conversation()
