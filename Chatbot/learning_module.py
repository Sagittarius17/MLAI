import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

CHATLOG_FILE = "Chatbot/chatlog.txt"
MODEL_FILE = "Chatbot/chatbot_model.pkl"

def update_chatlog(user_input, response):
    with open(CHATLOG_FILE, "a") as f:
        f.write(f"User: {user_input}\nChatBot: {response}\n\n")

def train_chatbot_model():
    if not os.path.exists(CHATLOG_FILE):
        print("No chatlog file found.")
        return

    try:
        # Load chatlog data
        with open(CHATLOG_FILE, "r") as f:
            lines = f.readlines()

        # Prepare data for training
        data = []
        labels = []
        for i in range(0, len(lines), 4):
            user_input = lines[i].replace("User:", "").strip()
            response = lines[i + 2].replace("ChatBot:", "").strip()
            data.append(user_input)
            labels.append(response)

        if not data or not labels:
            print("No data found for training.")
            return

        # Vectorize the data
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data)

        if X.shape[0] == 0 or X.shape[1] == 0:
            print("Empty feature matrix. Check your data.")
            return

        # Train a simple Naive Bayes classifier
        clf = MultinomialNB()
        clf.fit(X, labels)

        # Save the model
        with open(MODEL_FILE, "wb") as model_file:
            pickle.dump((vectorizer, clf), model_file)

        print("Model trained and saved successfully.")

    except Exception as e:
        print("Error during training:", str(e))


def get_chatbot_response(user_input):
    if not os.path.exists(MODEL_FILE):
        print("No model file found. Please train the model first.")
        return None

    # Load the model
    with open(MODEL_FILE, "rb") as model_file:
        vectorizer, clf = pickle.load(model_file)

    # Vectorize user input
    user_input_vectorized = vectorizer.transform([user_input])

    # Get the predicted response
    response = clf.predict(user_input_vectorized)[0]

    return response
