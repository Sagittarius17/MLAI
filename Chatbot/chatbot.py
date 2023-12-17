from nltk.chat.util import Chat, reflections
from learning_module import update_chatlog, train_chatbot_model, get_chatbot_response

def load_chat_pairs_from_file(file_path):
    pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into two parts at the first comma
            parts = line.split(',', 1)
            
            # If there are at least two parts, add them to the pairs list
            if len(parts) == 2:
                pattern, response = map(str.strip, parts)
                pairs.append([pattern, [response]])

    return pairs


# Load chat pairs from a file
pairs = load_chat_pairs_from_file('Chatbot/chat_pairs.txt')

# Create a Chat instance with the loaded pairs
chatbot = Chat(pairs, reflections)

# Main chat loop
while True:
    user_input = input("You: ")

    # Use the chatbot model to get a response
    response = get_chatbot_response(user_input)

    # If the model doesn't have a response, use the rule-based chatbot
    if not response:
        response = chatbot.respond(user_input)

    # Update the chatlog with the user input and response
    update_chatlog(user_input, response)

    print("Sagittarius:", response)

    # Train the chatbot model periodically (you can adjust the frequency)
    if user_input.lower() == 'train':
        train_chatbot_model()
