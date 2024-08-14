import random
import json
import torch
from model import model
from nltk_utils import tokenizer, bag_of_words

device = "cuda" if torch.cuda.is_available() else "cpu"

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE_SAVE_PATH = "data.pth"
ruby_data = torch.load(FILE_SAVE_PATH)

input_features = ruby_data["input_features"]
output_features = ruby_data["output_features"]
hidden_units = ruby_data["hidden_units"]
all_words = ruby_data["all_words"]
tags = ruby_data["tags"]
model_state = ruby_data["model_state"]

model = model(input_features, output_features, hidden_units)
model.load_state_dict(model_state)
model.eval()

bot_name = "Ruby"
print(f"Hi!😃 I'm {bot_name}\nA simple chatbot! Type 'quit' to exit")
while True:
    user = input("You: ")
    if user == "quit":
        break

    user_sentence = tokenizer(user)
    X = bag_of_words(user_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]
    
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I'm sorry i dont get you...Maybe try again later?? 😅")
