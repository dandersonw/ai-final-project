import json
import scrython

with open('guildsOfRavnica.json') as f:
    data = json.load(f)

cards = data['cards']
for card in cards:
    print(card['name'])
