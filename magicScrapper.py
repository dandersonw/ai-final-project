import json
import scrython

with open('guildsOfRavnica.json') as f:
    data = json.load(f)

cards = data['cards']
for card in cards:
    #print(card['name'])

firstCard = cards[0]

foundCard = srython.cards.Names(firstCard['name'])

print(foundCard)
