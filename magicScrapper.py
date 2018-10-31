import json
import scrython

with open('guildsOfRavnica.json') as f:
    data = json.load(f)

cards = data['cards']
# proof that all cards in the set are being loaded
#for card in cards:
    #print(card['name'])

firstCard = cards[0]

# proof that the scrython sdk works
foundCard = scrython.cards.Named(exact=firstCard['name'])

# an example of getting an art_crop image uri
# will be used to fetch pictures for training the model
print(foundCard.image_uris()['art_crop'])
