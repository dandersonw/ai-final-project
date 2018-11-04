import json
import scrython
import urllib.request as request
import pymysql.cursors

db = pymysql.connect(host="localhost",
                     user="root",
                     passwd="root",
                     db="aiproject",
                     port=3307)

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
uri = foundCard.image_uris()['art_crop']
print(uri)

r = request.Request(uri)
f = request.urlopen(r)
#print(f.read())

try:
    with db.cursor() as cursor:
        sql = "select * from card_types"
        cursor.execute(sql)
        result = cursor.fetchall();
        print(result)

    def getTypeId(typeName):
        

    with db.cursor() as cursor:
        sql = "insert into cards (card_uri, card_name, image_data, card_type) values (%s, %s, %s)"
        cursor.execute(sql, (uri, firstCard['name'], f.read(), firstCard[]))
    db.commit()

    for type in firstCard['types']:
        with db.cursor() as cursor:
            sql = "insert into type_join (card_id, type_id) values (%s, %s)"
            cursor.execute(sql, (getCardId, results))
finally:
    db.close()
