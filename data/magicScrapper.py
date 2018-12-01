import json
import scrython
import urllib.request as request
import pymysql.cursors
from random import randint

db = pymysql.connect(host="localhost",
                     user="root",
                     passwd="root",
                     db="aiproject",
                     port=3307)

errorCards = []

with open('sets.json') as f:
    sets = json.load(f)
try:
    with db.cursor() as cursor:
        sql = "select type_name from card_types"
        cursor.execute(sql)
        typesResponse = cursor.fetchall()
        types = [type[0] for type in typesResponse]

    with db.cursor() as cursor:
        sql = "select set_name from cards group by set_name"
        cursor.execute(sql)
        touchedSetsResponse = cursor.fetchall()
        touchedSets = [touchedSet[0] for touchedSet in touchedSetsResponse]

    for set in sets:
        if set not in touchedSets:
            print("starting: " + set)
            cards = sets[set]['cards']
            for card in cards:
                name = card['name']
                try:
                    foundCard = scrython.cards.Named(exact=name, set=set)
                    uri = foundCard.image_uris()['art_crop']
                    print(uri)
                    r = request.Request(uri)
                    f = request.urlopen(r)
                    with db.cursor() as cursor:
                        sql = "insert into cards (card_id, card_name, set_name, card_uri, image_data, is_for_testing) values (%s, %s, %s, %s, %s, %s)"
                        cursor.execute(sql, (card['uuid'], card['name'], set, uri, f.read(), randint(0,9) is 0))

                    for type in card['types']:
                        with db.cursor() as cursor:
                            sql = "insert into type_join (card_id, type_id) values (%s, %s)"
                            cursor.execute(sql, (card['uuid'], types.index(type)+1))
                    db.commit()
                    print("card added")
                except Exception as e:
                    print('individual card error occured')
                    print(e)
                    errorCards.append((name, set, e))

except Exception as e:
    print("Large error")
    print(e)
finally:
    db.close()
    print("errorCards")
    print(errorCards)





# cards = data['cards']
# # proof that all cards in the set are being loaded
# #for card in cards:
#     #print(card['name'])
#
# firstCard = cards[0]
#
# # proof that the scrython sdk works
# foundCard = scrython.cards.Named(exact=firstCard['name'])
#
# # an example of getting an art_crop image uri
# # will be used to fetch pictures for training the model
# uri = foundCard.image_uris()['art_crop']
# print(uri)
#
# r = request.Request(uri)
# f = request.urlopen(r)
# #print(f.read())
