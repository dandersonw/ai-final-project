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

def getStatus(status):
    if status is 0:
        return "TESTING"
    if status is 1:
        return "VALIDATION"
    return "TRAINING"

errorCards = []
counter = 0

# get the list of unique names we need to get info for
with db.cursor() as cursor:
    sql = "select distinct card_name from \
           quick_card_data natural join type_join \
           where should_use = 1"
    cursor.execute(sql)
    uniqueCardNamesResponse = cursor.fetchall()
    uniqueCardNames = [r[0] for r in uniqueCardNamesResponse]
    #print(uniqueCardNames)

try:
    for name in uniqueCardNames:
        try:
            with db.cursor() as cursor:
                sql = "select card_name from unique_card_data where card_name = %s"
                cursor.execute(sql, name)
                exists = cursor.fetchone()

            if exists is None:
                # get the first appearance of that name
                with db.cursor() as cursor:
                    sql = "select card_id, card_name, set_name, releaseDate from \
                    quick_card_data join sets on (quick_card_data.set_name = sets.setCode) \
                    where card_name = %s \
                    and releaseDate is not null \
                    order by releaseDate \
                    limit 1"
                    cursor.execute(sql, name)
                    idResponse = cursor.fetchone()
                    print(idResponse)
                    cardId = idResponse[0]
                    #print(cardId)

                # get all the data for that card
                with db.cursor() as cursor:
                    sql = "select * from cards where card_id = %s"
                    cursor.execute(sql, cardId)
                    cardResponse = cursor.fetchone()
                    #print(cardResponse)

                with db.cursor() as cursor:
                    sql = "select type_name from type_join natural join card_types where should_use = 1 and card_id = %s"
                    cursor.execute(sql, cardId)
                    typeResponse = cursor.fetchone()
                    #print(typeResponse)
                    type = typeResponse[0]

                statusInt = randint(0,9)
                # put the new entry in the new table
                with db.cursor() as cursor:
                    sql = "insert into unique_card_data (card_id, card_name, set_name, card_uri, image_data, experiment_status, label) values (%s,%s,%s,%s,%s,%s,%s)"
                    cursor.execute(sql, (cardResponse[0], cardResponse[1], cardResponse[2], cardResponse[3], cardResponse[4], getStatus(statusInt), type))
                    db.commit()
                    print("card added ", counter)
                    counter += 1
        except:
            print("card error for ", name)
            errorCards.append(name)

finally:
    db.close()
    print("error cards")
    print(errorCards)
