import json
import scrython
import urllib.request as request
import pymysql.cursors

db = pymysql.connect(host="localhost",
                     user="root",
                     passwd="root",
                     db="aiproject",
                     port=3307)

with open('sets.json') as f:
    data = json.load(f)
i = 0
for key in data:
    print(i)
    setName = data[key]["name"]
    releaseDate = data[key]['releaseDate']
    try:
        with db.cursor() as cursor:
            sql = "insert into sets (setCode, setName, releaseDate) values (%s, %s, %s)"
            cursor.execute(sql, (key, setName, releaseDate))
        db.commit()
        i += 1
    except:
        pass
    finally:
        pass

db.close()
