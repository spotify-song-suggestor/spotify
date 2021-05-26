from sqlite3.dbapi2 import connect
import pandas as pd
import sqlite3 as sq3
import csv

df = pd.read_csv("data/spotifyData.csv")
#df.drop(index=0, axis=0, inplace=True)
#df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.columns)
conn = sq3.connect("spotify.sqlite3")
curs = conn.cursor()
curs.execute("""CREATE TABLE IF NOT EXISTS spotify (
    'name' TEXT,
    popularity FLOAT,
    duration_ms FLOAT,
    explicity INT,
    artists TEXT,
    release_date DATETIME,
    danceability FLOAT,
    energy FLOAT,
    'key' FLOAT,
    loudness FLOAT,
    mode  FLOAT,
    speechiness FLOAT,
    acousticness FLOAT,
    instrumentalness FLOAT,
    liveness FLOAT,
    valence FLOAT,
    tempo FLOAT,
    time_signature FLOAT 
    );""")
conn.commit()

df.to_sql('spotify', conn, if_exists='replace', index=False)

curs.execute("""
SELECT COUNT(*) FROM spotify;
""").fetchall()

query = curs.execute("""
select name, artists, danceability from spotify 
where danceability > 0.8
""").fetchall()
