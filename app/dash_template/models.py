from flask_sqlalchemy import SQLAlchemy
from app import app, server,db


# server.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://spotify.sqlite3"
# db = SQLAlchemy(server)


def test():
    print("Hello")

class Spotify(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    popularity = db.Column(db.Float, nullable=False)
    duration_ms = db.Column(db.Float, nullable=False)
    explicity =  db.Column(db.Integer, nullable=False)
    artists = db.Column(db.String(120), unique=True, nullable=False)
    release_date = db.Column(db.String(10), nullable=False)
    danceability =  db.Column(db.Float, nullable=False)
    energy = db.Column(db.Float, nullable=False)
    key = db.Column(db.Float, nullable=False)
    loudness = db.Column(db.Float, nullable=False)
    mode =  db.Column(db.Float, nullable=False)
    speechiness = db.Column(db.Float, nullable=False)
    acousticness = db.Column(db.Float, nullable=False)
    instrumentalness = db.Column(db.Float, nullable=False)
    liveness = db.Column(db.Float, nullable=False)
    valence = db.Column(db.Float, nullable=False)
    tempo = db.Column(db.Float, nullable=False)
    time_signature = db.Column(db.Float, nullable=False) 

    def __repr__(self):
        return f"<User: {self.name}>"

    