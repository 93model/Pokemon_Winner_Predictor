import urllib.request as urllib
from flask import request
from flask import Flask, render_template
import sqlite3
import sys

from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from Section3_project.Pokemon_model import who_is_winner

app = Flask(__name__)

@app.route('/')
def index():       
    conn = sqlite3.connect('pokemonDB.db')
    cur = conn.cursor()
    cur.execute('SELECT "#","Kor_name" FROM pokemon_kor_name')
    data=cur.fetchall()
    conn.close()

    return render_template('index.html', data = data)


@app.route('/result', methods=['GET','POST'])
def result():
 
    first_pokemon = request.form.get('first_pokemon')
    my_pokemon = first_pokemon.split(',')
    my_pokemon_num = int(my_pokemon[0])

    second_pokemon = request.form.get('second_pokemon')
    ene_pokemon = second_pokemon.split(',')
    ene_pokemon_num = int(ene_pokemon[0])
    
    winner = who_is_winner(my_pokemon_num,ene_pokemon_num)
    #winner = list(winner)
    return render_template("result.html", winner=winner)
