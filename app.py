import gradio as gr
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import mysql
from mysql.connector import Error

# Charger le modèle et le vectoriseur
model = pickle.load(open('Model/trained_model.sav', 'rb'))
vectorizer = pickle.load(open('Model/vectorizer.sav', 'rb'))

# Configurer la connexion à la base de données MySQL
def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="sentimentAnalytic"
        )
        if connection.is_connected():
            print("Connexion réussie à la base de données MySQL")
    except Error as e:
        print(f"Erreur de connexion à la base de données MySQL : {e}")
    return connection

# Créer la table si elle n'existe pas déjà
def create_table(connection):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS dataset_arabic (
        id INT AUTO_INCREMENT PRIMARY KEY,
        text VARCHAR(255) NOT NULL,
        sentiment VARCHAR(255) NOT NULL
    )
    """
    cursor = connection.cursor()
    cursor.execute(create_table_query)
    connection.commit()

# Insérer les données dans la table
def insert_input(connection, text, sentiment):
    insert_query = """
    INSERT INTO dataset_arabic (text, sentiment)
    VALUES (%s, %s)
    """
    cursor = connection.cursor()
    cursor.execute(insert_query, (text, sentiment))
    connection.commit()

def predict_sentiment(text):
    # Transformer le texte avec le vectoriseur
    text_vectorized = vectorizer.transform([text])
    
    # Faire une prédiction
    prediction = model.predict(text_vectorized)
    
    # Déterminer le sentiment
    sentiment = "مشاعر سيئة للأسف 😢" if prediction[0] == 0 else "مشاعر رائعه 😊"
    
    # Connexion à la base de données
    connection = create_connection()
    if connection.is_connected():
        insert_input(connection, text, "neg" if sentiment == "مشاعر سيئة للأسف 😢" else "pos" )
        connection.close()
    
    return sentiment

# Créer l'interface Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="أدخل التغريدة هنا...."),
    outputs=gr.Textbox(),
    title="تحليل المشاعر العربية",
    description="أدخل تغريدة باللغة العربية للحصول على توقعات المشاعر"
)

# Lancer l'interface
iface.launch(share=True)

# Initialiser la connexion et créer la table lors du démarrage du script
connection = create_connection()
if connection.is_connected():
    create_table(connection)
    connection.close()
