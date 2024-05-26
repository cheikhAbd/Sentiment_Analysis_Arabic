import gradio as gr
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

# Charger le modèle et le vectoriseur
model = pickle.load(open('Dataset/trained_model.sav', 'rb'))

# Vous devez sauvegarder et charger le vectoriseur aussi, car il doit transformer le texte de la même manière qu'il l'a fait lors de l'entraînement du modèle
vectorizer = pickle.load(open('Dataset/vectorizer.sav', 'rb'))

# Chemin du fichier JSON
json_file_path = 'user_inputs.json'

# Initialiser le fichier JSON s'il n'existe pas
if not os.path.exists(json_file_path):
    with open(json_file_path, 'w') as f:
        json.dump([], f)


def predict_sentiment(text):
    # Transformer le texte avec le vectoriseur
    text_vectorized = vectorizer.transform([text])
    
    # Faire une prédiction
    prediction = model.predict(text_vectorized)
    
    # Déterminer le sentiment
    sentiment = "Negative sentiment" if prediction[0] == 0 else "Positive sentiment"
    
    # Charger les entrées existantes
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Ajouter la nouvelle entrée
    data.append({"Text": text, "Sentiment": sentiment})
    
    # Sauvegarder les données mises à jour dans le fichier JSON
    with open(json_file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return sentiment


# Créer l'interface Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a tweet here..."),
    outputs=gr.Textbox(),
    title="Arabic Sentiment Analysis",
    description="Enter an Arabic tweet to get the sentiment prediction."
)

# Lancer l'interface
iface.launch()
