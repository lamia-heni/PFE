from flask import Flask, request, jsonify
from FeatureExtractionModule import FeatureExtraction
from Util import Util
import os
import pandas as pd
from joblib import load

app = Flask(__name__)

# Charger les modèles et les features
print("Chargement du modèle...")
classifier = load("classifier.pkl")
most_relevant_features = load("features.pkl")
evaluation_metrics = load("metrics.pkl")
print("Modèle chargé avec succès.")

UPLOAD_FOLDER = "upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/analyze_apk', methods=['POST'])
def analyze_apk():
    apk = request.files.get('apk')

    if apk is None:
        return jsonify({"error": "Aucun fichier APK reçu."}), 400

    apk_path = os.path.join(UPLOAD_FOLDER, apk.filename)
    apk.save(apk_path)
    print(f"Fichier sauvegardé : {apk_path}")

    try:
        # Extraction des features
        extracted_features = FeatureExtraction.feature_extraction(apk_path, most_relevant_features)
        
        
        # Prédiction
        prediction = classifier.predict(extracted_features)
        print(f"Prédiction : {prediction[0]}")
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        print(f"Erreur pendant l'analyse : {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if os.path.exists(apk_path):
                os.remove(apk_path)
                print(f"Fichier supprimé : {apk_path}")
        except Exception as e:
            print("Erreur lors de la suppression du fichier :", e)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
