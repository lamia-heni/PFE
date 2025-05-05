from joblib import dump
from MachineLearningModule import MachineLearningFlow

# Load the model and features once when the server starts
dataset_path = "C:\\Users\\Lamia\\Desktop\\backend\\Android-Malware-Detection-with-Machine-Learning\\ProposedApproach\\Datasets\\Drebin.csv"
ml_algorithm = "RF"

print("Training model...")
classifier, features, metrics  = MachineLearningFlow.machine_learning_flow(dataset_path, ml_algorithm)

# Sauvegarde
dump(classifier, "classifier.pkl")
dump(features, "features.pkl")
dump(metrics, "metrics.pkl")
print("Modèle entraîné et sauvegardé.")
