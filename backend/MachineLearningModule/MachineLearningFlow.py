import sys

from sklearn.model_selection import GridSearchCV
from MachineLearningModule.Classifiers import Classifiers
from MachineLearningModule.DataAcquisition import ReadDatasetFromCSV
from MachineLearningModule.DataPreProcessing import Normalisation, Conversions, HandleMissingValues, DimReduction, \
    NumBalancing
from MachineLearningModule.DataSplittingAndCV import DataSplitAndCVMethods
from MachineLearningModule.ModelEvaluation import ModelEvaluation


def machine_learning_flow(dataset_path, ml_algorithm):
    """
    Function containing the flow of the Machine learning module.

    Input:
        dataset_path - path to the dataset's CSV file
        ml_algorithm - acronym of ML algorithm chosen

    Output:
        classifier - the obtained classifier
        most_relevant_features - the most relevant features, obtained after applying RRFS
        evaluation_metrics - all obtained evaluation metrics with the classifier
    """

    # --------------- Data Acquisition ---------------
    dataset, class_name = ReadDatasetFromCSV.get_dataset_from_csv_file(dataset_path)

    # --------------- Data pre-processing ---------------
    x, y = data_pre_processing(dataset, class_name)

    most_relevant_features = x.columns.values

    accuracies, confusion_matrices, precisions, recalls, f1_scores, roc_aucs = [], [], [], [], [], []

    # --------------- Cross-validation ---------------
    outer_cv = DataSplitAndCVMethods.stratified_ten_fold_cv(x, y)

    classifier = None
    for i, (train_index, test_index) in outer_cv:
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # --------------- Train the model ---------------
        classifier = Classifiers.get_classifier(ml_algorithm, x_train, y_train)

        # --------------- Hyperparameter tuning ---------------
        classifier = hyperparameter_tuning(classifier, ml_algorithm, x_train, y_train)

        # --------------- Predict ---------------
        y_pred = classifier.predict(x_test)

        # --------------- Model Evaluation ---------------
        accuracy, confusion_matrix, precision, recall, f1_score, roc_auc = ModelEvaluation.model_evaluation(y_test,
                                                                                                            y_pred)
        accuracies.append(accuracy)
        confusion_matrices.append(confusion_matrix)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        roc_aucs.append(roc_auc)

    evaluation_metrics = [
        accuracies,
        confusion_matrices,
        precisions,
        recalls,
        f1_scores,
        roc_aucs
    ]

    if classifier is None:
        print("ERROR - No ML classifier obtained")
        sys.exit(1)

    return classifier, most_relevant_features, evaluation_metrics


def hyperparameter_tuning(classifier, ml_algorithm, x_val, y_val):
    """
    Function to perform hyperparameter tuning of the classifier (version allégée).
    """

    # Paramètres réduits pour un entraînement plus rapide
    if ml_algorithm == "RF":
        parameters = {
            'n_estimators': [100, 200],       # Seulement 2 valeurs
            'max_depth': [None, 5],           # Seulement 2 valeurs
            'criterion': ['gini']             # 1 seule valeur
        }
    elif ml_algorithm == "SVM":
        parameters = {
            'kernel': ['rbf', 'linear'],
            'C': [1, 10],
            'gamma': ['scale']
        }
    elif ml_algorithm == "KNN":
        parameters = {
            'n_neighbors': [3, 5],
            'metric': ['minkowski'],
            'weights': ['uniform']
        }
    else:
        return classifier

    from sklearn.model_selection import GridSearchCV
    tuned_classifier = GridSearchCV(classifier, parameters, cv=3, n_jobs=-1).fit(x_val, y_val)

    return tuned_classifier


def data_pre_processing(dataset, class_name):
    """
    Function for data pre-processing.

    Input:
        dataset
        class name

    Output:
        x - data matrix
        y - targets
    """

    # Converting categorical features to numerical
    dataset = Conversions.convert_all_cat_features_to_num_via_label_encoding(dataset)

    # Dealing with missing values
    dataset = HandleMissingValues.impute_missing_values_with_feature_mean(dataset)

    # Normalising the data
    dataset = Normalisation.min_max_normalisation(dataset)

    # Separating the features from the class label
    x = dataset.drop(class_name, axis=1)
    y = dataset[class_name]

    # Dimensionality reduction by Relevance-Redundancy Feature Selection (RRFS)
    x = DimReduction.relevance_redundancy_feature_selection(dataset, class_name, "FR", 0.3)

    # Numerosity balancing by random oversampling
    x, y = NumBalancing.random_oversampling(x, y)

    return x, y
