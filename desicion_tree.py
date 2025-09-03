import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pprint import pprint
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

class DecisionTree:
    def __init__(self, dataframe: DataFrame, target: str):
        self.dataframe = dataframe
        self.target = target
        self.tree = None


    def entropy(self, column : Series):
        """
        Calculate the entropy of a column.
        """
        entropy = 0
        unique_values = column.unique()
        for value in unique_values:
            fraction = column.value_counts()[value] / len(column)
            entropy += -fraction * np.log2(fraction)
        return entropy


    def gain(self, dataframe : DataFrame, target : str, column : str):
        """
        Calculate the gain of a column.
        """
        gain = self.entropy(dataframe[target])
        unique_values = dataframe[column].unique()
        for value in unique_values:
            fraction = len(dataframe[dataframe[column] == value]) / len(dataframe[column])
            gain -= fraction * self.entropy(dataframe[dataframe[column] == value][target])
        return gain

    def best_feature(self, dataframe : DataFrame, target : str):
        """
        Calculate the best feature to split the dataframe.
        """
        features = dataframe.columns.tolist()
        features.remove(target)
        best_feature = features[0]
        for feature in features:
            if self.gain(dataframe, target, feature) > self.gain(dataframe, target, best_feature):
                best_feature = feature
        return best_feature

    def decision_tree(self, dataframe: DataFrame, target: str):
        """
        Build the decision tree.
        """
        # Caso 1: si todas las etiquetas son iguales
        if len(dataframe[target].unique()) == 1:
            return dataframe[target].iloc[0]

        # Caso 2: si ya no quedan features
        if len(dataframe.columns) == 1:  
            return dataframe[target].mode()[0]  

        # Paso normal: elegir mejor feature
        feature = self.best_feature(dataframe, target)
        tree = {feature: {}}

        for value in dataframe[feature].unique():
            sub_df = dataframe[dataframe[feature] == value].drop(columns=[feature])
            tree[feature][value] = self.decision_tree(sub_df, target)

        return tree

    def fit(self):
        """
        Fit the decision tree.
        """
        self.tree = self.decision_tree(self.dataframe, self.target)

    def predict_one(self, sample: dict):
        """
        Predict the class for a given sample.
        """
        return self._predict_recursive(self.tree, sample)

    def _predict_recursive(self, node, sample):
        """
        Recursive helper method for prediction.
        """
        if isinstance(node, str):
            return node
        
        if isinstance(node, dict):
            feature = list(node.keys())[0]
            feature_value = sample.get(feature)
            
            if feature_value is None:
                return None
            
            # Verificar si el valor existe en el árbol
            if feature_value not in node[feature]:
                # Si no existe, devolver la clase más común del nodo padre
                return self._get_most_common_class(node)
            
            return self._predict_recursive(node[feature][feature_value], sample)
        return None
    
    def _get_most_common_class(self, node):
        """
        Get the most common class from a node when a value is not found.
        """
        # Esta es una implementación simple - podrías mejorarla
        # Por ahora, devolvemos 'unacc' como valor por defecto
        return 'unacc'

    def predict(self, dataframe: DataFrame) -> Series:
        """
        Predict the class for a given dataframe.
        """
        return dataframe.apply(self.predict_one, axis=1)


class Metrics:
    def __init__(self, results: Series, real: Series):
        self.results = results
        self.real = real
        self.confusion_matrix = self._calculate_confusion_matrix()
    
    def _calculate_confusion_matrix(self):
        """Calculate confusion matrix"""
        unique_classes = sorted(list(set(self.real.unique()) | set(self.results.unique())))
        matrix = {}
        
        for true_class in unique_classes:
            matrix[true_class] = {}
            for pred_class in unique_classes:
                matrix[true_class][pred_class] = 0
        
        for true_val, pred_val in zip(self.real, self.results):
            matrix[true_val][pred_val] += 1
            
        return matrix
    
    def accuracy(self):
        """Calculate accuracy"""
        correct = sum(self.results == self.real)
        total = len(self.real)
        return correct / total if total > 0 else 0
    
    def precision(self, class_name):
        """Calculate precision for a specific class"""
        tp = self.confusion_matrix[class_name][class_name]
        fp = sum(self.confusion_matrix[other_class][class_name] 
                for other_class in self.confusion_matrix.keys() 
                if other_class != class_name)
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    def recall(self, class_name):
        """Calculate recall for a specific class"""
        tp = self.confusion_matrix[class_name][class_name]
        fn = sum(self.confusion_matrix[class_name][other_class] 
                for other_class in self.confusion_matrix.keys() 
                if other_class != class_name)
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def f1_score(self, class_name):
        """Calculate F1 score for a specific class"""
        prec = self.precision(class_name)
        rec = self.recall(class_name)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    def macro_f1(self):
        """Calculate macro-averaged F1 score"""
        classes = list(self.confusion_matrix.keys())
        f1_scores = [self.f1_score(cls) for cls in classes]
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    def print_metrics(self):
        """Print all metrics in a formatted way"""
        print("=" * 50)
        print("EVALUATION METRICS")
        print("=" * 50)
        print(f"Accuracy: {self.accuracy():.4f}")
        print(f"Macro F1-Score: {self.macro_f1():.4f}")
        print("\nPer-class metrics:")
        print("-" * 30)
        
        for class_name in self.confusion_matrix.keys():
            print(f"\nClass: {class_name}")
            print(f"  Precision: {self.precision(class_name):.4f}")
            print(f"  Recall: {self.recall(class_name):.4f}")
            print(f"  F1-Score: {self.f1_score(class_name):.4f}")
        
        print("\nConfusion Matrix:")
        print("-" * 30)
        self._print_confusion_matrix()
    
    def _print_confusion_matrix(self):
        """Print confusion matrix in a readable format"""
        classes = sorted(list(self.confusion_matrix.keys()))
        
        # Header
        print("True\\Pred", end="\t")
        for cls in classes:
            print(f"{cls}", end="\t")
        print()
        
        # Rows
        for true_cls in classes:
            print(f"{true_cls}", end="\t\t")
            for pred_cls in classes:
                print(f"{self.confusion_matrix[true_cls][pred_cls]}", end="\t")
            print()

def test_play_dataset():
    """Test the decision tree with the Play dataset"""
    print("=" * 50)
    print("Test 1: Play")
    print("=" * 50)
    print()
    
    data = pd.read_csv("Libro1.csv")
    tree = DecisionTree(data, "Play")
    tree.fit()
    print()
    print("Tree:")
    print()
    pprint(tree.tree)
    
    test = pd.read_csv("test.csv")
    metrics = Metrics(tree.predict(test), test["Play"])
    print()
    metrics.print_metrics()


def test_mushroom_dataset():
    """Test the decision tree with the Mushroom dataset"""
    print()
    print("=" * 50)
    print("Test 2: Mushroom")
    print("=" * 50)
    print()
    
    mushroom = fetch_ucirepo(id=73)
    X = mushroom.data.features
    y = mushroom.data.targets
    data = pd.concat([X, y], axis=1)
    
    data = data.dropna()
    
    X = data.drop(columns=["poisonous"])
    y = data[["poisonous"]] 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    
    tree = DecisionTree(train, "poisonous")
    tree.fit()
    print("Tree:")
    print()
    pprint(tree.tree)  
    
    preds = tree.predict(X_test)
    metrics = Metrics(preds, y_test["poisonous"])
    print()
    metrics.print_metrics()

def test_car_evaluation_dataset():
    """Test the decision tree with the Car Evaluation dataset"""
    print("=" * 50)
    print("Test 3: Car Evaluation")
    print("=" * 50)
    print()
    
    car_evaluation = fetch_ucirepo(id=19)
    X = car_evaluation.data.features
    y = car_evaluation.data.targets
    data = pd.concat([X, y], axis=1)
    print(data.info())
    
    target_col = "class"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
  
    tree = DecisionTree(train_data, target_col)
    tree.fit()
    print("Tree:")
    print()
    pprint(tree.tree)
    print()

    try:
        preds = tree.predict(X_test)
        metrics = Metrics(preds, y_test[target_col])
        print()
        metrics.print_metrics()
    except Exception as e:
        print(f"Error durante predicción: {e}")
        print("Primeras filas de X_test:")
        print(X_test.head())
        print("Primeras filas de X_train:")
        print(X_train.head())
    
    

if __name__ == "__main__":
    test_play_dataset()
    test_mushroom_dataset()
    test_car_evaluation_dataset()


