import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pprint import pprint


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
        self.tree = self.decision_tree(self.dataframe, self.target)

    def predict(self, sample: dict):
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
            
            return self._predict_recursive(node[feature][feature_value], sample)
        return None

if __name__ == "__main__":
    data = pd.read_csv("Libro1.csv")
    tree = DecisionTree(data, "Play")
    tree.fit()
    pprint(tree.tree)
    print(tree.predict({"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}))
