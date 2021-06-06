import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocessing_utils import *

class SimpleCLFForEvaluation:
    def __init__(self, **kwargs):
        self.df_path = kwargs.get('data_path', 'Data/diabetes.arff')
        self.model_type = kwargs.get('model', 'RandomForestClassifier')
        self.model = None
        self.model_trained = False
        if self.model_type == 'RandomForestClassifier':
            self.model = RandomForestClassifier()
        elif self.model_type == 'GradientBoostingClassifier':
            self.model = GradientBoostingClassifier()
        elif self.model_type == 'LogisticRegression':
            self.model = LogisticRegression()

        self.labels_to_num_dict = kwargs.get('labels_to_num_dict', {'tested_positive': 1,
                                                                    'tested_negative': -1})

        self.data_x, self.data_y = read_and_prepare_dataset(self.df_path,
                                                            labels_to_num_dict=self.labels_to_num_dict)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_x,
                                                                                self.data_y,
                                                                                test_size=0.30,
                                                                                random_state=42)

    def train_and_score_model(self):
        if self.model_trained:
            raise RuntimeWarning('already trained')
        self.train_model()
        model_score = self.score_model()
        print(f'model score on real data: {model_score}')

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        self.model_trained = True

    def score_model(self):
        return self.model.score(self.X_test, self.y_test)

    def get_feature_importance(self):
        if self.model_type in ['RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier']:
            return self.model.feature_importances_

        elif self.model_type in ['LogisticRegression']:
            return self.model.coef_
