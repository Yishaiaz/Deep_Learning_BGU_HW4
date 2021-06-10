from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from preprocessing_utils import *


class SimpleCLFForEvaluation:
    def __init__(self,
                 labels_to_num_dict: dict,
                 model_type: str = 'RandomForestClassifier',
                 data_path: str = DIABETES_PATH):
        np.random.seed(SEED)

        self.data_path = data_path
        self.model_type = model_type
        self.model_trained = False

        if self.model_type == 'RandomForestClassifier':
            self.model = RandomForestClassifier(random_state=SEED)
        elif self.model_type == 'GradientBoostingClassifier':
            self.model = GradientBoostingClassifier(random_state=SEED)
        elif self.model_type == 'LogisticRegression':
            self.model = LogisticRegression(random_state=SEED)

        self.labels_to_num_dict = labels_to_num_dict

        self.data_x, self.data_y, _, _ = read_and_prepare_dataset(self.data_path,
                                                                  labels_to_num_dict=self.labels_to_num_dict,
                                                                  decode_categorical_columns=True)
        self.X_train, self.X_test, self.y_train, self.y_test = split_into_train_test(self.data_x, self.data_y)

    def train_and_score_model(self):
        if self.model_trained:
            raise RuntimeWarning('already trained')
        self.train_model()
        model_score = self.score_model()
        print(f'model score on real data: {model_score}')

    def train_model(self):
        self.model.fit(self.X_train, self.y_train.iloc[:, 0].values)
        self.model_trained = True

    def score_model(self):
        return self.model.score(self.X_test, self.y_test)

    def get_feature_importance(self):  # TODO why we need featrure importance?
        if self.model_type in ['RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier']:
            return self.model.feature_importances_

        elif self.model_type in ['LogisticRegression']:
            return self.model.coef_

