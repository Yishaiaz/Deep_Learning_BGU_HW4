from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from preprocessing_utils import *


class SimpleCLFForEvaluation:
    """
    a class which constructs a simple classifier for ML-efficacy measurements and for
    section 2 of the assignment. We enabled the construction of several Sklearn’s classifiers
    in runtime. The models we enabled are RandomForestClassifier, GradientBoostingClassifier
    and the LogisticRegression model.
    """
    def __init__(self,
                 labels_to_num_dict: dict,
                 model_type: str = 'RandomForestClassifier',
                 data_path: str = DIABETES_PATH):
        """
        constructs the model by the model_type attribute and reads and prepares
        the dataset for training and testing (includes splitting)
        :param labels_to_num_dict:
        :param model_type:
        :param data_path:
        """
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

        self.data_x, self.data_y, _, _, _ = read_and_prepare_dataset(self.data_path,
                                                                     labels_to_num_dict=self.labels_to_num_dict,
                                                                     decode_categorical_columns=True)
        self.X_train, self.X_test, self.y_train, self.y_test = split_into_train_test(self.data_x, self.data_y)

    def train_and_score_model(self):
        """
        trains and scores the model. returns the model's score.
        if the model was already trained, throws a runtime exception.
        :return:
        """
        if self.model_trained:
            raise RuntimeWarning('already trained')
        self.train_model()
        model_score = self.score_model()

        return model_score

    def train_model(self):
        """
        trains the model.
        :return:
        """
        self.model.fit(self.X_train, self.y_train.iloc[:, 0].values)
        self.model_trained = True

    def score_model(self):
        """
        scores the model. if the model is not trained, throws a runtime exception.
        :return:
        """
        if not self.model_trained:
            raise RuntimeWarning('Model not yet trained, use the train_model method or the train_and_score method directly')
        return self.model.score(self.X_test, self.y_test)

    def model_confidence_score_distribution(self, to_predict: np.array = None):
        """
        returns the prediction probabilities for the test set.
        if the model wasn't trained, throws a runtime exception.
        :param to_predict:
        :return:
        """
        if not self.model_trained:
            raise RuntimeWarning('Model not yet trained, use the train_model method or the train_and_score method directly')
        if to_predict is None:
            to_predict = self.X_test
        return self.model.predict_proba(to_predict)

    def get_feature_importance(self):
        """
        returns the feature importance the model has given to each feature.
        if the model is LogisticRegression, returns the coefficient matrix.
        if the model wasn't trained, throws a runtime exception.
        :return:
        """
        if not self.model_trained:
            raise RuntimeWarning('Modedocumentation and production code level for preprocessing_utils.pyl not yet trained, use the train_model method or the train_and_score method directly')

        if self.model_type in ['RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier']:
            return self.model.feature_importances_

        elif self.model_type in ['LogisticRegression']:
            return self.model.coef_

