import numpy as np
import pandas as pd
from SimpleClassifierForEvaluation import *
from table_evaluator import load_data, TableEvaluator


class GeneratedResultsEvaluator:

    def __init__(self, real_data_df_path: str, generated_data_df_path: str, **kwargs):
        if 'diabetes' in generated_data_df_path.lower():
            self.dataset_type = 'D'
        else:
            self.dataset_type = 'G'

        self.real_data_df_path = real_data_df_path
        self.generated_data_df_path = generated_data_df_path

        self.real_data_df = read_arff_file_as_dataframe(self.real_data_df_path)
        self.generated_data_df = read_arff_file_as_dataframe(self.generated_data_df_path)


        self.table_evaluator = TableEvaluator
        self.labels_to_num_dict = kwargs.get('labels_to_num_dict', {'tested_positive': 1, 'tested_negative': -1})

        self.classifier_for_eval_type = kwargs.get('classifier_for_eval', 'RandomForestClassifier')

        self.classifier_for_eval_real_ds = SimpleCLFForEvaluation(model_type=self.classifier_for_eval_type,
                                                                  data_path=self.real_data_df_path,
                                                                  labels_to_num_dict=self.labels_to_num_dict)
        self.classifier_for_eval_generated_ds = SimpleCLFForEvaluation(model_type=self.classifier_for_eval_type,
                                                                       data_path=self.real_data_df_path,
                                                                       labels_to_num_dict=self.labels_to_num_dict)

    def evaluate_datasets_similarity(self):
        print(f'table evaluator results:')
        categorical_cols = None
        df_columns = self.real_data_df.columns
        if self.dataset_type == 'D':
            categorical_cols = []
        else:
            categorical_cols = ['1', '3', '4', '6', '7', '9', '10', '12', '14', '15', '17', '19', '20', '21']

        self.table_evaluator = self.table_evaluator(self.real_data_df, self.generated_data_df)#, cat_cols=categorical_cols)

        self.table_evaluator.visual_evaluation()

        for col in df_columns.values:
            print(f'mean and std for feature: {col}')
            #todo: check if axis are retrieved (to add colname as title, if not, maybe passing in kwargs will work
            figs = self.table_evaluator.evaluate(target_col=col)


        classifier_score_on_real_ds = self.classifier_for_eval_real_ds.train_and_score_model()
        classifier_score_on_generated_ds = self.classifier_for_eval_generated_ds.train_and_score_model()

if __name__ == '__main__':
    gre_with_div = GeneratedResultsEvaluator(generated_data_df_path='Data/DIABETES_div_altered_data.arff', real_data_df_path='Data/diabetes.arff')
    gre_with_div.evaluate_datasets_similarity()

    gre_with_rand = GeneratedResultsEvaluator(generated_data_df_path='Data/DIABETES_rand_altered_data.arff', real_data_df_path='Data/diabetes.arff')
    gre_with_div.evaluate_datasets_similarity()
