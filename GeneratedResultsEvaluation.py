import numpy as np
import pandas as pd
from SimpleClassifierForEvaluation import *
from table_evaluator import load_data, TableEvaluator


class GeneratedResultsEvaluator:
    def __init__(self, real_data_df: pd.DataFrame, generated_data_df: pd.DataFrame, **kwargs):
        self.real_data_df = real_data_df
        self.generated_data_df = generated_data_df

        self.table_evaluator = TableEvaluator

        self.classifier_for_eval_type = kwargs.get('classifier_for_eval', 'RandomForestClassifier')
        self.classifier_for_eval_real_ds = SimpleCLFForEvaluation(model=self.classifier_for_eval_type,
                                                                  prepared_ds=self.real_data_df)
        self.classifier_for_eval_generated_ds = SimpleCLFForEvaluation(model=self.classifier_for_eval_type,
                                                                       prepared_ds=self.real_data_df)

    def evaluate_datasets_similarity(self):
        print(f'table evaluator results:')
        self.table_evaluator = self.table_evaluator(self.real_data_df, self.generated_data_df)
        df_columns = self.real_data_df.columns
        for col in df_columns: #t odo get df columns
            print(f'mean and std for feature: {col}')
            #todo: check if axis are retrieved (to add colname as title, if not, maybe passing in kwargs will work
            figs = self.table_evaluator.evaluate(target_col=col)


        classifier_score_on_real_ds = self.classifier_for_eval_real_ds.train_and_score_model()
        classifier_score_on_generated_ds = self.classifier_for_eval_generated_ds.train_and_score_model()
