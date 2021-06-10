import os

from pandas_profiling import ProfileReport

from global_vars import DIABETES_PATH, GERMAN_CREDIT_PATH
from preprocessing_utils import gather_numeric_and_categorical_columns, read_arff_file_as_dataframe

# load diabetes file into DataFrame
from utils import tsne

df = read_arff_file_as_dataframe(DIABETES_PATH)
# Gather numeric and categorical columns into a list
numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df)
# iterate over all categorical columns and convert decode to string
df = df.apply(lambda col: col.str.decode(encoding='UTF-8') if col.name in categorical_columns else col)

diabetes_output_file = 'diabetes_eda_report.html'
if not os.path.exists(diabetes_output_file):
    design_report = ProfileReport(df)
    design_report.to_file(output_file=diabetes_output_file)

# diabetes tsne
tsne(df, categorical_columns, hue='class', filename='diabetes_tsne', save_figure=True)

# load german credit file into DataFrame
df = read_arff_file_as_dataframe(GERMAN_CREDIT_PATH)
# Gather numeric and categorical columns into a list
numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df)
# iterate over all categorical columns and convert decode to string
df = df.apply(lambda col: col.str.decode(encoding='UTF-8') if col.name in categorical_columns else col)

german_credit_output_file = 'german_credit_eda_report.html'
if not os.path.exists(german_credit_output_file):
    design_report = ProfileReport(df)
    design_report.to_file(output_file=german_credit_output_file)

# german_credit tsne
tsne(df, categorical_columns, hue='21', filename='german_credit_tsne', save_figure=True)
