import json
import logging
from typing import Union
import copy
import numpy as np
import pandas as pd
from lida.utils import clean_code_snippet, clean_column_names, naive_read_dataframe, check_duplicates, handle_duplicates
from lida.datamodel import TextGenerationConfig
from llmx import TextGenerator

logger = logging.getLogger(__name__)

system_prompt = """
System: You are a Data Analyst with experience in data analysis, python pandas, visualisation, data exploration & analysis who can help people relevant insights from their data.
Instruction: User will provide a description of the datasets, relevant column names, its data type (categorical, Numerical, DateTime, time series etc), a few sample rows etc. It will follow the structure - Table Description :: (column name || data type) ## (row value 1) ## (row value 2).  Analyse the given details and think step by step to create the following details in a JSON with the following details:
1. Shortened name of the column, that retains its semantic, functional & contextual meaning while less than 15 words & compatible with SQL DBs, LLMs, Pandas etc. The name should be understandable for a large language model like GPT4 (key: short_name)
2. A description of the column (key: description)
3. Different insights that could be generated out of this, as a list of questions (key: insights)
4. Relevance of the table with respect to the over all dataset (key: relevance)
"""

class Preprocessor():
    def __init__(self) -> None:
        self.cleaned_data = None

    def specific_drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = df.columns.tolist()
        for column in columns:
            if column.startswith('comment.'):
                df.drop(column, axis=1, inplace=True)
            elif column.endswith('.Text_box'):
                df.drop(column, axis=1, inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
    
    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns from a pandas DataFrame"""
        # Drop columns where all values are empty
        # data.dropna(axis=1, how='all', inplace=True)
        # Calculate the threshold for dropping columns (30% not nan)
        threshold = int(len(df) * 0.3)
        logger.info(f"Threshold to drop columns >> {threshold}")
        # Drop columns exceeding the threshold
        df.dropna(axis=1, thresh=threshold, inplace=True)
        return df

    def fill_na_by_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            dtype = df[col].dtype
            if np.issubdtype(dtype, np.number):
                # For numeric columns, fill NaN with the mean value
                df[col].fillna(df[col].mean(), inplace=True)
            elif dtype == 'object':
                # For object (string) columns, fill NaN with an empty string
                df[col].fillna('', inplace=True)
            elif np.issubdtype(dtype, np.datetime64):
                # For datetime columns, fill NaN with a specified value or method
                df[col].fillna(pd.Timestamp('1970-01-01'), inplace=True)
            else:
                # For other columns (e.g., bool), fill NaN with an appropriate value
                df[col].fillna('NA', inplace=True)
        return df

    def drop_row(self, index: int, df: pd.DataFrame) -> pd.DataFrame:
        """Drop a row from a pandas DataFrame"""
        df = df.drop(index)
        return df

    def modify_column_names(self, df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
        """Modify column names in a pandas DataFrame"""
        df = df.rename(columns=column_mapping)
        df = clean_column_names(df)
        return df

    def preprocess(
        self, data: Union[pd.DataFrame, str], file_name: str = "", n_samples: int = 3,
    ):
        """Preprocesses data from a pandas DataFrame or a file location"""

        # if data is a file path, read it into a pandas DataFrame, set file_name to the file name
        if isinstance(data, str):
            file_location = data
            file_name = data.split("/")[-1]
            data = naive_read_dataframe(data)
        self.specific_drop_columns(data)
        # data = self.drop_columns(data)
        column_names = data.columns.tolist()
        if check_duplicates(column_names):
            raise Exception('Duplicate column names found in the dataset')
        first_row_values = data.iloc[0].tolist()
        if check_duplicates(first_row_values):
            first_row_values = handle_duplicates(first_row_values)
        column_mapping = dict(zip(column_names, first_row_values))
        data = self.modify_column_names(data, column_mapping)
        data.drop(index=data.index[0], inplace=True)

        # alter column names
        with open('cleaned_col_names.json', 'r') as f:
            short_names = json.load(f)
        columns = data.columns.tolist()
        specific_col_map = {}
        for column in columns:
            if column in short_names:
                if short_names[column] == 'comment':
                    data.drop(column, axis=1, inplace=True)
                else:
                    specific_col_map[column] = short_names[column] 
        data = self.modify_column_names(data, specific_col_map)
        print(f"Data shape after dropping more columns >> {data.shape}")
        data = self.fill_na_by_dtype(data)
        self.cleaned_data = data
        return data

    def get_batch_short_column_names(self, summary: dict, text_gen: TextGenerator,
                textgen_config: TextGenerationConfig) -> dict:
        """Short names for long column names"""
        logger.info(f"Shortening column names")
        column_example = ''
        columns = []
        for field in summary['fields']:
            column_name = field['column']
            data_type = field['properties']['dtype']
            samples = field['properties']['samples']
            samples = list(filter(lambda x: x != '', samples))
            column_example += f"{column_name} || {data_type} "
            if samples:
                if len(samples) > 2:
                    column_example += f"## ({samples[0]}) ## ({samples[1]})"
                else:
                    column_example += f"## ({samples[0]})"
            column_example += '\n'
            columns.append(column_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
        {summary["dataset_description"]}
        {column_example}
        """},
        ]
        
        response = text_gen.generate(messages=messages, config=textgen_config)
        column_mapping = dict(zip(columns, columns))
        try:
            # json_string = clean_code_snippet(response.text[0]["content"])
            json_string = response.text[0]["content"]
            with open('test.txt', 'a') as f:
                f.write(f"\n{json_string}\n\n")
            response_json = json.loads(json_string)
            short_column_names = [response_json[each]['short_name'] for each in columns if each in response_json]
            if len(response_json) == len(summary['fields']):
                column_mapping = dict(zip(columns, short_column_names))
            else:
                raise Exception('The model did not return all column names')
        except json.decoder.JSONDecodeError:
            error_msg = f"The model did not return a valid JSON object while attempting to generate an enriched data summary. Consider using a base summary. {response.text[0]['content']}"
            logger.info(error_msg)
            # print(response.text[0]["content"])
            raise ValueError(f"{error_msg} : {response.usage}")
        except Exception as e:
            logger.error(f"{e}")
        return column_mapping

    def get_short_column_names(self, summary: dict, text_gen: TextGenerator,
        textgen_config: TextGenerationConfig) -> dict:
        """Short names for long column names"""
        logger.info(f"Shortening column names")
        summary_copy = copy.deepcopy(summary)
        batch_size = 10
        count = 0
        column_mapping = {}
        while True:
            summary_copy['fields'] = summary['fields'][count:count+batch_size]
            column_mapping_batch = self.get_batch_short_column_names(summary_copy, text_gen,textgen_config)
            column_mapping.update(column_mapping_batch)
            count += batch_size
            if count >= len(summary['fields']):
                break
        logger.info(f"column mapping >>> {column_mapping}")
        return column_mapping