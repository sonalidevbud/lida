import base64
import json
import logging
from typing import Any, List, Union
import os
import io
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import tiktoken
from diskcache import Cache
import hashlib
from PIL import Image


logger = logging.getLogger(__name__)


def get_dirs(path: str) -> List[str]:
    return next(os.walk(path))[1]


def clean_column_names(df):
    # create a copy of the dataframe to avoid modifying the original data
    cleaned_df = df.copy()

    # iterate over column names in the dataframe
    for col in cleaned_df.columns:
        # check if column name contains any special characters or spaces
        if re.search('[^0-9a-zA-Z_]', col):
            # replace special characters and spaces with underscores
            new_col = re.sub('[^0-9a-zA-Z_]', '_', col)
            # rename the column in the cleaned dataframe
            cleaned_df.rename(columns={col: new_col}, inplace=True)

    # return the cleaned dataframe
    return cleaned_df


def read_dataframe(file_location):
    file_extension = file_location.split('.')[-1]
    if file_extension == 'json':
        try:
            df = pd.read_json(file_location, orient='records')
        except ValueError:
            df = pd.read_json(file_location, orient='table')
    elif file_extension == 'csv':
        df = pd.read_csv(file_location)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file_location)
    elif file_extension == 'parquet':
        df = pd.read_parquet(file_location)
    elif file_extension == 'feather':
        df = pd.read_feather(file_location)
    elif file_extension == "tsv":
        df = pd.read_csv(file_location, sep="\t")
    else:
        raise ValueError('Unsupported file type')

    # clean column names and check if they have changed
    cleaned_df = clean_column_names(df)
    if cleaned_df.columns.tolist() != df.columns.tolist() or len(df) > 4500:
        if len(df) > 4500:
            logger.info(f"Dataframe has more than 4500 rows. We will sample 4500 rows.")
            cleaned_df = cleaned_df.sample(4500)
        # write the cleaned DataFrame to the original file on disk
        if file_extension == 'csv':
            cleaned_df.to_csv(file_location, index=False)
        elif file_extension in ['xls', 'xlsx']:
            cleaned_df.to_excel(file_location, index=False)
        elif file_extension == 'parquet':
            cleaned_df.to_parquet(file_location, index=False)
        elif file_extension == 'feather':
            cleaned_df.to_feather(file_location, index=False)
        elif file_extension == 'json':
            with open(file_location, 'w') as f:
                f.write(cleaned_df.to_json(orient='records'))
        else:
            raise ValueError('Unsupported file type')

    return cleaned_df

def naive_read_dataframe(file_location):
    file_extension = file_location.split('.')[-1]
    if file_extension == 'json':
        try:
            df = pd.read_json(file_location, orient='records')
        except ValueError:
            df = pd.read_json(file_location, orient='table')
    elif file_extension == 'csv':
        df = pd.read_csv(file_location)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file_location)
    elif file_extension == 'parquet':
        df = pd.read_parquet(file_location)
    elif file_extension == 'feather':
        df = pd.read_feather(file_location)
    elif file_extension == "tsv":
        df = pd.read_csv(file_location, sep="\t")
    else:
        raise ValueError('Unsupported file type')
    return df


def save_dataframe(df, file_location):
    df.to_csv(file_location, index=False)


def file_to_df(file_location: str):
    """ Get summary of data from file location """
    file_name = file_location.split("/")[-1]
    df = None
    if "csv" in file_name:
        df = pd.read_csv(file_location)
    elif "xlsx" in file_name:
        df = pd.read_excel(file_location)
    elif "json" in file_name:
        df = pd.read_json(file_location, orient="records")
    elif "parquet" in file_name:
        df = pd.read_parquet(file_location)
    elif "feather" in file_name:
        df = pd.read_feather(file_location)

    return df


def plot_raster(rasters: Union[str, List[str]], figsize: tuple = (10, 10)):
    plt.figure(figsize=figsize)

    if isinstance(rasters, str):
        rasters = [rasters]

    images = []

    # Load images, convert to RGB if needed, and resize to the same height
    max_height = 0
    for raster in rasters:
        decoded_image = base64.b64decode(raster)
        image_buffer = io.BytesIO(decoded_image)
        image = Image.open(image_buffer)

        # Convert RGBA images to RGB
        if image.mode == "RGBA":
            image = image.convert("RGB")

        max_height = max(max_height, image.height)

    for raster in rasters:
        decoded_image = base64.b64decode(raster)
        image_buffer = io.BytesIO(decoded_image)
        image = Image.open(image_buffer)

        # Convert RGBA images to RGB
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Resize the image to have the same height as max_height while preserving the aspect ratio
        width = int(image.width * (max_height / image.height))
        image = image.resize((width, max_height), Image.LANCZOS)

        # Convert the image back to numpy array
        image = np.array(image)

        images.append(image)

    # Concatenate images horizontally and normalize image data
    concatenated_image = np.concatenate(images, axis=1)
    if (
        concatenated_image.dtype == np.float32
        or concatenated_image.dtype == np.float64
    ):
        concatenated_image = np.clip(concatenated_image, 0, 1)
    else:
        concatenated_image = np.clip(concatenated_image, 0, 255)

    plt.imshow(concatenated_image)
    plt.axis("off")
    plt.box(False)
    plt.show()


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


def cache_request(cache: Cache, params: Any, values: Any = None) -> Any:
    # Generate a unique key for the request

    key = hashlib.md5(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()
    # Check if the request is cached
    if key in cache and values is None:
        print("retrieving from cache")
        return cache[key]

    # Cache the provided values and return them
    if values:
        print("saving to cache")
        cache[key] = values
    return values


def clean_code_snippet(markdown_string):
    # Extract code snippet using regex
    code_snippet = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', markdown_string)

    if code_snippet:
        return code_snippet.group(1)
    else:
        return markdown_string


def check_duplicates(array: list):
    """Check if there are any duplicate values in a list"""
    if len(array) == len(set(array)):
        return False
    else:
        return True

def handle_duplicates(array: list):
    """Handle duplicate values in a list"""
    new_array = array
    if len(array) != len(set(array)):
        new_array = []
        repeat_dict = {}
        count = 0
        for each in array:
            if each not in new_array:
                new_array.append(each)
            else:
                if each in repeat_dict:
                    count = repeat_dict[each] + 1
                    repeat_dict[each] = count
                else:
                    count = 1
                new_array.append(f'{each} {count}')
    return new_array