import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from huggingface_hub import InferenceClient
from smolagents import Tool, tool

from .db import ShelveDB
from .wiki import get_wiki_content


### convert table to markdown
@tool
def convert_pandas_table_to_markdown(table: pd.DataFrame) -> str:
    """
    Converts a pandas DataFrame to a markdown table.

    Args:
        table (pd.DataFrame): The DataFrame to convert.

    Returns:
        str: The markdown representation of the table.
    """
    return str(table.to_markdown())


### fetch text tool
@tool
def fetch_text_content(url: str) -> str:
    """
    Fetches the text content from a given URL.

    Args:
        url (str): The URL to fetch the text from.

    Returns:
        str: The text content of the page.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.RequestException as e:
        return f"Error fetching URL: {e}"


### Storage Tool
class RetrieveCSVStorageTool(Tool):
    name = "retrieve_csv_storage_tool"
    description = "Retrieves a CSV file from the storage and returns it as a pandas DataFrame."
    inputs = {
        "key": {
            "type": "string",
            "description": "The key to retrieve data from the table.",
        },
    }
    output_type = "any"

    def __init__(self, table_name: str, init_storage: bool, storage_path: str | None = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if storage_path is not None:
            ShelveDB.dir_path = Path(storage_path)
        self.storage = ShelveDB[pd.DataFrame](table_name, init=init_storage)

    def get_storage(self) -> ShelveDB[pd.DataFrame]:
        return self.storage

    def forward(self, key: str) -> pd.DataFrame:
        try:
            # Retrieve the CSV file from storage
            dataframe = self.storage.fetch(key)
        except Exception as e:
            return f"Error retrieving data: {e}"
        else:
            if dataframe is None:
                raise ValueError(f"No data found for key: {key}")
            return dataframe


### Wikipedia Content Extraction Tool


class WikiTool(Tool):
    name = "wiki_tool"
    description = """Get Wikipedia page content and tables.
    Returns a tuple containing the page content and a dictionary of tables extracted from the page.
    The page content is prefixed with the retrieved table key ({{table_1}}, {{table_2}}, ...).
    To understand what is contained in the tables, it is recommended to first display the content.
    Example 1:
        content, tables = get_wiki_content("Python_(programming_language)")
        print(content)
    
    The retrieved table object is are stored in storage.
    They can be retrieved using "retrieve_csv_storage_tool".
    Example 2:
        table:pd.DataFrame = retrieve_csv_storage_tool("table_1")
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The title of the Wikipedia page to visit. For example, 'Python_(programming_language)'.",
        },
        "language": {
            "type": "string",
            "description": "The language of the Wikipedia page. For example, 'en' for English, 'ja' for Japanese.",
        },
    }
    output_type = "array"

    def __init__(self, storage: ShelveDB[Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.storage = storage

    def forward(self, query: str, language: str) -> tuple[str, dict[str, pd.DataFrame]]:
        content, tables = get_wiki_content(query, language)
        self.storage.clear()
        for table_key, df in tables.items():
            self.storage.save(table_key, df)
        return content, tables


### Visual Question Answering Tool


def request_visual_qa(client: InferenceClient, question: str, image_url: str) -> str:
    contents = [{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": image_url}}]
    res = client.chat_completion(messages=[{"role": "user", "content": contents}], model="qwen/qwen2.5-vl-32b-instruct")
    content = res.choices[0].message.content
    if content is None:
        raise ValueError("No content returned from the model.")
    return content


class VisualQATool(Tool):
    name = "visual_qa_tool"
    description = "A tool that can answer questions about image."
    inputs = {
        "image_url": {
            "type": "string",
            "description": "The URL of the image to analyze. No extension needed.",
        },
        "question": {
            "type": "string",
            "description": "The question to ask about the image.",
        },
    }
    output_type = "string"
    client = InferenceClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    def forward(self, image_url: str, question: str) -> str:
        try:
            answer = request_visual_qa(self.client, question, image_url)
        except Exception as e:
            return f"Error: {str(e)}"
        else:
            return answer


### Speech Recognition Tool


def request_speech_recognition(client: InferenceClient, audio_file: str, model: str = "openai/whisper-large-v3") -> str:
    output = client.automatic_speech_recognition(audio_file, model=model)
    return output.text


class SpeechRecognitionTool(Tool):
    name = "speech_recognition"
    description = "Converts audio contents to text"
    inputs = {"audio_url": {"type": "string", "description": "URL of the audio file to transcribe. No extension needed."}}
    output_type = "string"
    client = InferenceClient(provider="fal-ai")
    _model = "openai/whisper-large-v3"

    def forward(self, audio_url: str) -> str:
        try:
            transcription = request_speech_recognition(self.client, audio_url, model=self._model)
        except Exception as e:
            return f"Error: {str(e)}"
        else:
            return transcription


### Excel Tool
@tool
def read_excel(file_url: str) -> pd.DataFrame:
    """
    Reads an Excel file from a given URL and returns the data as a DataFrame.

    Args:
        file_url (str): URL of the Excel file to read. No extension needed.
    Returns:
        pd.DataFrame: DataFrame containing the data from the first sheet of the Excel file
    """
    res = requests.get(file_url, timeout=30)
    res.raise_for_status()
    excel_data = BytesIO(res.content)
    df = pd.read_excel(excel_data)
    return df
