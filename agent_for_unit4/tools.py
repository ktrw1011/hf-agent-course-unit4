import os
import re
from io import BytesIO, StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
from smolagents import Tool, tool


### fetch text tool
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


### Wikipedia Content Extraction Tool


@tool
def get_wiki_content(title: str, language: str = "en") -> tuple[str, dict[str, pd.DataFrame]]:
    """
    Get Wikipedia page content and tables.

    Args:
        title: wikipedia page title (e.g., "Python_(programming_language)")
        language: wikipedia language (e.g., "en" for English, "ja" for Japanese)

    Returns:
        A tuple containing the page content as a string and a dictionary of tables
        extracted from the page. The keys of the dictionary are "table_1", "table_2", etc.
        and the values are pandas DataFrames representing the tables.
    """
    # パースAPIのURLを構築
    api_url = f"https://{language}.wikipedia.org/w/api.php"

    # APIパラメータ
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "disabletoc": True,
    }

    # リクエストを送信
    response = requests.get(api_url, params=params, timeout=30)  # type: ignore

    # レスポンスをチェック
    if response.status_code != 200:
        print(f"エラー: APIリクエストが失敗しました (ステータスコード: {response.status_code})")
        return "", {}

    # JSONレスポンスをパース
    data = response.json()

    # エラーチェック
    if "error" in data:
        print(f"API エラー: {data['error']['info']}")
        return "", {}

    if "parse" not in data:
        print(f"ページ '{title}' が見つかりませんでした")
        return "", {}

    # HTMLコンテンツを取得
    html_content = data["parse"]["text"]["*"]

    # テーブル情報を取得
    tables_dict: dict[str, pd.DataFrame] = {}

    # pd.read_htmlでテーブルをデータフレームとして抽出
    html_io = StringIO(html_content)
    dfs = pd.read_html(html_io)

    # テーブルごとに処理し、辞書に格納
    for i, df in enumerate(dfs):
        table_key = f"table_{i + 1}"
        tables_dict[table_key] = df

    # オリジナルのHTMLをコピーして、テーブルをプレースホルダに置き換える
    content_soup = BeautifulSoup(html_content, "html.parser")

    # テーブルをプレースホルダに置き換え
    for i, table in enumerate(content_soup.find_all("table", class_="wikitable")):
        table_placeholder = content_soup.new_tag("p")
        table_placeholder.string = f"{{{{table_{i + 1}}}}}"
        table.replace_with(table_placeholder)

    # クリーンな本文テキストを抽出（テーブルはプレースホルダに置き換え済み）
    for element in content_soup.find_all(["sup", "div.hatnote", "div.navbox"]):
        element.decompose()

    # 見出しとパラグラフを取得
    elements = content_soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"])
    text_content = []

    for element in elements:
        if element.name.startswith("h"):  # type: ignore
            level = int(element.name[1])  # type: ignore
            heading_text = element.get_text().strip()
            if heading_text:  # 空の見出しをスキップ
                text_content.append("\n" + "#" * level + " " + heading_text)
        elif element.name == "p":  # type: ignore
            paragraph_text = element.get_text().strip()
            if paragraph_text:  # 空のパラグラフをスキップ
                # テーブルプレースホルダの場合はそのまま追加
                if re.match(r"^\{\{table_\d+\}\}$", paragraph_text):
                    text_content.append(paragraph_text)
                else:
                    text_content.append(paragraph_text)

    # テキストコンテンツを結合
    content = "\n\n".join(text_content)

    return content, tables_dict


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
            "description": "The URL of the image to analyze.",
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
    inputs = {"audio_url": {"type": "string", "description": "URL of the audio file to transcribe"}}
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
        file_url (str): URL of the Excel file to read
    Returns:
        pd.DataFrame: DataFrame containing the data from the first sheet of the Excel file
    """
    res = requests.get(file_url, timeout=30)
    res.raise_for_status()
    excel_data = BytesIO(res.content)
    df = pd.read_excel(excel_data)
    return df
