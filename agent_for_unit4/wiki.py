import re
from io import StringIO
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup


def process_list_element(list_element: Any, indent: int = 0) -> str:
    """リスト要素を再帰的に処理する関数"""
    result = []

    is_ordered = list_element.name == "ol"

    for i, li in enumerate(list_element.find_all("li", recursive=False)):
        # リスト項目のテキストを取得
        # ネストされたリストを除いたテキストを取得
        item_text = ""
        for content in li.contents:
            if content.name not in ["ul", "ol"]:
                item_text += str(content)

        item_text = BeautifulSoup(item_text, "html.parser").get_text().strip()

        # 順序付きリストなら番号を、そうでなければ記号を使用
        prefix = "  " * indent + (f"{i + 1}. " if is_ordered else "* ")
        if item_text:
            result.append(prefix + item_text)

        # ネストされたリストを処理
        for nested_list in li.find_all(["ul", "ol"], recursive=False):
            nested_content = process_list_element(nested_list, indent + 1)
            if nested_content:
                result.append(nested_content)

    return "\n".join(result)


def get_wiki_content(title: str, language: str = "en") -> tuple[str, dict[str, pd.DataFrame]]:
    """
    Get Wikipedia page content and tables.

    Returns:
        A tuple containing the page content as a string and a dictionary of tables
        extracted from the page. The keys of the dictionary are "table_1", "table_2", etc.
        and the values are pandas DataFrames representing the tables.

    Example:
        content, tables = get_wiki_content("Python_(programming_language)")
        print(content)
        print(tables["table_1"])  # Access the first table

    Args:
        title: wikipedia page title (e.g., "Python_(programming_language)")
        language: wikipedia language (e.g., "en" for English, "ja" for Japanese)
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
        raise Exception(f"api error: {response.status_code} - {response.text}")

    # JSONレスポンスをパース
    data = response.json()

    # エラーチェック
    if "error" in data:
        raise Exception(f"api error: {data['error']['info']}")

    if "parse" not in data:
        raise Exception("api error: No parse data found")

    # HTMLコンテンツを取得
    html_content = data["parse"]["text"]["*"]

    # HTMLをパース
    soup = BeautifulSoup(html_content, "html.parser")
    content_soup = BeautifulSoup(html_content, "html.parser")

    # テーブル情報を取得
    tables_dict: dict[str, pd.DataFrame] = {}
    table_ids: list[tuple[str, str]] = []  # (table_id, table_html) のリスト

    # ターゲットとするテーブルを特定: wikitableとinfobox
    table_index = 1

    # まず、infobox（バイオグラフィーテーブル）を処理
    infoboxes = soup.find_all("table", class_=lambda c: c and "infobox" in c)
    for i, table in enumerate(infoboxes):
        table_id = f"table_{table_index}"
        table_ids.append((table_id, str(table)))
        table_index += 1

    # 次に、wikitableを処理
    wikitables = soup.find_all("table", class_="wikitable")
    for i, table in enumerate(wikitables):
        table_id = f"table_{table_index}"
        table_ids.append((table_id, str(table)))
        table_index += 1

    # 抽出したテーブルをpandasで処理
    for table_id, table_html in table_ids:
        try:
            dfs = pd.read_html(StringIO(table_html))
            if dfs:
                tables_dict[table_id] = dfs[0]
        except Exception:
            # テーブル解析に失敗した場合はスキップ
            continue

    # コンテンツ内のテーブルをプレースホルダに置き換え
    table_placeholders: dict[str, str] = {}

    # infoboxの処理
    for i, table in enumerate(content_soup.find_all("table", class_=lambda c: c and "infobox" in c)):
        table_id = f"table_{i + 1}"
        if table_id in tables_dict:
            placeholder = f"{{{{{table_id}}}}}"
            table_placeholders[table_id] = placeholder
            table_placeholder_tag = content_soup.new_tag("p")
            table_placeholder_tag.string = placeholder
            table.replace_with(table_placeholder_tag)

    # wikitableの処理（インデックスは続きから）
    wikitable_start_index = len(infoboxes) + 1
    for i, table in enumerate(content_soup.find_all("table", class_="wikitable")):
        table_id = f"table_{wikitable_start_index + i}"
        if table_id in tables_dict:
            placeholder = f"{{{{{table_id}}}}}"
            table_placeholders[table_id] = placeholder
            table_placeholder_tag = content_soup.new_tag("p")
            table_placeholder_tag.string = placeholder
            table.replace_with(table_placeholder_tag)

    # クリーンな本文テキストを抽出
    for element in content_soup.find_all(["sup", "div.hatnote", "div.navbox", "span.mw-editsection"]):
        element.decompose()

    # 見出し、パラグラフ、リストを取得
    elements = content_soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol"])
    text_content = []

    for element in elements:
        if element.name and element.name.startswith("h"):  # type: ignore
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
        elif element.name in ["ul", "ol"] and element.parent.name not in ["li", "ul", "ol"]:  # type: ignore
            # トップレベルのリストのみ処理（ネストされたものは親liで処理）
            list_content = process_list_element(element)
            if list_content:
                text_content.append(list_content)

    # テキストコンテンツを結合
    content = "\n\n".join(text_content)

    return content, tables_dict
