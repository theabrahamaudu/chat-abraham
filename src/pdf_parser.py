import os
import dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter


dotenv.load_dotenv()
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY"),
)


def scanDir(directory: str, extension: str) -> list[str]:
    """Check specified directory and return list of files with
    specified extension

    Args:
        extension (str): extension type to be searched for e.g. ".txt"

    Returns:
        list: strings of file names with specified extension
    """
    files: list = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            files.append(filename)
    files.sort()
    return files


def get_pdf_text(directory: str) -> str:
    text = str()

    for file in scanDir(directory, ".pdf"):
        reader = PdfReader(directory + "/" + file)
        for page in reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
    )
    return text_splitter.split_text(text)
