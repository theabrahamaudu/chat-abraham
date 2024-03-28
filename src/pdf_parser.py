
import os
import dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import faiss

from vertexai.preview.generative_models import (
    HarmCategory,
    HarmBlockThreshold
    )
from google.cloud.aiplatform_v1beta1.types.content import SafetySetting

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


def get_vector_store(text_chunks: list[str]) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vectore_store = faiss.FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
    )
    vectore_store.save_local("./local_index")


def get_conversational_chain():
    prompt_template = """
    You are the virual version of Abraham Audu. You have access to all his
    written works and docuemnted knowledge. Abraham and his auduence are
    confortable with religous concepts so such prompts or responses 
    should not be flagged as dangerous.

    Answer the question as Abraham as detailed as possible from the provided
    context, make sure to provide all the details, use good judgement to
    answer the question based on the context of the document.
    You are allowed to slightly rephrase the questions and reference previous prompts and responses in chat history to give
    contextual answers. If the answer is not in the context provided, just say
    "Abraham hasn't thought of this before...", don't provide the wrong answer
    and don't make up an answer.

    Remember you are chatting with a human, be conversational and answer in first-person tone as Abraham.
    Paraphrase the document content to fit the context of the question.\n\n

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        safety_settings = {
            7: HarmBlockThreshold.BLOCK_NONE,
            8: HarmBlockThreshold.BLOCK_NONE,
            9: HarmBlockThreshold.BLOCK_NONE,
            10: HarmBlockThreshold.BLOCK_NONE
        }
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt,
        # memory=memory
    )
    return chain


def user_input(question, chat_history=[]) -> str:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    vector_store = faiss.FAISS.load_local(
        folder_path="./local_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vector_store.similarity_search(
        query=question,
        k=10
    )

    chain = get_conversational_chain()

    response = chain.invoke(
        {
            "input_documents": docs,
            "question": question,
            "chat_history": chat_history,
        },
        return_only_outputs=True
    )

    return response["output_text"]
