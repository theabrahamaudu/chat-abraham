
import os
import dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import faiss

dotenv.load_dotenv()
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY"),
)


def get_pdf_text(pdf_doc: str) -> str:
    text = str()

    reader = PdfReader(pdf_doc)
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
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    use good judgement to answer the question based on the context of the document. If the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt,
    )
    return chain


def user_input(question):
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
        k=20
    )

    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents": docs,
            "question": question
        },
        return_only_outputs=True
    )

    return response["output_text"]
