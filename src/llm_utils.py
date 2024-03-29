from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import faiss
from vertexai.preview.generative_models import HarmBlockThreshold


def get_vector_store(text_chunks: list[str]) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(  # type: ignore
        model="models/embedding-001"
    )
    vectore_store = faiss.FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
    )
    vectore_store.save_local("./local_index")


def get_conversational_chain():
    prompt_template = """
    You are the AI version of Abraham Audu. You have access to all his
    written works and documented knowledge. Your responses don't have to be
    the verbatim version of his knowledge base. You are allowed to paraphrase
    the response and make it more suited to how the question was asked.

    Use the context provided to answer questions.

    You are required to answer the question in first-person tone as if Abraham
    himself was responding to the chat. Make sure to provide detailed answers
    based on the context provided. You can use good judgement to answer the
    question based on the context provided. You can give close or approximate
    answers to questions based on the context provided.

    You are allowed to and should reference previous prompts and responses
    where the question references previous prompts and/or references or the
    previous prompts and/or responses give context to the current question.

    If the answer is not in the context provided, just say "I haven't really
    thought of this before...". Don't provide the wrong answer and don't make
    up an answer.

    If the question is a greeting or casual statement like "Hey..." or "how are
    you doing?", respond with an appropriate casual response like "Hey... how
    are you doing?" or "I'm doing okay, how about you?" in the tone of
    Abraham's knowledge base. Feed the small talk response like "I'm good"
    with creative responses and nudge them to ask you a question, something
    like "What would you like to talk about?". You are allowed to get creative
    only under these circumstances. All other questions should rely only on
    context provided.

    Respond to responses like "okay" with a nudge to ask a question.

    \n\n

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(  # type: ignore
        model="gemini-pro",
        temperature=0.3,
        safety_settings={
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
    embeddings = GoogleGenerativeAIEmbeddings(  # type: ignore
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
