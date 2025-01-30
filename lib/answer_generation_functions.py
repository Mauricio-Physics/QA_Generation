import time
import re
from tqdm import tqdm

from langchain_community.vectorstores import Chroma 
from chromadb.errors import InvalidDimensionException

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from lib.Utilities import get_questions_from_file, custom_doc_splitter


def answer_with_rag(query, vectorstore, n_retrieved, chat_model):
    """
    Input:
    - query: str, the question to be answered
    - vectorstore: the vectorstore containing the chunk embeddings
    - n_retrieved: int, the number of retrieved chunks
    - chat_model: the chat model to be used for answering the questions
    Output:
    - answer: a string containing the answer to the question.
    """

    retrieved_chunks = vectorstore.similarity_search(query, n_retrieved)
    all_chunks = [x.page_content for x in retrieved_chunks]
    # Extract initial page of each chunk
    chunk_initial_pages = [re.match(r'page_\d+', x).group(0) for x in all_chunks]
    chunk_initial_pages = [re.search(r'\d+', x).group(0) for x in chunk_initial_pages]

    #beginning_of_each_chunk = [x.page_content[0:10] for x in retrieved_chunks]
    #print(beginning_of_each_chunk)

    all_chunks = "\n\n".join(all_chunks)

    system = "<s> [INST] Rispondi in modo sintetico alla domanda considerando solamente le informazioni fornite nel contesto.\n\
        Non devi utilizzare alcune conoscenze pregresse.\n\
        Se non hai abbastanza informazioni per rispondere, scrivi 'Non ho abbastanza informazioni per rispondere alla domanda'.\n\
        Rispondi in italiano.\n\
        La risposta deve contenere la pagina contenente le informazioni necessarie per rispondere alla domanda. [/INST]"
    
    user = f"Domanda: {query}\n\
        Le informazioni che puoi usare sono riportate sotto:\n\
        {all_chunks}\n\
        Rispondi in modo sintetico alla domanda considerando solamente le informazioni fornite nel contesto.\n\
        La risposta deve contenere la pagina contenente le informazioni necessarie per rispondere alla domanda.\n\
        Se non hai abbastanza informazioni per rispondere, scrivi 'Non ho abbastanza informazioni per rispondere alla domanda'.\n\
        La risposta deve contenere la pagina contenente le informazioni necessarie per rispondere alla domanda."
    
    user = user.replace('}', '').replace('{', '')
    system = system.replace('}', '').replace('{', '')

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
    chain = prompt | chat_model
    answer = chain.invoke({})
    return answer, chunk_initial_pages


def answer_questions(questions, vectorstore, n_retrieved, chat_model, show_progress_bar=True):
    """
    Input:
    - questions: list, a list of questions to be answered
    - vectorstore: the vectorstore containing the chunk embeddings
    - n_retrieved: int, the number of retrieved chunks
    - chat_model: the chat model to be used for answering the questions
    Output:
    - answers: a list of strings containing the answers to the questions.
    """
    answers = []
    retrieved_chunks_initial_pages = []

    if show_progress_bar:
        # Initialize progress bar
        progress_bar = tqdm(total=len(questions), desc='Answered Questions', unit=' question')
    
    for idx, question in enumerate(questions):
        answer, chunk_initial_pages = answer_with_rag(question, vectorstore, n_retrieved, chat_model)
        answers.append(answer)
        retrieved_chunks_initial_pages.append(chunk_initial_pages)
        if show_progress_bar:
            progress_bar.update(1)  # Update progress bar

        # To avoid incurring in groq limitations.
        if idx % 2 == 0:
           time.sleep(60)

    progress_bar.close()  # Close progress bar when done
    return answers, retrieved_chunks_initial_pages

def extract_answers_rag(question_path, document_path, n_retrieved, chunk_pages, overlap_pages, chat_model, ir_system, embeddings_model_name):
    """
    Input:
    - document_path: str, the path to the file containing the document from which to extract the questions.
    - question_path: str, the path to the file containing the questions
    - n_retrieved: int, the number of retrieved chunks
    - chunk_pages: int, the number of pages in each chunk
    - overlap_pages: int, the number of pages to be included in the overlap
    - chat_model: the chat model to be used for answering the questions
    - embeddings: the embeddings to be used for the vectorstore
    Output:
    - answers: a list of strings containing the answers to the questions.
    """
    questions = get_questions_from_file(question_path)
    documents_formatted = custom_doc_splitter(document_path, chunk_pages, overlap_pages)
    if ir_system == 'chromadb':
        # To avoid issues if we want to experiment with multiple different embeddings (see: https://github.com/langchain-ai/langchain/issues/5046)
        # Construct the Chroma db from scratch (delete first, otherwise it just appends documents)
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        Chroma().delete_collection()
        vectorstore = Chroma.from_documents(documents=documents_formatted, embedding=embeddings)
    

    answers, retrieved_chunks_initial_pages = answer_questions(questions, vectorstore, n_retrieved, chat_model)
    return answers, retrieved_chunks_initial_pages