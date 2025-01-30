import re
import json 
import os
import string

from collections import Counter
from nltk.corpus import stopwords

from langchain_core.documents.base import Document
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

def setup_chat_model(use_groq=None, groq_api=None, use_gpt=None, openai_key=None):
    # Setup chat model.
    # Run using groq with Mixtral
    if use_groq == 'True':
        if groq_api is None:
            raise ValueError("If use_groq is True, the groq_api parameter must be set.")
        api = groq_api
        chat_model = ChatGroq(temperature=0, groq_api_key=api, model_name="mixtral-8x7b-32768")
    
    # Run using gpt 3.5 turbo
    elif use_gpt == 'True':
        print(f"Using GPT-3.5 Turbo.")
        if openai_key is None:
            raise ValueError("If use-gpt is True, the openai-key parameter must be set.")
        openai_api_key = openai_key
        chat_model = ChatOpenAI(
            model_name='gpt-3.5-turbo-16k',
            temperature = 0,
            openai_api_key = openai_key,  
            max_tokens=500
        )
    # Run locally using Mixtral
    else:
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:1111"
        chat_model = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            model="TheBloke/Mixtral-8x7B-Instruct-v0.2-GPTQ",
        )
    return chat_model


def save_questions_answers(input_list, document_path, model_name, simple=True, save_to_questions=True, split_documents_vector=None, input_pages=None, retrieved_chunks_vec=None, n_retrieved=None, chunk_pages=None, overlap_pages=None, ir_system=None, embedding_model=None):
    """
    Saves input_list to a file. The file is named after the document it was generated from.
    If save_to_questions is True, then they are saved in data/questions otherwise in data/answers.
    The file is saved as document_name + model_name + .txt where model_name is the name of the model used to generate the questions or answers.
    If question was generated using generate_simple, ensure simple is True as it will be saved in the corresponding directory.
    
    Input:
    - input_list: list of strings, the list of questions or answers to be saved.
    - document_path: str, the path to the document from which the questions or answers were generated.
    - model_name: str, the name of the model used to generate the questions or answers.
    - simple: bool, set to True if the questions were generated using the simple method, False otherwise.
    - save_to_questions: bool, set to True if the input_list contains questions, False if it contains answers.

    - split_documents_vector: list of ints, the vector containing the split documents used for question generation.
    - input_pages: int, the number of pages in each split used for question generation.

    - retrieved_chunks_vec: list of lists, where each list contains the ID of the chunks used to generate that answer. 
    - n_retrieved: int, the number of retrieved chunks used for answer generation.
    - chunk_pages: int, the number of pages in each chunk used for answer generation.
    - overlap_pages: int, the number of pages to overlap between chunks used for answer generation.
    - ir_system: str, the IR system used for retrieving the documents.
    - embedding_model: str, the name of the Huggingface sentence transformer model used for embedding the documents with vector storages.
    """
    # Replace otherwise invalid characters in the model name.
    if embedding_model:
        embedding_model = embedding_model.replace('/', '-')
     
    # If there are no input_list generated, return None.
    if input_list is None:
        print(f"No questions generated.")
        return
    # Ensure that the input_list are numbered coherently
    question_or_answer = 'questions' if save_to_questions else 'answers'
    input_list = [re.sub(r'^\d+\.', f'{question_or_answer} {idx+1}.', x) for idx, x in enumerate(input_list)]

    filename_without_extension = re.search(r'([^\/]+)(?=\.\w+$)', document_path).group(1)
    
    # Saving questions + associated document splits.
    if save_to_questions:
        if simple:
            save_path = 'data/questions_simple/' + filename_without_extension + '_' + model_name + '_' + '.txt'
            save_path_doc_split = 'data/questions_simple/' + filename_without_extension + '_'  + model_name +  '_doc_split.txt'
        else:
            save_path = 'data/questions/' + filename_without_extension + '_' + model_name + '.txt'
            save_path_doc_split = 'data/questions/' + filename_without_extension + '_'  + model_name + '_doc_split.txt'
    
    # Saving answers + associated chunks of retrieved documents.
    else:
        if simple:
            save_path = 'data/answers_simple/' + filename_without_extension + '_' + model_name + '_' + ir_system + '_' + embedding_model + '_' + str(n_retrieved) + '_' + str(chunk_pages) + '_' + str(overlap_pages) + '.txt'
            save_path_chunks_retrieved = 'data/answers_simple/' + filename_without_extension + '_' + model_name + '_' + ir_system + '_' + embedding_model + '_' + str(n_retrieved) + '_' + str(chunk_pages) + '_' + str(overlap_pages) + '_chunks_retrieved.txt'
        else:
            save_path = 'data/answers/' + filename_without_extension + '_' + model_name + '_' + ir_system + '_' + embedding_model + '_' + str(n_retrieved) + '_' + str(chunk_pages) + '_' + str(overlap_pages) + '.txt'
            save_path_chunks_retrieved = 'data/answers/' + filename_without_extension + '_' + model_name + '_' + ir_system + '_' + embedding_model + '_' + str(n_retrieved) + '_' + str(chunk_pages) + '_' + str(overlap_pages) + '_chunks_retrieved.txt'
                
    with open(save_path, 'w') as file:
        for q in input_list:
            file.write(q + '\n')
    
    # Save the document splits used for question generation
    if input_pages is not None and split_documents_vector is not None:
        with open(save_path_doc_split, 'w') as file:
            file.write(f"{input_pages}\n")
            for q in split_documents_vector:
                file.write(str(q) + '\n')
    
    # Save the chunks of the retrieved documents used for answer generation
    if n_retrieved is not None and chunk_pages is not None and overlap_pages is not None:
        with open(save_path_chunks_retrieved, 'w') as file:
            file.write(f"{n_retrieved} {chunk_pages} {overlap_pages}\n")
            for sublist in retrieved_chunks_vec:
                for item in sublist:
                    file.write(f"{item} ")
                file.write('\n')

##### UTILITIES FOR QUESTION GENERATION ####
def prepare_document(document_path):
    """
    Takes as input a document path and returns a string with the document's content.
    Some preprocessing is applied to remove curly braces from the document as they can cause issues with the LLM model.
    """
    with open(document_path, 'r') as file:
        document = json.load(file)

    document = [list(document.keys())[i] + ' ' + list(document.values())[i] for i in range(len(document))] 
    document = ''.join(document)
    document = document.replace('}', '')
    document = document.replace('{', '')
    
    return document

def remove_answers(questions_list):
    """
    Input:
    - questions_list: A list of questions (and potentially answers which are removed by filtering).
    Output:
    - filtered_questions_list: A list of questions without any answers.
    Filter to remove answers from the list of questions.
    """
    # Remove them if they do not contain a question mark.
    questions_list = [x for x in questions_list if '?' in x]
    
    # Remove them if they start by Risposta or Response or Answer
    pattern = r"^(Risposta|Response|Answer)"
    questions_list = [x for x in questions_list if not re.match(pattern, x)]
    
    return questions_list


def split_document_pages(document, input_pages=1):
    """
    Given a document, it splits it into pages (according to page_i delimiters) and then joins them in groups of input_pages
    """
    pages = re.split(r'page_\d+', document)
    pages = [page.strip() for page in pages if page.strip()]

    pages_join = [''.join(pages[i:i+input_pages]) for i in range(0, len(pages), input_pages)]
    return pages_join

def renumber_questions(questions):
    """
    Input list of questions and renumber them
    """
    questions_no_number = [re.sub(r'^\d+\.', '', x) for x in questions]
    questions_number = [f'{idx+1}. {x}' for idx, x in enumerate(questions_no_number)]

    return questions_number

#### UTILITIES FOR ANSWER GENERATION ####
def split_with_pattern(string, pattern):
    """
    Splits a string with a given pattern while also retaining the pattern inside the text.
    """
    result = []
    current = ""
    for substr in re.split(pattern, string):
        if re.match(pattern, substr):
            current = substr
        elif substr.strip() != "":
            current += substr
            result.append(current.strip())
            current = ""
    return result


def custom_doc_splitter(document_path, chunk_pages, overlap_pages = None):
    """
    Takes a document and splits it into pages while retaining the page information. Then each chunk consists of n_pages. 
    Additionally if an overlap parameter is specified, the previous overlap_pages are included in the next chunk as well.
    It returns a list of Document objects.
    Input:
    - document_path: str, path to the document
    - n_pages: int, number of pages to be contained in each chunk
    Output:
    - documents_formatted: list of Document objects, each containing a chunk of the document
    """

    # Read document from the associated path 
    with open (document_path, 'r') as f:
        documents = json.load(f)

    # Count the number of pages in the document and ensure there are sufficient pages to split
    n_pages = len(documents)
    while chunk_pages > n_pages:
        chunk_pages -= 1

    # Check if the amount of overlap is allowed
    if overlap_pages > chunk_pages:
        raise ValueError("Overlap pages cannot be greater than the number of pages in each chunk")
    
    # Split the document into pages and retain the page numbering
    documents_formatted = [list(documents.keys())[i] + ': ' + list(documents.values())[i] for i in range(len(documents))]

    # Remove the curly braces from the document (cause issues with LLM for some reason).
    document_formatted = [x.replace('}', '').replace('{', '') for x in documents_formatted]

    # Join chunk_pages into a single chunk
    documents_formatted = [''.join(documents_formatted[i:i+chunk_pages]) for i in range(0, len(documents_formatted), chunk_pages)]
    
    # Include specified amount of overlap
    if overlap_pages is not None:
        pattern = r"(page_\d+)"
        
        documents_formatted_overlap = [None] * len(documents_formatted)
        documents_formatted_overlap[0] = documents_formatted[0]

        for idx, doc in enumerate(documents_formatted):
            if idx == 0:
                pass
            else:
                # Extract the overlap_pages from the previous chunk and then prepend them to the current one.
                previous_element = documents_formatted[idx-1]
                previous_pages = split_with_pattern(previous_element, pattern)
                previous_last_pages = previous_pages[-overlap_pages:]
                documents_formatted_overlap[idx] = ''.join(previous_last_pages) + doc

    documents_formatted = documents_formatted_overlap
    documents_formatted = [Document(doc) for doc in documents_formatted]
    
    return documents_formatted


def get_questions_from_file(file_path):    
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            questions = f.readlines()

        return questions
    return None

def extract_document_name(document_path):
    """
    Given the input document path, it extracts just the name of the document.
    This is needed to append to the questions directory to ensure the correct file is selected.
    """
    match = re.search(r'[^/]+(?=\.\w+$)', document_path)
    
    if match:
        return match.group(0)
    else:
        return None

#### Utilities for evaluation
def preprocess(text, concatenate=True):
    """
    Input:
    - text: A list of questions.
    - concatenate: A boolean indicating whether to concatenate the questions into a single list of words or not.
    Output:
    - processed_text: A list of lists of words if concatenate is False, otherwise a list of words.
    
    Takes as input a list of questions.
    If concatenate is True: Returns a list containing the words from all questions concatenated together.
    If concatenate is False: Returns a list of lists of words.
    """
    # Extract italian stopwords
    stopwords_ita = set(stopwords.words('italian'))
    
    processed_text = text

    # Remove page information
    processed_text = [re.sub(r'page_\d+', '', sub_text).strip() for sub_text in processed_text]

    # Replace apostrophe with whitespaces
    processed_text = [sub_text.replace("‘", " ").replace("’", " ").replace("'", " ").replace("–", " ") for sub_text in processed_text]

    # Remove punctuation
    processed_text = [sub_text.translate(str.maketrans('', '', string.punctuation)) for sub_text in processed_text]

    # Remove numbers
    processed_text = [re.sub(r'\d+', '', sub_text) for sub_text in processed_text]

    # Remove single letters
    processed_text = [re.sub(r'\b\w\b', '', sub_text) for sub_text in processed_text]

    # Lowercase
    processed_text = [sub_text.lower() for sub_text in processed_text]
    
    # Remove additional whitespaces
    processed_text = [re.sub(r'\s+', ' ', sub_text) for sub_text in processed_text]

    # Remove stopwords
    processed_text = [sub_text.split(' ') for sub_text in processed_text]
    processed_text = [[word for word in sub_text if word not in stopwords_ita] for sub_text in processed_text]

    if concatenate:
        processed_text = [w for question in processed_text for w in question]
    
    return processed_text

    

def reduce_fastext_dimension(fastext_path, reduced_fastext_path, dimension=50):
    """
    Given a fasttext model, it reduces the dimensionality to the specified dimension and saves it to the specified path.
    """
    ft = fasttext.load_model(fastext_path)
    fasttext.util.reduce_model(ft, dimension)
    
    ft.save_model(reduced_fastext_path)    


def enlist_answers(text):
    """
    Given the answers in the output format, they are correctly split and placed in a list.
    """
    reformatted_text = []
    current_answer = ""
    
    for line in text:
        if line.startswith("answers"):
            if current_answer:
                reformatted_text.append(current_answer.strip())
            current_answer = line.strip() + ": "
        else:
            current_answer += line.strip() + " "
    
    # Append the last answer
    if current_answer:
        reformatted_text.append(current_answer.strip())
    
    return reformatted_text


def filter_answers(answers):
    """
    Given a list of answers, if an answer contains ... "Non ho abbastanza informazioni" ..., replace simply with "Non ho abbastanza informationi", 
     deleting all additional text
    """
    answers = ['Non ho abbastanza informazioni per rispondere alla domanda' if 'Non ho abbastanza informazioni per rispondere alla domanda' in ans else ans for ans in answers]
    return answers


def extract_retrieved_chunks(answers, answers_chunks, document_path):

    # Read first page of each of the chunks.
    answers_chunks_ids = answers_chunks[1:]
    answers_chunks_ids = [x.replace('\n', '').strip() for x in answers_chunks_ids]
    retrieved_chunks_initial_pages = [tuple([f"page_{x}" for x in y.split()]) for y in answers_chunks_ids]

    n_retrieved, chunk_pages, overlap_pages = tuple([int(x) for x in answers_chunks[0].split(' ')])
    
    doc_splitted_chunks = custom_doc_splitter(document_path, chunk_pages, overlap_pages)  # Split in same way and extract text
    doc_splitted_chunks = [x.page_content for x in doc_splitted_chunks]

    retrieved_chunks_joined = []
    for initial_pages in retrieved_chunks_initial_pages:
        retrieved_chunks = [x for x in doc_splitted_chunks if x.startswith(initial_pages)]
        retrieved_chunks = '\n'.join(retrieved_chunks)
        retrieved_chunks_joined.append(retrieved_chunks)

    return retrieved_chunks_joined