import json
import re
import string
import time
from tqdm import tqdm

from langchain_core.prompts import ChatPromptTemplate

from lib.Utilities import remove_answers, prepare_document, split_document_pages, renumber_questions

def extract_machine_name(document, chat_model):
    """
    Input: Document
    Output: The name of the machine described in the document if found, otherwise None.

    Given a document, the function uses the model to extract the name of the machine from the document.
    To reduce the number of tokens usage, only the first page of the document is passed.
    """
    system = f"<s> [INST] Devi analizzare dei documenti provenienti da un impianto manifatturiero.\n\
               Devi determinare il nome del macchinario descritto nel documento. [/INST]"

    # Retain only the first page of the document for the name extraction task.
    split_document = split_document_pages(document, input_pages=1)

    # Remove initial empty strings and only submit first page for name extraction
    while '' in split_document:
        split_document.remove('')
    
    # Pass only the first page for the name extraction task.
    document = split_document[0]
    
    # Use the model to extract the name of the machine
    user_machine_name = f"Quale è il nome del macchinario descritto nel documento?\n \
             Ritorna solamente il nome del macchinario.\n \
             Prima del nome del macchinario nella risposta inserisci il tag '[NAME]'. \n \
             Dopo il nome del macchinario nella risposta inserisci il tag '[NAME]' \n \
             Il documento completo è {document}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user_machine_name)])
    chain = prompt | chat_model
    machine_name = chain.invoke({})
    
    # Extract name, sometimes it may fail due to missing tags by the model.
    match = re.search(r'\[NAME\](.*?)\[NAME\]', machine_name.content)
    if match:
        machine_name = match.group(1)
        return machine_name
    else:
        #nprint("Nome macchina non trovato")
        return None


def extract_questions_simple(document_path, topics, chat_model, input_pages):
    """
    Input:
    - document: a string containing the document's content.
    - topics: a string (ITALIAN) containing the topics to generate questions about.
              The text should be formulated to be included in the system sentence described below.
    """
    document = prepare_document(document_path)
    n_pages = re.findall(r"page_\d+", document)
    if len(n_pages) < 3:
        # print(f"Document is too short, we assume it does not contain useful information for question generation")
        return None

    # Extract the name of the machine
    machine_name = extract_machine_name(document, chat_model)
    if machine_name is None:
        return None

    split_document = split_document_pages(document, input_pages=input_pages)

    questions_simple_filtered = []

    for doc in split_document:
        system = f"<s> [INST] Sei un operaio che lavora su un macchinario e stai generando delle domande le cui risposte si trovano in un documento.\
                Le domande devono riguardare {topics} (non includere domande sul copyright).\
                Hai accesso al seguente documento:\n{doc}\n \
                Le domande devono utilizzare il nome specifico del macchinario: {machine_name}.\n\
                Le domande devono essere in italiano e le risposte devono essere contenute nel documento. [/INST]"


        user = f"Genera quante più domande secondo le istruzioni precedentemente descritte ma non le risposte.\n\
                Se non ci sono informazioni sufficienti per generare coppie di domande e risposte, ritorna una lista vuota.\n\
                Le domande devono utilizzare il nome specifico del macchinario: {machine_name}.\n"
        
        prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])
        chain = prompt | chat_model
        doc_questions_simple = chain.invoke({})

        doc_questions_simple = doc_questions_simple.content
        doc_questions_simple = doc_questions_simple.split('\n')
        if doc_questions_simple is not None:
            doc_questions_simple_filtered = remove_answers(doc_questions_simple)   

        questions_simple_filtered += doc_questions_simple_filtered

    questions_simple_filtered = renumber_questions(questions_simple_filtered)

    return questions_simple_filtered


## Functions for the more involved approach that uses regex for filtering.
def ask_yes_no_questions(document, topics, chat_model):
    """
    Input: 
    - document: a string containing the document's content.
    - topics: a list of strings (in ITALIAN) containing the topics to generate questions about.
                The text should be formulated to be included in the user sentence described below
                'Il documento descrive {topic} di un macchinario specifico.'
    Output: Contiene la risposta a ogni domanda.
    """
    system = f"<s> [INST] Devi analizzare dei documento provenienti da un impianto manifatturiero.\n\
               Devi decidere se il documento contenuto nel contesto contiene un determinato tipo di informazioni.\n\
               Le risposte devono essere in italiano. [/INST]"
    n_topics = len(topics)
    user = f"Rispondi alle {n_topics} domande seguenti con o <sì> o <no> sul documento nel contesto.\n"

    for idx, topic in enumerate(topics):
        user += f"{idx+1}. Il documento contiene informazioni riguardo a {topic} di un macchinario specifico?\n"
    
    user += f"\nIl documento è il seguente:\n{document}.\n\nRispondi solamente con o <sì> o <no>."
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])
    chain = prompt | chat_model
    answer = chain.invoke({})
    return answer.content

def extract_yes_no(answer, n_topics, chat_model):
    """
    Input:
    - answer: The string containing the yes/no answers from the model.
    - n_topics: The number of topics, i.e. the number of questions asked
    Output
    - topic_question: A list of length n_topics, where each entry contains True if the document contains info on that topic and False otherwise.
    """

    yes_no_answers = [None] * n_topics
    

    pattern1 = r"<sì>|<no>|<Sì>|<No>"
    pattern2 = r"[^\w]+(Sì|No|sì|no|yes|Yes)[^\w]+" 
    
    for idx, line in enumerate(answer.split('\n')):

        # Sometimes the model adds additional comments at the end
        if idx >= n_topics-1:
            break
    
        matches1 = re.findall(pattern1, line)
        if len(matches1) > 1:    # If multiple matches on the line, be on the safe side and assume there are no correct ones.
            yes_no_answers[idx] = 'no'
        elif len(matches1) == 1:
            yes_no_answers[idx] = matches1[0][1:-1]
            
        # If we did not find any matches try the other pattern
        if yes_no_answers[idx] is None:
            matches2 = re.findall(pattern2, line)
            if len(matches2) > 1:
                yes_no_answers[idx] = 'no'
            elif len(matches2) == 1:
                yes_no_answers[idx] = ['sì' if matches2[0].lower() in ['sì', 'yes'] else 'no'][0]
        if idx == n_topics: # Sometimes the model adds comments below
            break
    topic_question = [True if x == 'sì' else False for x in yes_no_answers]

    return topic_question


def generate_question_topics(document, topics, topic_question, machine_name, chat_model):
    """
    Input:
    - document: the document currently under investigation.
    - topics: A list containing the topics that the questions should be about.
              The text should be formulated to be included in the user sentence described below
                'Genera domande sul {topic} del macchinario descritto nel documento.'
    - topic_question: A list containing whether the model should generate questions about each topic (T/F).
    - machine_name: The name of the machine described in the document.
    """

    questions = []

    if machine_name is None:
        return None

    # If there are no questions being generated, return None
    if not any(topic_question):
        return None

    system = f"<s>[INST] Devi generare domande su un tipo di informazioni contenute in un documento.\n\
             Le risposte alle domande devono essere presenti nel documento.\n\
             Le domande devono utilizzare il nome specifico del macchinario: {machine_name}.\n\
             Le domande devono essere in italiano.\n\
             Vai a capo dopo ogni domanda.[/INST]"

    for idx, topic in enumerate(topic_question):
        # Only ask if the topic is relevant
        if topic:
            user = f"Genera domande sul {topic} del macchinario descritto nel documento.\n \
             Le domande devono includere il nome specifico del macchinario: {machine_name}.\n \
             Le risposte alle domande devono essere contenute nel documento.\n \
             Il documento è il seguente:\n{document}."
            
            prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])
            chain = prompt | chat_model
            topic_questions = chain.invoke({})

            questions += topic_questions.content.split('\n')
    
    return questions


def extract_questions(document_path, topics, chat_model, input_pages):
    """
    Input:
    - document_path: The path to the document to be analyzed.
    - topics: A list containing the topics that the questions should be about.
              The text should be formulated to be included in the user sentence described below
                'Genera domande sul {topic} del macchinario descritto nel documento.'
                'Il documento descrive {topic} di un macchinario specifico.'
    Output:
    - generated_questions: A list containing the generated questions. If there are some issues or no questions can be generated it returns None.
    - generated_questions_split: A list containing based on what split was each question generated from. (It is a list of numbers needed later for evaluation)
    """
    
    document = prepare_document(document_path)

    n_pages = re.findall(r"page_\d+", document)
    if len(n_pages) < 3:
        # print(f"Document is too short, we assume it does not contain useful information for question generation")
        return None
    
    # Extract the name of the machine
    machine_name = extract_machine_name(document, chat_model)

    split_document = split_document_pages(document, input_pages=input_pages)

    questions_filtered = []
    generated_questions_split = []

    for doc_idx, doc in enumerate(split_document):
        # Extract the answer containing whether the document contains information relevant to those topics
        doc_answer = ask_yes_no_questions(doc, topics, chat_model)
        
        # Extract from the answer the yes/no 
        doc_topic_question = extract_yes_no(doc_answer, n_topics=len(topics), chat_model=chat_model)
        
        # Generate the questions which have to be returned
        doc_generated_questions = generate_question_topics(doc, topics, doc_topic_question, machine_name, chat_model)

        if doc_generated_questions is not None:
            doc_generated_questions_filtered = remove_answers(doc_generated_questions)    
            questions_filtered += doc_generated_questions_filtered  
            generated_questions_split += [doc_idx] * len(doc_generated_questions_filtered)

    questions_filtered = renumber_questions(questions_filtered)

    return questions_filtered, generated_questions_split


