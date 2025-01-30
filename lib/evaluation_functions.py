# Functions for evaluation of questions and answers.
import gensim
import re
import time

from collections import Counter
from scipy.stats import entropy

from langchain_core.prompts import ChatPromptTemplate
from lib.Utilities import preprocess

from tqdm import tqdm


def evaluate_question_relevance(document, questions, chat_model_evaluation, cot):
    """
    Input:
    - document: the part of document from which the questions were generated.
    - questions: A list of questions to be evaluated.
    - cot: A boolean indicating whether to use the chain of thought prompt.
    Output:
    - relevance_scores: A list of scores from 1 to 5, where 5 is the most relevant and 1 is the least relevant.

    The goal of this function is to evaluate each question with a relevance score (wrt to the document) from 1 to 5, where 5 is the most relevant and 1 is the least relevant.
    To do so, a LLM is used. If cot is set to True, an additional chain of thought prompt is passed in the pipeline.
    """
    system = f"<s>[INST] Il tuo ruolo è valutare quanto è plausibile che un addetto macchine possa avere la necessità di conoscere la risposta a una domanda.\n\
        L'operaio cerca la risposta alla domanda su un documento che verrà fornito.\n\
        Assegna un punteggio da 1 a 5, dove 5 rappresenta una domanda rilevante e 1 una domanda non pertinente, tenendo conto del contesto fornito dato dal documento. [/INST]"

    if cot:
        user_cot = f"Quali caratteristiche deve avere una domanda rispetto al contesto del documento per avere un punteggio di 5?"
        prompt_cot = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot)])
        chain_cot = prompt_cot | chat_model_evaluation
        risposta_cot = chain_cot.invoke({}).content
        
    relevance_scores = [None] * len(questions)
    user = f"Valuta se è plausibile che un addetto macchine cerchi le seguenti domande sul documento.\n\n\
        Documento: {document}\n\n\
        Domande: {questions}\n\n\
        Assegna un punteggio da 1 a 5 a ogni domanda.\n\
        Formatta in modo che su ogni riga ci sia il punteggio della domanda corrispondente.\n\
        Il punteggio deve essere inserito tra parentesi quadre."
   
    if cot:
        prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot), ('assistant', risposta_cot), ('user', user)])
    else:
        prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])
    
    chain = prompt | chat_model_evaluation
    chain_output = chain.invoke({}).content

    relevance_scores = [int(x) for x in re.findall(r'\[(\d+)\]', chain_output)]        
    return relevance_scores


def evaluate_question_global_relevance(questions, chat_model_evaluation, cot, show_progress_bar=True):
    """
    Input:
    - questions: A list of questions to be evaluated.
    - cot: A boolean indicating whether to use the chain of thought prompt.
    Output:
    - global_relevance_scores: A list of scores from 1 to 5, where 5 is the most relevant and 1 is the least relevant.
    
    The goal of this function is to evaluate each question with a score from 1 to 5, where 5 is the most relevant and 1 is the least relevant.
    To do so, a LLM is used. If cot is set to True, an additional chain of thought prompt is passed in the pipeline.
    """
    if show_progress_bar:
        progress_bar = tqdm(total=len(questions), desc='Evaluated questions', unit=' questions')
    
    system = f"<s>[INST] Il tuo ruolo è valutare quanto è plausibile che un addetto macchine possa avere la necessità di conoscere la risposta a una domanda.\n\
        Assegna un punteggio da 1 a 5, dove 5 rappresenta una domanda rilevante e 1 una domanda non pertinente [/INST]"
        
    # Replace to avoid the model printing also the question number and hence the score being extracted as the question number
    questions = [re.sub(r'questions \d+\.', '', x).strip() for x in questions] 
    if cot:
        user_cot = f"Quali caratteristiche deve avere una domanda per avere un punteggio di 5?"
        prompt_cot = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot)])
        chain_cot = prompt_cot | chat_model_evaluation
        risposta_cot = chain_cot.invoke({}).content
        
    global_relevance_scores = [None] * len(questions)
    
    for idx, q in enumerate(questions):
        user = f"Valuta la seguente domanda:\n{q}\n\
            Ritorna solo un numero da 1 a 5."
        if cot:
            prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot), ('assistant', risposta_cot), ('user', user)])
        else:
            prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])
        
        chain = prompt | chat_model_evaluation
        chain_content = chain.invoke({}).content

        score = re.search(r'\d+', chain_content)
        global_relevance_scores[idx] = score.group(0)
        if show_progress_bar:
            progress_bar.update(1)
    progress_bar.close()
    return global_relevance_scores

def compute_kl_divergence(x, y, smoothing):
    """
    Takes as input word counters of x and y and returns the KL divergence between the respective word distributions
    If smoothing is True, it applies Laplace smoothing to the word distributions
    """
    all_words = set(list(x) + list(y))  # Extract all unique words.
    x_freq = [None] * len(all_words)
    y_freq = [None] * len(all_words)
    x_tot = x.total()
    y_tot = y.total()

    if smoothing:
        for idx, word in enumerate(all_words):
            x_freq[idx] = (x[word] + 1) / (x_tot + len(all_words))
            y_freq[idx] = (y[word] + 1) / (y_tot + len(all_words))
    else:
        for idx, word in enumerate(all_words):
            x_freq[idx] = x[word] / x_tot
            y_freq[idx] = y[word] / y_tot
    
    # Compute KL divergence
    return entropy(x_freq, y_freq)
    
def evaluate_question_overlap(document, questions, smoothing):
    """
    Given an input document and a set of questions, it evaluates the KL divergence between the corresponding word probability distributions.
    A lower value indicates higher overlap and is hence desireable.
    Standard pre-processing steps are applied (removal of punctuation, numbers, stopwords, lowercasing)
    If smoothing is True, (Laplace) smoothing is applied to compute the word distributions.
    """
    # Preprocess document and questions.
    questions_processed = preprocess(questions)
    document_processed = preprocess(document)

    # Counter of words
    questions_processed_cnt = Counter(questions_processed)
    document_processed_cnt = Counter(document_processed)

    return compute_kl_divergence(document_processed_cnt, questions_processed_cnt, smoothing)


def evaluate_question_diversity(questions, word_embedding_model_path):
    """
    Given a set of questions, it computes the diversity between those as the average Word Mover's distance (WMD) between them. A higher value is desireable
    The questions should already be pre-processed to be a list of lists of words.
    The word embedding model is assumed to be of the fasttext family.
    """
    word_embedding_model = gensim.models.fasttext.load_facebook_model(word_embedding_model_path)
    total = 0
    for idx1, q1 in enumerate(questions):
        for idx2, q2 in enumerate(questions[idx1+1:]):
            idx2 += idx1 + 1
            total += word_embedding_model.wv.wmdistance(q1, q2)
    total /= len(questions) * (len(questions) - 1) / 2
    return total


def evaluate_questions_coverage(document, questions, chat_model_evaluation, cot):
    """
    Given a document and a list of questions, it evaluates the coverage of the questions, i.e. whether their answers can be found in the associated documents.
    The evaluation is done using a chat model, that assigns a score from 1 to 5, with 1 meaning the answer can not be formulated based on the document, while 5 means that a nice answer can be extracted from the document.
    If cot is True, an additional chain of thought prompt is passed in the pipeline.
    """
    system = f"<s>[INST] Il tuo ruolo è valutare se la risposta a una domanda può essere formulata con le informazioni fornite nel contesto.\n\
        Assegna un punteggio da 1 a 5 a ogni domanda, dove 5 rappresenta una domanda la cui risposta è contenuta nel contesto e 1 una domanda alla quale non si può rispondere con le informazioni del documento.[/INST]"
        
    if cot:
        user_cot = f"Quali caratteristiche deve avere una domanda per avere un punteggio di 5?"
        prompt_cot = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot)])
        chain_cot = prompt_cot | chat_model_evaluation
        risposta_cot = chain_cot.invoke({}).content

    user = f"Assegna un punteggio da 1 a 5 a ogni domanda, dove 5 rappresenta una domanda la cui risposta è contenuta nel contesto e 1 una domanda alla quale non si può rispondere con le informazioni del documento.\n\
        Documento: {document}\n\n\
        Domande: {questions}\n\n\
        Assegna un punteggio da 1 a 5 ad ogni domanda.\n\
        Formatta in modo che su ogni riga ci sia il punteggio della domanda corrispondente.\n\
        Il punteggio deve essere inserito tra parentesi quadre."
    
    if cot:
        prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot), ('assistant', risposta_cot), ('user', user)])
    else:
        prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])
    
    chain = prompt | chat_model_evaluation
    chain_output = chain.invoke({}).content

    coverage_scores = [int(x) for x in re.findall(r'\[(\d+)\]', chain_output)]     
    return coverage_scores
    
####### ANSWER EVALUATION ####### 

def evaluate_answer_relevance(questions, answers, chat_model_evaluation, cot, show_progress_bar=True):
    system = f"<s>[INST] Il tuo compito è valutare se la risposta soddisfa le richieste poste dalla domanda in modo adeguato.\n\
        Assegna un punteggio da 1 a 5, dove 5 rappresenta una risposta rilevante alla domanda e 1 una risposta non pertinente. [/INST]"
    
    if show_progress_bar:
        progress_bar = tqdm(total=len(questions), desc='Evaluated QA Pairs', unit=' QAPairs')
    
    if cot:
        user_cot = f"Quali caratteristiche deve avere una risposta per avere un punteggio di 5 rispetto a una domanda?"
        prompt_cot = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot)])
        chain_cot = prompt_cot | chat_model_evaluation
        risposta_cot = chain_cot.invoke({}).content

    relevance_scores = []
    for q, a in zip(questions, answers):
        if a == 'Non ho abbastanza informazioni per rispondere alla domanda':
            relevance_scores.append(None)
            next
   
        user = f"Valuta se la risposta alla seguente domanda è soddisfacente:\n\n\
            La domanda è\n{q}\n\n\
            La risposta è:\n{a}\n\n\
            Assegna un punteggio da 1 a 5.\n\
            Il punteggio deve essere inserito tra parentesi quadre."
        if cot:
            prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot), ('assistant', risposta_cot), ('user', user)])
        else:
            prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])
        
        chain = prompt | chat_model_evaluation
        chain_content = chain.invoke({}).content
        score = re.findall(r'\[(\d+)\]', chain_content)

        if score:
            relevance_scores.append(int(score[0]))
        else:
            relevance_scores.append(None)
        
        if show_progress_bar:
            progress_bar.update(1)
    
    if show_progress_bar:
        progress_bar.close()
    return relevance_scores
    
def evaluate_answer_groundedness(questions, answers, retrieved_chunks, chat_model_evaluation, cot, show_progress_bar=True):
    system = f"<s>[INST] Il tuo compito è valutare se la risposta segue in maniera logica alla domanda e rispetto al nel contesto.\n\
        Assegna un punteggio da 1 a 5, dove 5 rappresenta una risposta rilevante alla domanda e al contesto e 1 una risposta non pertinente.[/INST]"
    
    if show_progress_bar:
        progress_bar = tqdm(total=len(questions), desc='Evaluated QA Pairs', unit=' QAPairs')

    if cot:
        user_cot = f"Quali caratteristiche deve avere una risposta per avere un punteggio di 5 rispetto a una domanda e al contesto?"
        prompt_cot = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot)])
        chain_cot = prompt_cot | chat_model_evaluation
        risposta_cot = chain_cot.invoke({}).content

    groundedness_score = []

    idx = 0
    for q, a, c in zip(questions, answers, retrieved_chunks):
        # Problems with groq rate limit
        
        time.sleep(60)
        idx += 1
        user = f"Valuta se la risposta alla seguente domanda è soddisfacente e coerente con il documento:\n\n\
            Il contesto è:\n{c}\n\n\
            La domanda è:\n{q}\n\n\
            La risposta è:\n{a}\n\n\
            Assegna un punteggio da 1 a 5 ad ogni domanda.\n\
        Il punteggio deve essere inserito tra parentesi quadre."
        if cot:
            prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot), ('assistant', risposta_cot), ('user', user)])
        else:
            prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])
        chain = prompt | chat_model_evaluation
        chain_content = chain.invoke({}).content

        score = re.findall(r'\[(\d+)\]', chain_content)
        if score:
            groundedness_score.append(int(score[0]))
        else:
            groundedness_score.append(None)

        if show_progress_bar:
            progress_bar.update(1)
    if show_progress_bar:
        progress_bar.close()
    return groundedness_score