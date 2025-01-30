"""
This script takes as input a set of questions and answers generated from the document, and returns a summary of the scores
  assessing the quality of the QA Generation procedure.

IDEA:
 - Input: Document name
 - Takes the corresponding question file and all the answer files and stores in a .csv file 
     the metrics from lib.evaluate along with all the hyper-parameters.
     Additionally it also stores the percentage of answered questions.

Idea is to have a folder experiments. 
Inside we have a csv file for each document, along with the following columns:
- model_name (mixtral, ...)
- input_pages (number of pages given as input when formulating the questions)
- IR system used (elasticsearch, chromaDB, ...)
- embedding used (if working with vector storages such as chromaDB)
- n_retrieved (number of retrieved chunks when answering)
- chunk_pages (number of pages in each chunk)
- overlap_pages (number of pages to be included in the overlap)

python evaluate
"""
import argparse
import time
import os
import re

from lib.Utilities import setup_chat_model, prepare_document, split_document_pages, extract_retrieved_chunks
from lib.evaluation_functions import evaluate_question_relevance, evaluate_question_global_relevance, evaluate_questions_coverage, evaluate_question_diversity, evaluate_question_overlap
from lib.evaluation_functions import evaluate_answer_groundedness, evaluate_answer_relevance

from lib.Utilities import enlist_answers, filter_answers

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--document_path', type=str, help='The path to the input document that was used to generate the QA pairs.')
    parser.add_argument('--questions_dir', type=str, default='data/questions', help='The directory where the generated questions are stored.')
    parser.add_argument('--answers_dir', type=str, default='data/answers', help='The directory where the generated answers are stored.')
    parser.add_argument('--generator_model', type=str, default='mixtral-8x7b', choices = ['mixtral-8x7b', 'mixtral-8x7b-GGUF'],help='The model used for generating the QA pairs.')

    parser.add_argument('--experiments_dir', type=str, default='data/experiments', help='The directory where the results will be stored.')
    
    parser.add_argument('--evaluator_model', type=str, default='mixtral-8x7b', choices = ['mixtral-8x7b', 'mixtral-8x7b-GGUF'], help='The model used for automatic evaluation of QA pairs.')
    parser.add_argument('--embedding_model_path', type=str, default='data/cc.it.50.bin', help='The name of the fasttext nodel to be used to evaluate diversity.')
    parser.add_argument('--use_groq', type=str, help='Set to True if you want to use the Groq API, False otherwise and model runs locally.')
    parser.add_argument('--groq_api', type=str, help='The Groq API key for the model. Only necessary if use_groq is True.')
    return parser.parse_args()


NULL_ANSWER = 'Non ho abbastanza informazioni per rispondere alla domanda'


def main():
    args = parse_command_line_arguments()
    evaluator_model = setup_chat_model(use_groq=args.use_groq, groq_api=args.groq_api)

    # Extract the document name
    pattern = r'([^/]+)\.json$'
    match = re.search(pattern, args.document_path)
    if match:
        document_name = match.group(1)
        print(f"Document name: {document_name}")
    else:
        raise ValueError('Invalid document path provided.')
    
    # Extract questions.
    questions_path = args.questions_dir + '/' + document_name + '_' + args.generator_model +  '.txt'
    questions_split_path = args.questions_dir + '/' + document_name + '_' + args.generator_model + '_doc_split.txt'
    with open(questions_path, 'r') as f:
        questions = f.readlines()
    with open(questions_split_path, 'r') as f:
        questions_doc_split = f.readlines()
    
    # print(f"Number of questions: {len(questions)}")

    # Extract number of input from the first line
    input_pages = int(questions_doc_split[0])

    # Extract the split used to generate each question
    questions_doc_split = [int(x) for x in questions_doc_split[1:]]
    
    # First check if .csv file exists, if not create it
    csv_file = args.experiments_dir + '/' + document_name + '.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write('model_name,input_pages,IR_system,embedding_model,n_retrieved,chunk_pages,overlap_pages, q_rel, q_grel, q_overlap, q_diversity, q_coverage, answered_percentage, qa_groundedness, qa_rel\n')
    
    # print(f"Created csv file")
    document = prepare_document(args.document_path)
    # print(f"Input pages: {input_pages}")

    doc_split = split_document_pages(document, input_pages)
    print(f"Split document into {len(doc_split)} parts.")    

    # Extract global relevance
    print(f"Extracting global relevance")
    q_grel = evaluate_question_global_relevance(questions, evaluator_model, cot=True)

    # Extract question diversity
    print(f"Extracting question diversity")
    q_diversity = evaluate_question_diversity(questions, args.embedding_model_path)

    # Extract relevance, overlap and coverage over each split
    q_relevance_scores = []
    q_overlap_scores = []
    q_coverage_scores = []
    print(f"Extracting relevance, overlap and coverage")
    for s in list(set(questions_doc_split)):
        qs = [q for idx, q in enumerate(questions) if questions_doc_split[idx] == s] 
        doc = doc_split[s]
        q_relevance_scores += evaluate_question_relevance(doc, qs, evaluator_model, cot=True)
        #print(f"Relevance scores: {q_relevance_scores}")
        q_overlap_scores.append(evaluate_question_overlap(doc, qs, smoothing=True))
        #print(f"Overlap scores: {q_overlap_scores}")
        q_coverage_scores += evaluate_questions_coverage(doc, qs, evaluator_model, cot=True)
        #print(f"Coverage scores: {q_coverage_scores}")
        # Problem with groq rate limit
        time.sleep(60)
        
    q_rel = sum(q_relevance_scores) / len(q_relevance_scores)
    q_overlap = sum(q_overlap_scores) / len(q_overlap_scores)
    q_coverage = sum(q_coverage_scores) / len(q_coverage_scores)


    # Extract the answers (go to answers folder and extract all files related to that)
    answer_files = []
    for file in os.listdir(args.answers_dir):
        if file.startswith(document_name) and not file.endswith('retrieved.txt'):
            answer_files.append(file)
             
    for file in answer_files:
        answer_path = args.answers_dir + '/' + file
        answer_chunks_path = args.answers_dir + '/' + file.replace('.txt', '_chunks_retrieved.txt')
        with open(answer_path, 'r') as f:
            answers = f.readlines()
        with open(answer_chunks_path, 'r') as f:
            answer_chunks = f.readlines()
        
        # Ensure one answer per list element.
        answers = enlist_answers(answers)
        answers = filter_answers(answers)
        print(f"file: {file}")

        answered_questions = sum(1 for x in answers if x != NULL_ANSWER)
        answered_percentage = answered_questions/len(answers)

        # Extract the hyper-parameters from the file name
        pattern = r'([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)\.txt$'
        match = re.search(pattern, file)
        
        if match:
            model_name = match.group(1)
            print(f"Model name: {model_name}")
            ir_system = match.group(2)
            embedding_model = match.group(3)
            n_retrieved = int(match.group(4))
            chunk_pages = int(match.group(5))
            overlap = int(match.group(6))
        else:
            raise ValueError('Invalid file name provided.')
        
        print(f"Model: {model_name}, IR System: {ir_system}, Embedding Model: {embedding_model}, n_retrieved: {n_retrieved}, chunk_pages: {chunk_pages}, overlap: {overlap}")

        # Re-construct chunks. 
        retrieved_chunks = extract_retrieved_chunks(answers, answer_chunks, args.document_path)
            
        # Evaluate the answers:
        print(f"Evaluating answers groundedness")
        qa_groundedness = evaluate_answer_groundedness(questions, answers, retrieved_chunks, evaluator_model, cot=True)

        print(f"Evaluating answers relevance")
        qa_rel = evaluate_answer_relevance(questions, answers, evaluator_model, cot = True)

        # Save the results to the csv file
        with open(csv_file, 'a') as f:
            f.write(f"{model_name},{input_pages},{ir_system},{embedding_model},{n_retrieved},{chunk_pages},{overlap},{q_rel},{q_grel},{q_overlap},{q_diversity},{q_coverage},{answered_percentage},{qa_groundedness},{qa_rel}\n")
         


if __name__ == "__main__":
    main()