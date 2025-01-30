import argparse

from lib.Utilities import save_questions_answers, setup_chat_model
from lib.question_generation_functions import extract_questions, extract_questions_simple

"""
Script to generate questions
Usage with Groq

With simple generation
python generate.py --input_document data/1_batch/2298-A-documentazione-ITA-R2.json --simple True --topics_simple 'il funzionamento della macchina o di alcune sue componenti specifiche, e le possibili manutenzioni' --save False --use_groq True --groq_api <groq_api_key>

python generate.py --input_document data/1_batch/2298-A-documentazione-ITA-R2.json --simple False --topics 'il funzionamento' 'istruzioni per la manutenzione' --save False --use_groq True --groq_api <groq_api_key>
"""


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_document', type=str, required=True, help='The path to the input document')
    parser.add_argument('--simple', type=str, default='False', help='Set to True if you want to extract questions using the simple method, False otherwise.')
    
    parser.add_argument('--topics','--list', nargs='+', help='The topics to extract questions about when simple is False. Should be a list of strings where each entry is inserted in: Genera domande sul <topic> del macchinario descritto nel documento.')
    parser.add_argument('--topics_simple', type=str, help='The topics to extract questions about when simple is True. Should be phrased as a sentence to insert in: Le domande devono riguardare <topics>.')

    parser.add_argument('--input_pages', type=int, default=10, help='The number of pages to include at a time as context for question generation. Defaults to 10.')

    parser.add_argument('--save', type=str, default='False', help='Set to True if you want to save the questions to file, otherwise they will simply be printed.')
    
    parser.add_argument('--use_groq', type=str, help='Set to True if you want to use the Groq model, False otherwise and model runs locally.')
    parser.add_argument('--groq_api', type=str, help='The groq api key for the model. Only necessary if use_groq is True.')

    parser.add_argument('--use_gpt', type=str, default='False', help='Set to True if you want to use the GPT3.5 model, False otherwise and Mixtral will be used.')
    parser.add_argument('--openai_key', type=str, help='The openai api key for the model. Only necessary if use_gpt3.5 is True.')
    return parser.parse_args()




def main():
    args = parse_command_line_arguments()

    if args.use_gpt == 'True':
        model_name = 'gpt-3.5'
    elif args.use_groq == 'True':
        model_name = 'mixtral-8x7b'
    else:
        model_name = 'mixtral-8x7b-GGUF' # quantized version

    chat_model = setup_chat_model(args.use_groq, args.groq_api, args.use_gpt, args.openai_key)
        
    # Extract the questions
    if args.simple == 'True':
        questions, questions_doc_split = extract_questions_simple(args.input_document, args.topics_simple, chat_model, args.input_pages)
    else:
        questions, questions_doc_split = extract_questions(args.input_document, args.topics, chat_model, args.input_pages)

    if args.save == 'True':
        if args.simple == 'True':
            save_questions_answers(questions, args.input_document, model_name, save_to_questions = True, simple=True, input_pages=args.input_pages, split_documents_vector=questions_doc_split)
        else:
            save_questions_answers(questions, args.input_document, model_name, save_to_questions = True, simple=False, input_pages=args.input_pages, split_documents_vector=questions_doc_split)
    else:
        print(questions)

if __name__ == "__main__":
    main()