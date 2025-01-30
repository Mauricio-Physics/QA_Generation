import argparse 

from lib.Utilities import setup_chat_model, extract_document_name, get_questions_from_file, save_questions_answers
from lib.answer_generation_functions import extract_answers_rag

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--document_path', type=str, help='The path to the input document.')
    parser.add_argument('--question_dir', type=str, default='data/questions', help='The directory where the generated questions are stored.')
    parser.add_argument('--simple', type=str, default='False', help='Set to True if the questions were created using the simple method, False otherwise.')

    parser.add_argument('--n_retrieved', type=int, default=5, help='The number of documents to retrieve to answer each question.')
    parser.add_argument('--chunk_pages', type=int, default=3, help='The number of pages to include in each chunk for the RAG system.')
    parser.add_argument('--overlap_pages', type=int, default=1, help='The number of pages to overlap between different chunks.')

    parser.add_argument('--ir_system', type=str, default='chromadb', choices=['chromadb'], help='The IR system to be used for retrieving the documents.')
    parser.add_argument('--embedding_model_name', type=str, default='intfloat/multilingual-e5-base', help='The name of the Huggingface sentence transformer model to be used for embedding the documents with vector storages.')

    parser.add_argument('--save', type=str, default='False', help='Set to True if you want to save the answers to file, otherwise they will simply be printed.')

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
    
    # Setup chat model    
    chat_model = setup_chat_model(args.use_groq, args.groq_api, args.use_gpt, args.openai_key)

    # Extract the questions into a list of strings
    file_name = extract_document_name(args.document_path)
    if args.simple == 'True':
        questions_path = args.question_dir + '_simple/' + file_name + '_' + model_name + '.txt'
    else:
        questions_path = args.question_dir + '/' + file_name + '_' + model_name + '.txt'

    # Get the answers
    answers, retrieved_chunks_initial_pages = extract_answers_rag(questions_path, args.document_path, args.n_retrieved, args.chunk_pages, args.overlap_pages, chat_model, args.ir_system, args.embedding_model_name)
    answers_formatted = [f"{idx+1}. {x.content}" for idx, x in enumerate(answers)]
    if args.save == 'True':
        if args.simple == 'True':
            save_questions_answers(answers_formatted, args.document_path, model_name, save_to_questions = False, simple=True, retrieved_chunks_vec=retrieved_chunks_initial_pages, n_retrieved=args.n_retrieved, chunk_pages=args.chunk_pages, overlap_pages=args.overlap_pages, ir_system=args.ir_system, embedding_model=args.embedding_model_name)
        else:
            save_questions_answers(answers_formatted, args.document_path, model_name, save_to_questions = False, simple=False, retrieved_chunks_vec=retrieved_chunks_initial_pages, n_retrieved=args.n_retrieved, chunk_pages=args.chunk_pages, overlap_pages=args.overlap_pages, ir_system=args.ir_system, embedding_model=args.embedding_model_name)
    else:
        print(answers_formatted)

if __name__ == "__main__":
    main()
