import argparse
import json

from unstructured.partition.pdf import partition_pdf
from collections import Counter

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--document_path', type=str, help='The path to the input PDF document.')
    parser.add_argument('--save_path', type=str, help='The path where to print the parsed json document.')

    return parser.parse_args()

def prepare_counter(elements):
    # Extract Titles and text and setup the preprocess
    titles_text = [el.text for el in elements if el.category == "Title"]
    text_text = [el.text for el in elements if el.category == "Text"]
    narrtext_text = [el.text for el in elements if el.category == "NarrativeText"]

    all_text = titles_text + narrtext_text + text_text
    all_counter = Counter(all_text)

    return all_counter

def create_parsed_dictionary(elements, all_counter):
    parsed_doc = {'page_1': ''}
    curr_page = 'page_1'
    page_number = 1

    for el in elements:
        # New page -> Create new entry in the dictionary.
        if el.category == 'PageBreak':
            page_number += 1
            curr_page = f"page_{page_number}"
            parsed_doc[curr_page] = ''

        elif el.category == 'ListItem':
            if all_counter[el.text] < 10 and all_counter[el.text] != 0:
                parsed_doc[curr_page] += el.text
                parsed_doc[curr_page] += '\n'
        
        elif el.category == 'NarrativeText':
            if all_counter[el.text] < 10 and all_counter[el.text] != 0:
                parsed_doc[curr_page] += el.text
                parsed_doc[curr_page] += ' '

        elif el.category == 'Table':
            parsed_doc[curr_page] += '\n'
            parsed_doc[curr_page] += el.metadata.text_as_html
            parsed_doc[curr_page] += '\n'

        elif el.category == 'Text':
            if all_counter[el.text] < 10 and all_counter[el.text] != 0:
                parsed_doc[curr_page] += el.text
                parsed_doc[curr_page] += ' '

        elif el.category == 'Title':
            # Some headers appear to be read in as titles, so we filter them to remove them
            # The page_number > 1 is to avoid removing machine name information if present.
            if page_number == 1:
                parsed_doc[curr_page] += el.text
                parsed_doc[curr_page] += '\n'
            elif all_counter[el.text] < 10 and all_counter[el.text] != 0:
                parsed_doc[curr_page] += el.text
                parsed_doc[curr_page] += '\n'
        
    # Remove all empty entries.
    parsed_doc = {k:v for k,v in parsed_doc.items() if v}   

    return parsed_doc

def save_parsed_document(parsed_doc, save_path):
    """
    Saves the parsed document to the specified path
    """
    with open(save_path, 'w') as f:
        json.dump(parsed_doc, f)




def main():
    args = parse_command_line_arguments()
    elements = partition_pdf(filename=args.document_path, infer_table_structure=True, include_page_breaks=True, languages=['ita', 'eng'])
    all_counter = prepare_counter(elements=elements)

    parsed_doc = create_parsed_dictionary(elements, all_counter)

    save_path = args.document_path.replace('.pdf', '.json')
    save_parsed_document(parsed_doc, save_path)

if __name__ == "__main__":
    main()
