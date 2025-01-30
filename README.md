# QA_Generation
This repository contains scripts for generating both questions and answers from a given text corpus. It provides tools for generating questions based on the content of the text as well as generating answers to those questions.\

The ChatModel used is [`TheBloke/Mixtral-8x7B-v0.1-GGUF`](https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF); the quantized version of `mixtral-8x7b`. 

## Files and Directory Structure

- `generate_questions.py`: This script generates questions based on the content of a given document.
- `generate_answers.py`: This script is responsible for generating answers to questions based on a given document.
- `lib/`: This directory contains library files utilized by the main scripts.
  - `Utilities.py`: Provides utility functions used by other scripts.
  - `answer_generation_functions.py`: Functions related to generating answers. 
  - `evaluate_functions.py`: Contains the functions to evaluate the QA pairs based on Balaguer et al. (2024)[1]. 
  - `question_generation_functions.py`: Functions related to generating questions.
- `data`
    - `questions`: This folder contains the generated questions using the more involved method. It also contains information about the part of document used to generate each question (needed for evaluation).
    - `questions_simple`: Same as above, but generated with the simple method.
    - `answers`: This folder contains the generated answers to questions generated using the more involved method. It also contains information about the retrieved chunks of documents to answer each question.
    - `answers_simple`: Same as above, but for questions generated with the simple method (needed for evaluation).

## Usage
> [!IMPORTANT]
> The executables expect input documents as json files where the keys are the document pages and the values are the parsed content.

### Generating Questions
To generate questions, execute the `generate_questions.py` script and provide the document as input along with some topics that the questions should be about. This script utilizes functions from `questions_generation_functions.py` to generate questions.\
Two question generating pipelines are implemented. A naive one (used by setting parameter `--simple` True) and a more involved one which ensures the document contains information relevant to the topics provided before generating questions.
```bash
python generate_questions.py --input_document <input_document_path> --topics <topic1> <topic2> ... --input_pages <input_pages> 
```
where:
- `input_document`: path to the document (in json format) to generate questions from.
- `topics`: list of topics about which the questions should be about.
- `input_pages`: number of pages considered at a time for question generation.  (possible parameter to tune)
Additional information and optional parameters can be found in the associated script.

### Generating Answers
To generate answers, execute the `generate_answers.py` script and provide the document that was used to generate the questions. 
```bash
python generate_answers.py --input_document <input_document_path>  --n_retrieved <n_retrieved> --chunk_pages <chunk_pages> --overlap_pages <overlap_pages>
```
where:
- `input_document`: path to the document (in json format) that the questions are generated from.
- `n_retrieved`: the number of retrieved chunks in the RAG pipeline to answer the question. (possible parameter to tune)
- `chunk_pages`: the number of pages that should be placed in each chunk which is then indexed for RAG. (possible parameter to tune)
- `overlap_pages`: number of overlapping pages between different chunks (possible parameter to tune).
Additional optional parameters are described in the associated script.

### Evaluation of questions and answers
The automatic evaluation techinques for questions and answers are implemented based on those proposed by Balaguer et al. (2024) [1]. Refer to the aforementioned paper for further explanation.
> [!IMPORTANT]
> For evaluation of questions diversity a pre-trained fasttext embedding model is employed. An italian version (which I already reduced to work with embeddings of size 50) can be downloaded from the following [link](https://drive.google.com/file/d/10lAGiPArsw4jc05TwPTtxsFwLvGwGy6K/view?usp=sharing). 

### TODO
Write python script to run all evaluation metrics and save with corresponding hyper-parameters to a .csv file to keep track.

## References
[1] Balaguer, A., Benara, V., Cunha, R. L. F., Estevão Filho, R. de M., Hendry, T., Holstein, D., Marsman, J., Mecklenburg, N., Malvar, S., Nunes, L. O., Padilha, R., Sharp, M., Silva, B., Sharma, S., Aski, V., & Chandra, R. (2024). RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture. In *arXiv preprint arXiv:2401.08406*.
