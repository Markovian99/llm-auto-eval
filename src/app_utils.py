import pandas as pd
import streamlit as st
import os
import json
import csv
import numpy as np
import time
import re

#sklearn cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import ConversationalRetrievalChain, LLMChain

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import PromptTemplate

from config import (MODELS, TEMPERATURE, MAX_TOKENS, NO_RETRIEVER_NAME, EMBEDDING_MODELS, ANSWER_EVAL_MODEL, PARSE_METHOD, PROCESSED_DOCUMENTS_DIR,  USE_AZURE,
            MIN_QUESTION_LENGTH, MIN_ANSWER_SIMILARITY, MIN_QUESTION_SIMILARITY, NO_RESPONSE_EXCEPTION, BAD_FORMAT_EXCEPTION, BAD_QUESTION_EXCEPTION, BAD_ANSWER_SIMILARITY,
            SUMMARY_QA_GEN, BASIC_QA_GEN, COMPLEX_QA_GEN, MULTIHOP_QA_GEN, BAD_CONTEXT)
from genai_utils import token_split_docs, generate_response, generate_langchain_response, generate_subqueries, create_knowledge_base, generate_kb_response


import openai
openai.api_key = os.getenv('OPENAI_API_KEY')
if os.getenv('OPENAI_API_BASE'):
    openai.api_base = os.getenv('OPENAI_API_BASE')
if os.getenv('OPENAI_API_TYPE'):
    openai.api_type = os.getenv('OPENAI_API_TYPE')
if os.getenv('OPENAI_API_VERSION'):
    openai.api_version = os.getenv('OPENAI_API_VERSION')

#question	jeopardy_answer	kb-gpt35_answer	kb-gpt4_answer	kb-40b_answer	kb-llama2-13b_answer	kb-llama2-13b_templated_answer	kb-llama2-70b_answer (4 bit)	kb-llama2-70b_answer (8 bit)

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


def initialize_session_state():
    """ Initialise all session state variables with defaults """
    SESSION_DEFAULTS = {
        "cleared_responses" : False,
        "generated_responses" : False,
        "chat_history": [],
        "uploaded_file": None,
        "generation_models": [MODELS[0]],
        "embedding_model": ANSWER_EVAL_MODEL,
        "parse_method": PARSE_METHOD[0],
        "experiment_dir": PROCESSED_DOCUMENTS_DIR,
        "experiment_name": "",
        "general_context": "",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [],
        "ran_rag_eval": False
    }

    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


# get embedding for one sentence
def calc_similarity(vec1,vec2):
    try:
        # could change to cosine_similarity
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except Exception as e:
        print(e)
        return 0

#delete files from a directory and subdirectories, but keep the subdirectories
def delete_files_from_dir(dir, keep_files=[".gitignore"]):
    for f in os.listdir(dir):
        if f not in keep_files:
            path = os.path.join(dir, f)
            if os.path.isdir(path):
                delete_files_from_dir(path,keep_files)
            else:
                os.remove(path)

def get_experiment_category(qa_file):
    if not qa_file.lower().endswith("_qa.csv"):
        return "N/A"
    elif qa_file.lower().find("summary")>-1:
        return "Summary"
    elif qa_file.lower().find("multihop")>-1 or qa_file.lower().find("compare")>-1:
        return "Multi-hop"
    elif qa_file.lower().find("complex")>-1 or qa_file.lower().find("basic")>-1  or qa_file.lower().find("granular")>-1:
        return "Vanilla RAG"
    else:
        return "All Doc Questions"

def check_question_answer_and_split(response):
    if response==NO_RESPONSE_EXCEPTION:
        return response, response
    else:
        #split response into question and answer
        try:
            if response.find("Answer:")>-1:
                question, answer = response.split("Answer:")
            elif response.find("Answer :")>-1:
                print(response)
                question, answer = response.split("Answer :")
            else:
                print(response)
                question, answer = response.split("answer:")
            
            if question.find("Question:")>-1:
                question = question.split("Question:")[1].strip()
            
            #remove any whitespace
            question = question.strip()
            answer = answer.strip()
            return question, answer
        except Exception as e:
            print(response)
            print(e)
            return BAD_FORMAT_EXCEPTION, response

def run_jeopardy(file_path,column_name='page_content',template=BASIC_QA_GEN, max_subsample=None, suffix="_qa", full_response=False):
    """ Function to run jeopardy on a csv file with a column of text
    Args:
        file_path (str): path to csv file
        column_name (str): name of column with text
        mod_reducer (int): number to reduce the number of rows to generate questions for
    Returns:
        None - writes new csv file ending in _qa
    """
    df = pd.read_csv(file_path)

    df['question'] = ""
    df['answer'] = ""

    #randomly sample rows and always keep the last row
    if max_subsample and len(df)>max_subsample+1:
        df = pd.concat([df.sample(max_subsample),df.tail(1)])

    embedding_function = HuggingFaceEmbeddings(
        model_name=st.session_state["embedding_model"],
        cache_folder="../models/sentencetransformers"
    )  

    df_list = []
    for model in st.session_state["generation_models"]:
        df_new = df.copy()
        #loop over dataframe and generate questions
        for index, row in df.iterrows():
            context = row[column_name]
            system_text = "You are a helpful assistant that, when generating questions, DOES NOT refer to the context and DOES NOT use 'document', 'context', or any similar words."
            template = template
            temperature = st.session_state["temperature"]
            if not context or len(context)<MIN_QUESTION_LENGTH:
                df_new.loc[index,'question'] = BAD_CONTEXT
                df_new.loc[index,'answer'] = BAD_CONTEXT
                continue
            if template:
                prompt = template.format(context=context)
            else:
                prompt = context
            print(prompt)
            response = generate_response(prompt, model, system_prompt=system_text, temperature=temperature)
            question, answer = check_question_answer_and_split(response)
            df_new.loc[index,'question'] = question
            df_new.loc[index,'answer'] = answer
            time.sleep(1)
            # split context into chunks of 128 tokens with 64 token overlap
            context_chunks = token_split_docs(context, chunk_size=128, overlap_size=64)
            # check if context is has similarity to question
            max_similarity = np.max(cosine_similarity(embedding_function.embed_documents(context_chunks),embedding_function.embed_documents([question])))

            # df.loc[index,'max_similarity'] = max_similarity
            if not str(df_new.loc[index,'question']).startswith("Exception") and max_similarity<MIN_QUESTION_SIMILARITY:
                    df_new.loc[index,'answer'] = BAD_QUESTION_EXCEPTION + " -> " + df_new.loc[index,'answer']
                    df_new.loc[index,'answer_full'] = BAD_QUESTION_EXCEPTION
            elif full_response and not str(df_new.loc[index,'question']).startswith("Exception") and  str(df_new.loc[index,'question'])!="" and  str(df_new.loc[index,'question'])!="nan":
                prompt = f"Answer based on the following context:\n{context}\n\nQuestion: {question}"
                response = generate_response(prompt, model, system_prompt="You are a helpful assistant that only uses the provided context to answer questions.", temperature=temperature)
                df_new.loc[index,'answer_full'] = response
                answer_sim = calc_similarity(embedding_function.embed_documents([answer])[0],embedding_function.embed_documents([response])[0])
                if answer_sim <MIN_ANSWER_SIMILARITY:
                    df_new.loc[index,'answer'] = BAD_ANSWER_SIMILARITY + " -> " + df_new.loc[index,'answer']
                    df_new.loc[index,'answer_full'] = BAD_ANSWER_SIMILARITY + " -> " + df_new.loc[index,'answer']

            df_new['model'] = model
            time.sleep(1)
        df_list.append(df_new)
    df_new = pd.concat(df_list)
            
    #create a new filepath with _qa.csv at the end
    new_file_path = file_path[:-4] + suffix +".csv"
    df_new.to_csv(new_file_path, index=False)



def run_rag(model_name, emb_model_name, file_path, question_column, system_prompt="", template=None, faiss_dir="../data/faiss-db", temperature=0, k=4):

    df_input = pd.read_csv(file_path)

    print(f"Running RAG with {len(df_input)} questions from file {file_path}")

    df_input['source_documents'] = None
    
    if emb_model_name==NO_RETRIEVER_NAME:
        embedding_function = None
        db = None
        retriever = None
    else:
        embedding_function = HuggingFaceEmbeddings(
            model_name=emb_model_name,
            # model_kwargs={'device': 'cuda'},
            # encode_kwargs={'normalize_embeddings': True}
            cache_folder="../models/sentencetransformers"
        )
        db = FAISS.load_local(faiss_dir, embedding_function)
        retriever = VectorStoreRetriever(vectorstore=db, search_kwargs={"k": k})

    for idx, row in df_input.iterrows():
        prompt = str(row[question_column])
        if prompt.startswith("Exception:"):
            df_input.loc[idx,'rag_response'] = 'Exception: Invalid question.'
        elif prompt!="" and prompt!="nan":
            data_dict = {}
            data_dict['prompt'] = prompt
            data_dict['chat_history'] = []

            # response = chain(inputs={"question":data_dict['prompt'], "chat_history":data_dict['chat_history']})
            if emb_model_name==NO_RETRIEVER_NAME:
                if template:
                    prompt = template.format(prompt=data_dict['prompt'])
                response = generate_response(prompt, model_name, system_prompt=system_prompt, temperature=temperature)
                df_input.loc[idx,'rag_response'] = response
                df_input.loc[idx,'source_documents'] = "Exception: No retriever used."
            else:
                response = generate_kb_response(data_dict['prompt'], model_name, system_prompt=system_prompt, template=template,temperature=temperature, include_source=False, retriever = retriever)
                df_input.loc[idx,'rag_response'] = response['answer']
                df_input.loc[idx,'source_documents'] = str(response['source_documents'])
        else:
            print('Exception: No question provided.')
            df_input.loc[idx,'rag_response'] = 'Exception: No question provided.'
    return df_input


def run_rag_multihop(model_name, emb_model_name, file_path, question_column, faiss_dir="../data/faiss-db", temperature=0, k=4):

    df_input = pd.read_csv(file_path)
    model=model_name[8:]

    print(f"Running RAG with {len(df_input)} questions from file {file_path}")

    # Will download the model the first time it runs
    # embedding_function = SentenceTransformerEmbeddings(
    #     model_name=emb_model_name,
    #     cache_folder="../models/sentencetransformers",
    # )
    embedding_function = HuggingFaceEmbeddings(
        model_name=emb_model_name,
        # model_kwargs={'device': 'cuda'},
        # encode_kwargs={'normalize_embeddings': True}
        cache_folder="../models/sentencetransformers"
    )

    df_input['source_documents'] = None

    if model_name.startswith("OpenAI: "):
        llm = ChatOpenAI(model=model, temperature=temperature)
    else:
        df_input['rag_response'] = 'Exception: Please select an OpenAI model.'
        return df_input

    db = FAISS.load_local(faiss_dir, embedding_function)

    retriever = VectorStoreRetriever(vectorstore=db, search_kwargs={"k": k})
    chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever,return_source_documents=True) #, return_source_documents=True

    prompt_template = """Use the context below to answer the question below:
    Context: {context}

    Question: {prompt}
    Answer:"""
    multi_retriever = VectorStoreRetriever(vectorstore=db, search_kwargs={"k": max(1,int(k/2))})
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "prompt"])
    multi_chain = LLMChain(llm=llm, prompt=PROMPT)

    for idx, row in df_input.iterrows():
        prompt = str(row[question_column])
        if prompt.startswith("Exception:"):
            df_input.loc[idx,'rag_response'] = 'Exception: Invalid question.'
        elif prompt!="" and prompt!="nan":
            data_dict = {}
            data_dict['prompt'] = prompt
            data_dict['chat_history'] = []

            queries = []
            queries = generate_subqueries(prompt, model_name[8:])

            if len(queries) == 0:
                response = chain(inputs={"question":data_dict['prompt'], "chat_history":data_dict['chat_history']})
                df_input.loc[idx,'rag_response'] = response['answer']
                df_input.loc[idx,'source_documents'] = str(response['source_documents'])
            else:
                source_documents=[]
                #append original question to queries
                queries.append(prompt)
                print(f"Queries: {queries}")
                for query in queries:
                    source_documents.extend(multi_retriever.get_relevant_documents(query))
                print(f"Queries {len(queries)}, # docs {len(source_documents)}")

                # function used in sorted that sorts by source, section, doc_idx, start_index if they exists
                def document_sort_key(doc):
                    if 'source' in doc.metadata.keys() and 'section' in doc.metadata.keys() and 'doc_idx' in doc.metadata.keys() and 'start_index' in doc.metadata.keys():
                        return (doc.metadata['source'], doc.metadata['section'], doc.metadata['doc_idx'], doc.metadata['start_index'])
                    elif 'source' in doc.metadata.keys() and 'section' in doc.metadata.keys() and 'doc_idx' in doc.metadata.keys():
                        return (doc.metadata['source'], doc.metadata['section'], doc.metadata['doc_idx'], -1)
                    elif 'source' in doc.metadata.keys() and 'section' in doc.metadata.keys():
                        return (doc.metadata['source'], doc.metadata['section'], -1, -1)
                    elif 'source' in doc.metadata.keys() and 'start_index' in doc.metadata.keys():
                        return (doc.metadata['source'], "______", -1, doc.metadata['start_index'])
                    elif 'source' in doc.metadata.keys():
                        return (doc.metadata['source'], "______", -1, -1)
                    else:
                        return ("______", "______", -1)
                source_documents = sorted(source_documents, key=lambda x: document_sort_key(x))
                # remove duplicate documents
                source_documents_content = []
                for doc in source_documents:
                    if doc.page_content not in source_documents_content:
                        source_documents_content.append(doc.page_content)
                
                #save non-duplicate documents to save in csv
                source_documents = source_documents_content.copy()
                def get_overlap(a, b, min_overlap=100):
                    for i in range(min_overlap, min(len(a), len(b))):
                        if b.startswith(a[-i:]):
                            return i        
                    return 0
                updated_source_documents_content = [source_documents_content[0]]
                num_docs_used = 1
                #if end of one document is the same as the start of the next document, remove the start of the next document
                for i in range(len(source_documents_content)-1):
                    if len(source_documents_content[i])==0:
                        pass
                    #identify if one document containes the first 100 characters of next document
                    elif updated_source_documents_content[-1].find(source_documents_content[i+1][:min(100,len(source_documents_content[i+1]))])>-1:
                        # identify how long the overlap is
                        overlap = get_overlap(updated_source_documents_content[-1], source_documents_content[i+1])
                        # remove overlapping portions of documents
                        updated_source_documents_content[-1] = updated_source_documents_content[-1] + source_documents_content[i+1][overlap:]
                        source_documents_content[i+1] = ""
                        num_docs_used += 1
                        # updated_source_documents_content.append(source_documents_content[i])
                        # print(f"Removed overlap document to include in: {updated_source_documents_content[-1]}")
                    else :
                        updated_source_documents_content.append(source_documents_content[i+1])
                        num_docs_used += 1
                source_documents_content = updated_source_documents_content
                print(f"Source documents used (before concat): {num_docs_used}")
                print(f"Length of source documents: {len(source_documents_content)}")

                #resort documents on cosine similarity to prompt 
                # source_documents_content = sorted(source_documents_content, key=lambda x: calc_similarity(embedding_function.embed_documents([x])[0],embedding_function.embed_documents([prompt])[0]), reverse=True)

                #concatenate list seperated by \n
                source_documents_content = '\n\n'.join(source_documents_content)
                response = generate_langchain_response(data_dict['prompt'], model=model, system_text="You are an expert in the field using the following information to answer a question:\n" +source_documents_content, template="{context}\n", temperature=temperature)
                # response = multi_chain(inputs={"context": source_documents_content, "prompt": data_dict['prompt']} )
                # df_input.loc[idx,'rag_response'] = response['text']
                df_input.loc[idx,'rag_response'] = response
                df_input.loc[idx,'source_documents'] = str(source_documents)
        else:
            print('Exception: No question provided.')
            df_input.loc[idx,'rag_response'] = 'Exception: No question provided.'
            df_input.loc[idx,'source_documents']
    return df_input


def run_rag_eval(input_path, output_path):
    df_all = pd.read_csv(input_path)
    print(f"Length of df_all {len(df_all)} before dropping exceptions")

    df_all = df_all[df_all['question'].notnull()]
    #ignore questions that are too short
    df_all = df_all[df_all['question'].str.len() > MIN_QUESTION_LENGTH]

    embedding_function = HuggingFaceEmbeddings(
        model_name=ANSWER_EVAL_MODEL,
        # model_kwargs={'device': 'cuda'},
        # encode_kwargs={'normalize_embeddings': True}
        cache_folder="../models/sentencetransformers"
    )
    print(ANSWER_EVAL_MODEL)

    #remove all questions that start with exception
    df_all = df_all[~df_all['question'].str.startswith("Exception: ")]
    df_all = df_all[~df_all['answer'].str.startswith("Exception: ")]
    df_all = df_all[~df_all['answer_full'].str.startswith("Exception: ")]
    print(f"Length of df_all {len(df_all)} after dropping exceptions")

    # apply eval function to question column and rag_response
    df_all['similarity_1'] = df_all.apply(lambda x: calc_similarity(embedding_function.embed_documents([str(x['answer'])])[0],embedding_function.embed_documents([str(x['rag_response'])])[0]), axis=1)
    df_all['similarity_2'] = df_all.apply(lambda x: calc_similarity(embedding_function.embed_documents([str(x['answer_full'])])[0],embedding_function.embed_documents([str(x['rag_response'])])[0]), axis=1)
    #set similarity to max of similarity_1 and similarity_2
    df_all['similarity'] = df_all[['similarity_1','similarity_2']].max(axis=1)

    #let's flag questions that have very low similarities across the board
    #for each question, calculate the average similarity
    df_grouped = df_all[['question','similarity']].groupby('question').mean()['similarity'].reset_index()
    df_grouped_max = df_all[['question','similarity']].groupby('question').max()['similarity'].reset_index()
    print(df_grouped.head())

    df_all = df_all.merge(df_grouped, on='question', how='left', suffixes=('','_avg'))
    df_all = df_all.merge(df_grouped_max, on='question', how='left', suffixes=('','_max'))

    df_all.to_csv(output_path, index=False)
    st.session_state["ran_rag_eval"]=True