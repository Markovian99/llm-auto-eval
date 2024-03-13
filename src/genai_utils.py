import pandas as pd
import streamlit as st
import os
import json
import csv
import tiktoken
import numpy as np
import time

import re
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from bs4 import BeautifulSoup

#sklearn cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, PDFMinerPDFasHTMLLoader, UnstructuredPDFLoader
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter

from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.base import VectorStoreRetriever

# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain, ConversationalRetrievalChain, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain

from langchain.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage 

from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

from config import (MODELS, TEMPERATURE, MAX_TOKENS, EMBEDDING_MODELS, PARSE_METHOD, PROCESSED_DOCUMENTS_DIR, 
                    NO_RESPONSE_EXCEPTION, BAD_FORMAT_EXCEPTION, MIN_QUESTION_LENGTH, USE_AZURE,
                    SUMMARY_QA_GEN, BASIC_QA_GEN, COMPLEX_QA_GEN, MULTIHOP_QA_GEN)

import openai
import google.generativeai as genai
from anthropic import Anthropic

# make sure load_dotenv is run from main app file first
if os.getenv('OPENAI_API_KEY'):
    openai.api_key = os.getenv('OPENAI_API_KEY')
if os.getenv('OPENAI_API_BASE'):
    openai.api_base = os.getenv('OPENAI_API_BASE')
if os.getenv('OPENAI_API_TYPE'):
    openai.api_type = os.getenv('OPENAI_API_TYPE')
if os.getenv('OPENAI_API_VERSION'):
    openai.api_version = os.getenv('OPENAI_API_VERSION')

anth_client=None
if os.getenv('ANTHROPIC_API_KEY'):
    anth_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

if os.getenv('GOOGLE_GEMINI_KEY'):
    genai.configure(api_key=os.getenv("GOOGLE_GEMINI_KEY"))

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
#bard = Bard(token=os.getenv('BARD_API_KEY'))

def safe_loads(json_string):
    try:
        return json.loads(json_string)
    except json.decoder.JSONDecodeError:
        # Replace single quotes with double quotes
        print(f"JSONDecodeError with {json_string}")
        fixed_string = json_string.replace("'", '"')
    except Exception as e:
        print(f"Error with {json_string}")
        print(e)
        return {}
    try:
        return json.loads(fixed_string)
    except json.decoder.JSONDecodeError:
        #Sometimes the single quotes are escaped - this is a hacky fix
        print(f"JSONDecodeError with {json_string}")
        #remove instances of "" when within a sigle quoted string
        fixed_string = json_string.replace("{\'", '{"').replace(" \'", ' "').replace("\':", '":').replace("', \"", '", "').replace("\\'", "'")
    except Exception as e:
        print(f"Error 1 with {fixed_string}")
        print(e)
        return {}
    try:
        return json.loads(fixed_string)
    except Exception as e:
        print(f"Error 2 with {fixed_string}")
        print(e)
        return {}

#turn langchain document to dict
def doc_to_dict(doc):
    return {"page_content":doc.page_content,"metadata":doc.metadata}

def docs_to_dict_strings(docs):
    return [json.dumps({"page_content":doc.page_content,"metadata":doc.metadata}) for doc in docs]

# turn dict string to langchain document
def dict_to_doc(dict_string):
    doc_dict = safe_loads(dict_string)
    if doc_dict == {}:
        return None
    return Document(page_content=doc_dict["page_content"],metadata=doc_dict["metadata"])

# get embedding for one sentence
def get_embedding(sentence):
    # Will download the model the first time it runs
    embedding_function = HuggingFaceEmbeddings(
        model_name=st.session_state["embedding_model"],
        # model_kwargs={'device': 'cuda'},
        # encode_kwargs={'normalize_embeddings': True}
        cache_folder="../models/sentencetransformers"
    )
    try:
        return embedding_function.embed_documents([sentence])[0]
    except Exception as e:
        print(e)
        return np.zeros(384)

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def token_split_docs(docs,chunk_size=256,overlap_size=128, add_start_index = True, use_recursive_chunking = True):
    """Split documents or string into chunks."""
    return_string_list = False
    if isinstance(docs, Document):
        docs = [docs]
    elif isinstance(docs, str):
        docs = [Document(page_content=docs)]
        return_string_list = True
    if use_recursive_chunking:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = chunk_size, chunk_overlap  = overlap_size, add_start_index = add_start_index)
    else:
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size, add_start_index=add_start_index)
    docs_split = text_splitter.split_documents(docs)
    if return_string_list:
        return [doc.page_content for doc in docs_split]
    return docs_split

def get_new_top_pos(c, prev_top_pos=None):
    try:
        return int(re.findall("top:\d+px",c.attrs['style'])[0][4:-2])
    except Exception as e:
        print(e)
        return prev_top_pos


def parse_soup_font_size(content,metadata_orig={}):
    cur_fs = None
    cur_text = ''
    snippets = []   # first collect all snippets that have the same font size
    for c in content:
        sp = c.find('span')
        if not sp:
            continue
        st = sp.get('style')
        if not st:
            continue
        fs = re.findall('font-size:(\d+)px',st)
        if not fs:
            continue
        fs = int(fs[0])
        if not cur_fs:
            cur_fs = fs
        if fs == cur_fs:
            cur_text += c.text
        else:
            snippets.append((cur_text,cur_fs))
            cur_fs = fs
            cur_text = c.text
    snippets.append((cur_text,cur_fs))

    print(f"Found {len(snippets)} snippets")

    cur_idx = -1
    semantic_snippets = []
    # Assumption: headings have higher font size than their respective content
    for s in snippets:
        # if current snippet's font size > previous section's heading => it is a new heading
        if not semantic_snippets or s[1] > semantic_snippets[cur_idx].metadata['heading_font']:
            cur_idx += 1
            metadata={'section':s[0], 'content_font': 0, 'heading_font': s[1], 'doc_idx':cur_idx}
            metadata.update(metadata_orig)
            semantic_snippets.append(Document(page_content='',metadata=metadata))
            
            continue
        
        # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
        # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
        if not semantic_snippets[cur_idx].metadata['content_font'] or s[1] <= semantic_snippets[cur_idx].metadata['content_font']:
            semantic_snippets[cur_idx].page_content += s[0]
            semantic_snippets[cur_idx].metadata['content_font'] = max(s[1], semantic_snippets[cur_idx].metadata['content_font'])
            continue
        
        # if current snippet's font size > previous section's content but less than previous section's heading than also make a new 
        # section (e.g. title of a PDF will have the highest font size but we don't want it to subsume all sections)
        cur_idx += 1
        metadata={'section':s[0], 'content_font': 0, 'heading_font': s[1], 'doc_idx':cur_idx}
        metadata.update(metadata_orig)
        semantic_snippets.append(Document(page_content='',metadata=metadata))        
    return semantic_snippets

def parse_pdf_document(this_pdf, parse_method="timestamp"):
    """ Function to read pdf and split the content into a list of documents"""
    # https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf

    if parse_method=="MAX_TOKENS":
        loader = UnstructuredPDFLoader(this_pdf)
        docs = loader.load()   # entire PDF is loaded as a single Document
        if num_tokens_from_string(docs[0].page_content) > MAX_TOKENS:
            # split into chunks with minimal overlap since this is not for RAG / just to use the pdf to create QA
            docs = token_split_docs(docs,chunk_size=MAX_TOKENS, overlap_size=min(MAX_TOKENS//8, 64))
        return docs
    elif parse_method=="page":
        loader = PyPDFLoader(this_pdf)
        pages = loader.load_and_split()
        return pages
    elif parse_method=="elements":
        loader = UnstructuredPDFLoader(this_pdf,mode="elements")
        docs = loader.load()
        return docs

    # Use PDF as HTML is using "timestamp", "line", "spacing"
    loader = PDFMinerPDFasHTMLLoader(this_pdf)
    data = loader.load()[0]   # entire PDF is loaded as a single Document

    soup = BeautifulSoup(data.page_content,'html.parser')
    content = soup.find_all('div')

    print("Length of content is {}".format(len(content)))
    
    cur_text = ''
    last_top_pos = 0
    new_page = True    

    metadata={}
    metadata.update(data.metadata)

    docs = []   # first collect all snippets that have the same font size

    # just return the lines of text if specified
    if parse_method=="line":
        for idx, c in enumerate(content):
            sp = c.find('span')
            if not sp:
                continue
            st = sp.get('style')
            if not st:
                continue
            fs = re.findall('font-size:(\d+)px',st)
            if not fs:
                continue
            docs.append(Document(page_content=c.text,metadata=metadata.copy()))
        return docs
    elif parse_method=="font-change":
        return parse_soup_font_size(content,metadata_orig=metadata)
    
    #create spacing based parsing first then combine with timestamp if specified
    for idx, c in enumerate(content):
        new_top_pos = get_new_top_pos(c, prev_top_pos=last_top_pos)
        if c.text=='Transcribed by readthatpodcast.com \n': # CHANGE TO BE MORE GENERAL FOR OTHER PDFS
            new_page = True
            continue
        # check if new page based on top position
        if new_top_pos<last_top_pos-30:
            new_page = True

        sp = c.find('span')
        if not sp:
            continue
        st = sp.get('style')
        if not st:
            continue
        fs = re.findall('font-size:(\d+)px',st)
        if not fs:
            continue
        if not last_top_pos:
            last_top_pos = new_top_pos

        #check if not 2 line spaces or if new page is a continuation of previous line by checking case
        if (not new_page and new_top_pos<last_top_pos+30) or (new_page and not c.text[0].isupper()):
            cur_text += c.text
        elif new_page:
            docs.append(Document(page_content=cur_text,metadata=metadata.copy()))
            cur_text = c.text
        else:        
            docs.append(Document(page_content=cur_text,metadata=metadata.copy()))
            cur_text = c.text

        last_top_pos = new_top_pos
        new_page = False

    if cur_text!='':    
        docs.append(Document(page_content=cur_text,metadata=metadata.copy()))

    # return docs if only spacing based parsing is required
    if parse_method=="spacing":
        return docs

    section=""
    if parse_method=="timestamp":
        section="Introduction"
    new_section=False
    final_docs=[]
    doc_idx=0
    #combine document sections based on provided timestamps
    for idx, doc in enumerate(docs):
        #check if new section / if it was a timestamp
        timestamp=re.search("\d+:\d+:\d+",doc.page_content)
        if not timestamp:
            timestamp=re.search("\d+:\d+",doc.page_content)
        if idx==0:
            doc.metadata.update({'section':section,'doc_idx':doc_idx})
            final_docs.append(doc)
            doc_idx+=1
        elif timestamp and timestamp.start()==0 and not new_section:
            section=doc.page_content   
            new_section=True
            if idx<len(docs)-1:
                #get the last sentence from the previous content page
                last_sent=docs[idx-1].page_content.split(".")[-1]
                if len(last_sent)<10:
                    last_sent=docs[idx-1].page_content

                # CHANGE THIS TO ITERATE OVER SENTENCES INSTEAD OF JUST LOOK AT THE FIRST SENTENCE
                next_sent=docs[idx+1].page_content.split(".")[0]
                if next_sent[0].islower() and len(next_sent)<50:
                    final_docs[-1].page_content=final_docs[-1].page_content + next_sent + "."
                    docs[idx+1].page_content=".".join(docs[idx+1].page_content.split(".")[1:])#remove the first sentence from the next document
                elif len(next_sent)<len(docs[idx+1].page_content):
                    this_emb=get_embedding(section)
                    last_emb=get_embedding(last_sent)
                    next_emb=get_embedding(next_sent)
                    #if the next sentence is more similar to the previous sentence than the current section, then combine
                    if cosine_similarity([this_emb],[next_emb])[0][0] <cosine_similarity([last_emb],[next_emb])[0][0]:
                        final_docs[-1].page_content=final_docs[-1].page_content + next_sent + "."
                        docs[idx+1].page_content=".".join(docs[idx+1].page_content.split(".")[1:]) #remove the first sentence from the next document
        else:
            # metadata=doc.metadata
            doc.metadata.update({'section':section,'doc_idx':doc_idx})
            if new_section:
                doc.page_content=section + "\n" + doc.page_content
                new_section=False     
            # doc.metadata=metadata
            final_docs.append(doc)
            doc_idx+=1

    return final_docs

####### Generation Functions ########


# Main function to call LLMs to generate responses
def generate_response(prompt, model, system_prompt="", temperature=0, second_try=False):
    response = "No model selected"
    if second_try:
        time.sleep(5)

    if model != "None":
        try:
            print(f"*{system_prompt}*\n\n{prompt}")
            if model.startswith("Google: "):
                model = genai.GenerativeModel(model[8:])
                response_full = model.generate_content(f"*{system_prompt}*\n\n{prompt}")
                response = response_full.text
            elif model.startswith("OpenAI: "):
                response_full = openai.chat.completions.create(model=model[8:], messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": prompt }],temperature=temperature)
                response = response_full.choices[0].message.content
            elif model.startswith("Anthropic: "):
                message = anth_client.messages.create(max_tokens=1024, system=system_prompt, messages=[{"role": "user","content": prompt}],model=model[11:])
                response = message.content[0].text
        except Exception as e:
            st.warning(f"{model} API call failed. Waiting 5 seconds and trying again.")
            response = generate_response(prompt, model, system_prompt=system_prompt, temperature=temperature, second_try=True)
    return response


# This is a dummy function to simulate generating responses.
def generate_langchain_response(context, system_text="",model='gpt-3.5-turbo', template="", temperature=0, second_try=False):
    if USE_AZURE:
        llm = AzureChatOpenAI(model=model, temperature=temperature)
    else:
        llm = ChatOpenAI(model=model, temperature=temperature)

    response = NO_RESPONSE_EXCEPTION
    try:
        if template != "":
            template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=(system_text)),
                    HumanMessagePromptTemplate.from_template(template),
                ]
            )
            response = llm(template.format_messages(context=context)).content
        else:
            response = llm(ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template("{context}")]).format_messages(context=context)).content
    except Exception as e:
        print(e)
        if not second_try:
            time.sleep(5)
            response = generate_langchain_response(context, system_text=system_text,model=model, template=template, temperature=temperature, second_try=True)
        else:
            pass
    print(response)

    return response


def create_knowledge_base(docs,faiss_dir="../data/faiss-db",embedding_model="all-MiniLM-L6-v2",chunk_size=256,overlap_size=128):
    """Create knowledge base for chatbot."""

    print(f"Splitting {len(docs)} documents")

    for doc in docs:
        if 'start_index' in doc.metadata:
            doc.metadata['start_index_orig']=doc.metadata['start_index']
    
    docs_split = token_split_docs(docs, chunk_size=chunk_size, overlap_size=overlap_size)

    # add back the original start index to the new start index
    for doc in docs_split:
        if 'start_index' in doc.metadata and 'start_index_orig' in doc.metadata:
            doc.metadata['start_index']=doc.metadata['start_index_orig']+doc.metadata['start_index']
            #drop the original start index from dict
            doc.metadata.pop('start_index_orig', None)

    print(f"Created {len(docs_split)} documents")

    # Will download the model the first time it runs
    # embedding_function = SentenceTransformerEmbeddings(
    #     model_name=embedding_model,
    #     cache_folder="../models/sentencetransformers",
    # )
    embedding_function = HuggingFaceEmbeddings(
        model_name=embedding_model,
        # model_kwargs={'device': 'cuda'},
        # encode_kwargs={'normalize_embeddings': True}
        cache_folder="../models/sentencetransformers"
    )

    texts = [doc.page_content for doc in docs_split]
    metadatas = [doc.metadata for doc in docs_split]
    print("""
        Computing embedding vectors and building FAISS db.
        WARNING: This may take a long time. You may want to increase the number of CPU's in your noteboook.
        """
    )

    db = FAISS.from_texts(texts, embedding_function, metadatas=metadatas)  
    # Save the FAISS db 
    db.save_local(faiss_dir)

    print(f"FAISS VectorDB has {db.index.ntotal} documents")


# Update: https://github.com/Raudaschl/rag-fusion/blob/master/main.py
# This appears to do the same as what I am doing here, but also re-ranks the documents
# So, maybe pull more than needed , re-rank, and drop
# Function to generate queries using OpenAI's ChatGPT
def generate_subqueries(original_query, model, second_try=False):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that helps users identify how to search for relevant information. You do not introduce any new topics, specific details, or key words, but provide users with sub-queries phrased as questions to gather information to answer the original input query."},
                {"role": "user", "content": 
                f"""Please return 2 sub-queries that would be helpful to gather the required information to answer the original query. The format of your response should be:
                1. Sub-query 1
                2. Sub-query 2

                Original query: {original_query}

                Sub-queries:"
                """}
            ]
        )
    except Exception as e:
        print(e)
        time.sleep(5)
        if second_try:
            return []
        return generate_subqueries(original_query, model, second_try=True)

    generated_queries = response.choices[0]["message"]["content"].strip()
    if len(generated_queries) < MIN_QUESTION_LENGTH or generated_queries[:MIN_QUESTION_LENGTH].lower().find("n/a") >-1:
        generated_queries = []
    else:
        generated_queries_list = generated_queries.split("\n")
        generated_queries = []
        for i in range(len(generated_queries_list)):            
            generated_queries_list[i] = generated_queries_list[i].strip()
            print(generated_queries_list[i])
            # if generated_queries[i] startswith a number and a period, remove it
            if len(generated_queries_list[i])>MIN_QUESTION_LENGTH and generated_queries_list[i][0].isdigit():
                generated_queries_list[i] = generated_queries_list[i][min(4,generated_queries_list[i].find(" ")+1):]
            if len(generated_queries_list[i]) > MIN_QUESTION_LENGTH:
                generated_queries.append(generated_queries_list[i])
            
    return generated_queries

def generate_kb_response(prompt, model, system_prompt="",template=None,faiss_dir="../data/faiss-db", temperature=0, k=4, include_source=False, retriever = None):
    # (prompt, model, template=None,faiss_dir="../data/faiss-db", temperature=0, k=4)

    if not model.startswith("OpenAI: "):
        return "Please select an OpenAI model."

    if not retriever:
        embedding_function = HuggingFaceEmbeddings(
            model_name=st.session_state["embedding_model"],
            # model_kwargs={'device': 'cuda'},
            # encode_kwargs={'normalize_embeddings': True}
            cache_folder="../models/sentencetransformers"
        )
        db = FAISS.load_local(faiss_dir, embedding_function)
        retriever = VectorStoreRetriever(vectorstore=db, search_kwargs={"k": k})

    relevant_docs = retriever.get_relevant_documents(prompt)

    relevant_docs_str = ""
    docs_with_source = ""
    for doc in relevant_docs:
        if include_source:
            docs_with_source += doc.page_content + "\n" + "Source: " + str(doc.metadata) + "\n\n"
        else:
            relevant_docs_str += doc.page_content + "\n\n"
            docs_with_source += doc.page_content + "\n" + "Source: " + str(doc.metadata) + "\n\n"
    if include_source:
        relevant_docs_str = docs_with_source

    if template is None:
        prompt_full = f"""Answer based on the following context

        {relevant_docs_str}

        Question: {prompt}"""
    else:
        prompt_full = template.format(prompt=prompt, context=relevant_docs_str)

    response = generate_response(prompt_full, model=model, system_prompt=system_prompt, temperature=temperature)
    
    return {'answer':response, 'source_documents':docs_with_source}



####### SUMMARIZATION FUNCTIONS ########

def run_map_reduce_summary(docs):

    model = st.session_state["generation_models"][0]
    temperature = st.session_state["temperature"]

    if model.startswith("OpenAI: "):
        if USE_AZURE:
            llm = AzureChatOpenAI(model=model[8:], temperature=temperature)
        else:
            llm = ChatOpenAI(model=model[8:], temperature=temperature)
    else:
        return "Please select an OpenAI model."

    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please summarize the main themes and key details
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of summaries:
    {doc_summaries}
    Take these and distill it into a final, consolidated summary of the main themes. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=MAX_TOKENS,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=True,
    )

    map_results = map_reduce_chain.__call__(docs)
    map_results_df = pd.DataFrame(map_results)#
    map_results_df['input_documents']=map_results_df['input_documents'].apply(doc_to_dict)
    return map_results_df
    

def run_refine_summary(docs):
    model = st.session_state["generation_models"][0]
    temperature = st.session_state["temperature"]

    if model.startswith("OpenAI: "):
        if USE_AZURE:
            llm = AzureChatOpenAI(model=model[8:], temperature=temperature)
        else:
            llm = ChatOpenAI(model=model[8:], temperature=temperature)
    else:
        return "Please select an OpenAI model."
    
    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary of a document.\n"
        "We have provided an existing summary of text up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with context from continued reading of the document below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the additional context, refine the original summary. If the additional isn't useful, return the original summary."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )
    result = chain({"input_documents": docs}, return_only_outputs=False)
    refine_results_df = pd.DataFrame(result)
    # refine_results_df['input_documents']=map_results_df['input_documents']
    refine_results_df['input_documents'] = refine_results_df['input_documents'].apply(doc_to_dict)
    return refine_results_df



####### LDA THEME FUNCTIONS ########

def preprocess_for_lda(text, stop_words):
    """
    Tokenizes and preprocesses the input text, removing stopwords and short 
    tokens.

    Parameters:
        text (str): The input text to preprocess.
        stop_words (set): A set of stopwords to be removed from the text.
    Returns:
        list: A list of preprocessed tokens.
    """
    result = []
    for token in simple_preprocess(text, deacc=True):
        if token not in stop_words and len(token) > 3:
            result.append(token)
    return result


def run_lda_topics(documents, num_topics, words_per_topic):
    print("Run LDA")
    """
    Extracts topics and their associated words from a list of langchain docs using the 
    Latent Dirichlet Allocation (LDA) algorithm.

    Parameters:
        documents (list): List of langchain docs.
        num_topics (int): The number of topics to discover.
        words_per_topic (int): The number of words to include per topic.

    Returns:
        list: A list of num_topics sublists, each containing relevant words 
        for a topic.
    """

    # Preprocess the documents
    nltk.download('stopwords')
    stop_words = set(stopwords.words(['english','spanish']))
    processed_documents = [preprocess_for_lda(doc.page_content, stop_words) for doc in documents]

    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary(processed_documents)
    corpus = [dictionary.doc2bow(doc) for doc in processed_documents]

    # Build the LDA model
    lda_model = LdaModel(
        corpus, 
        num_topics=num_topics, 
        id2word=dictionary, 
        passes=15
        )

    # Retrieve the topics and their corresponding words
    topics = lda_model.print_topics(num_words=words_per_topic)

    # Store each list of words from each topic into a list
    topics_ls = []
    for topic in topics:
        words = topic[1].split("+")
        topic_words = [word.split("*")[1].replace('"', '').strip() for word in words]
        topics_ls.append(topic_words)

    return topics_ls, lda_model, dictionary



def topics_from_docs(documents, num_topics, words_per_topic):
    """
    Generates descriptive prompts for LLM based on topic words extracted from a 
    PDF document.

    This function takes the output of `get_topic_lists_from_pdf` function, 
    which consists of a list of topic-related words for each topic, and 
    generates an output string in table of content format.

    Parameters:
        llm (LLM): An instance of the Large Language Model (LLM) for generating 
        responses.
        file (str): The path to the PDF file for extracting topic-related words.
        num_topics (int): The number of topics to consider.
        words_per_topic (int): The number of words per topic to include.

    Returns:
        str: A response generated by the language model based on the provided 
        topic words.
    """

    # Extract topics and convert to string
    list_of_topicwords, lda_model, dictionary = run_lda_topics(documents, num_topics, words_per_topic)
    string_lda = ""
    for list in list_of_topicwords:
        string_lda += str(list) + "\n"

    print(string_lda)

    system_text = f"Describe the topic of each of the {num_topics} double-quote delimited lists in a simple sentence and also write down three possible different subthemes."
    # Create the template
    template_string = '''The following lists are the result of an 
        algorithm for topic discovery.
        Do not provide an introduction or a conclusion, only describe the 
        topics. Do not mention the word "topic" when describing the topics.
        Use the following template for the response.

        1: <<<(sentence describing the topic)>>>
        - <<<(Phrase describing the first subtheme)>>>
        - <<<(Phrase describing the second subtheme)>>>
        - <<<(Phrase describing the third subtheme)>>>

        2: <<<(sentence describing the topic)>>>
        - <<<(Phrase describing the first subtheme)>>>
        - <<<(Phrase describing the second subtheme)>>>
        - <<<(Phrase describing the third subtheme)>>>

        ...

        n: <<<(sentence describing the topic)>>>
        - <<<(Phrase describing the first subtheme)>>>
        - <<<(Phrase describing the second subtheme)>>>
        - <<<(Phrase describing the third subtheme)>>>

        Lists: """{context}""" '''   

    response = generate_langchain_response(string_lda, system_text=system_text,model=st.session_state['generation_models'][0], template=template_string, temperature=st.session_state['temperature'], second_try=False)

    return response, list_of_topicwords, lda_model, dictionary