import streamlit as st
from datetime import date, datetime
import numpy as np
import pandas as pd
from io import StringIO
import json
import os
import itertools
import seaborn as sns
import textwrap
import plotly.express as px

import zipfile
import tarfile

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings


from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from config import (MODELS, TEMPERATURE, MAX_TOKENS, APP_NAME, PROCESSED_DOCUMENTS_DIR, REPORTS_DOCUMENTS_DIR, EMBEDDING_MODELS, PARSE_METHOD, CHUNK_SIZE,
            NO_RETRIEVER_NAME, MIN_QUESTION_LENGTH, MIN_ANSWER_SIMILARITY, ANSWER_EVAL_MODEL, MAX_QA_PER_TEST,
            SUMMARY_QA_GEN, BASIC_QA_GEN, COMPLEX_QA_GEN, COMPARISON_QA_GEN, SMALL_CONTEXT_QA_GEN, MULTIHOP_QA_GEN)
from app_utils import (delete_files_from_dir, get_experiment_category,
                run_jeopardy, run_rag, run_rag_multihop, run_rag_eval)
from genai_utils import (parse_pdf_document, docs_to_dict_strings, safe_loads, token_split_docs, create_knowledge_base,
                         run_map_reduce_summary, run_refine_summary, topics_from_docs)


def run_sidebar():

    experiment_dir = st.session_state["experiment_dir"]

    """This function runs the sidebar container"""
    models = st.multiselect(f"GenAI Models (Summaries and Q&A)", MODELS, default=[MODELS[0]])
    st.caption("We recommend using a less expensive models when for summaries in Step 1.")
    st.session_state["generation_models"] = models

    temperature = st.slider("temperature", min_value=0.0, max_value=2.0, value=1.0*TEMPERATURE, step=.1)
    st.session_state["temperature"] = temperature

    # emb_model = st.selectbox(f"Semantic Join Embeddings", EMBEDDING_MODELS)
    # st.session_state["embedding_model"] = emb_model

    parse_method = st.selectbox(f"Parse PDF Method", PARSE_METHOD)
    st.session_state["parse_method"] = parse_method

    map_summary = st.checkbox("Map Reduce Summary Q&A", value=True)
    refine_summary = st.checkbox("Refine Summary Q&A", value=True)
    semantic_all_qa = st.checkbox("All Doc Semantic Search", value=True)
    per_chunk_qa = st.checkbox("Per Doc Chunk Q&A", value=True)
    grouped_chunk_qa = st.checkbox("Grouped Doc Chunk Q&A", value=True)
    granular_qa = st.checkbox("Granular Q&A", value=True)

    st.text("")

    #check if a user uploaded a .zip, .tar, or .tar.gz file
    compressed_files = [file for file in os.listdir("../data/raw/") if file.endswith(".zip") or file.endswith(".tar") or file.endswith(".tar.gz")]
    if len(compressed_files)>0:
        #extract the files
        for file in compressed_files:
            st.write(f"Extracting {file}")
            if file.endswith(".zip"):
                with zipfile.ZipFile(f"../data/raw/{file}", 'r') as zip_ref:
                    zip_ref.extractall("../data/raw/")
                    zip_ref.close()
            elif file.endswith(".tar") or file.endswith(".tar.gz"):
                with tarfile.open(f"../data/raw/{file}", 'r') as tar_ref:
                    tar_ref.extractall("../data/raw/")
                    tar_ref.close()
            try:
                os.remove(f"../data/raw/{file}")
            except Exception as e:
                print(f"Error: {e}")
        # #loop through all the directories and move the files to the raw folder
        # for root, dirs, files in os.walk("../data/raw/"):
        #     for file in files:
        #         os.rename(os.path.join(root, file), os.path.join("../data/raw/", file))

    #add streamlit dropdown for file selection based on list of files in raw folder
    files = []
    #find all files in all folders
    for root, dirs, all_files in os.walk("../data/raw/"):
        for file in all_files:
            if file.endswith(".pdf") or file.endswith(".txt") or file.endswith(".md"):
                files.append(os.path.join(root, file))
    print(files)

    #add select option
    files.insert(0,"Use All Files")

    selected_file = "Use All Files"
    if len(files)>0:
        selected_file = st.selectbox("Use All or Select a file", files)

    st.text("")

    build_docs_button = st.button("Step 1: Build Docs and Summaries")
    if build_docs_button:

        # create filter to get all .txt and .md files for glob
        SOURCE_DOCUMENTS_FILTERS = ["**/*.txt", "**/*.md"]
        docs_txt_md = []
        for sss in SOURCE_DOCUMENTS_FILTERS:
            loader = DirectoryLoader(f"../data/raw/", glob=sss)
            docs = loader.load()
            docs_txt_md.extend(docs)
        print(f"Loaded {len(docs_txt_md)} .txt. or .md documents ")
        if len(docs_txt_md)>0:
            docs_txt_md = token_split_docs(docs_txt_md,chunk_size=MAX_TOKENS,overlap_size=min(MAX_TOKENS//8, 64))
        print(f"After splitting: {len(docs_txt_md)} .txt. or .md documents ")

        #parse pdf files in the raw folder
        docs=[]
        if selected_file != "Use All Files":
            docs = parse_pdf_document(os.path.join("../data/raw/",selected_file), parse_method=parse_method) 
        else:
            for file in files[1:]:
                docs.extend(parse_pdf_document(os.path.join("../data/raw/",file), parse_method=parse_method))
        #split docs into chunks of 3500 tokens if not using MAX_TOKENS already
        if parse_method != "MAX_TOKENS" and len(docs)>0:
            docs = token_split_docs(docs,chunk_size=MAX_TOKENS,overlap_size=min(MAX_TOKENS//8, 64))
        
        docs = docs + docs_txt_md

        print(f"Created {len(docs)} total documents from parsing processes")

        # Build input documents csv
        df_input=pd.DataFrame(docs_to_dict_strings(docs),columns=['input_documents'])
        df_input['input_documents']=df_input['input_documents'].apply(lambda x: json.loads(x))
        df_input['page_content']=df_input['input_documents'].apply(lambda x: x['page_content'])
        df_input['metadata']=df_input['input_documents'].apply(lambda x: x['metadata'])
        for key in df_input['metadata'][0].keys():
            df_input[key]=df_input['metadata'].apply(lambda x: x[key])
        df_input.to_csv(os.path.join(experiment_dir,"input_documents.csv"),index=False)

        # split docs to create docs to randomly select from to create docs for grouped qa
        docs_split = token_split_docs(docs,chunk_size=1024,overlap_size=64)
        print(f"Created {len(docs_split)} documents for Grouped Q&A")
        columns=['page_content','metadata1','metadata2','metadata3']
        df_grouped = pd.DataFrame([],columns=columns)
        for ii in range(len(docs)):
            doc1 = docs_split[np.random.randint(0,len(docs_split))]
            doc2 = docs_split[np.random.randint(0,len(docs_split))]
            doc3 = docs_split[np.random.randint(0,len(docs_split))]
            df_grouped=pd.concat([df_grouped,pd.DataFrame([[str(doc1.page_content)+"\n\n"+str(doc2.page_content)+"\n\n"+str(doc3.page_content),\
                json.dumps(doc1.metadata),json.dumps(doc2.metadata),json.dumps(doc2.metadata)]],columns=columns)],ignore_index=True)
        df_grouped.to_csv(os.path.join(experiment_dir,"input_documents_grouped.csv"),index=False)

        # grab 2 sentences from each document to use for granular qa
        columns=['page_content','metadata','sentence_idx']
        df_sentence = pd.DataFrame([],columns=columns)
        #grab 2 sentences from each document
        for doc in docs:
            page_content = doc.page_content.split(".")
            #add the period back on
            page_content = [sentence+"." for sentence in page_content]
            if len(page_content)==1:
                sentence = page_content[0]
                sentence_idx=-1
            else: 
                for ll in range(5):
                    sentence_idx=np.random.randint(0,len(page_content)-1)
                    sentence = page_content[sentence_idx] + page_content[sentence_idx+1]
                    #check if sentence is contains website address
                    if "http" in sentence or "www" in sentence or ".com" in sentence or ".edu" in sentence:
                        continue
                    if len(sentence)>30: #if question isn't too short to have any context
                        break
                    else:
                        sentence_idx=-1
                        sentence = doc.page_content
            if len(sentence)>30 and not ("http" in sentence or "www" in sentence or ".com" in sentence or ".edu" in sentence): #if sentence isn't too short and not a website reference
                #replace \n with a space
                sentence = sentence.replace(" \n"," ")
                sentence = sentence.replace("\n"," ")
                df_sentence=pd.concat([df_sentence,pd.DataFrame([[sentence,json.dumps(doc.metadata),sentence_idx]],columns=columns)],ignore_index=True)
        df_sentence.to_csv(os.path.join(experiment_dir,"input_documents_granular.csv"),index=False)

        # NEED TO ADD CODE HERE
        if map_summary:
            print("Running Map Reduce Summary")
            map_results_df = run_map_reduce_summary(docs)
            map_results_df.to_csv(os.path.join(experiment_dir,'summary_map_results.csv'),index=False)
        if refine_summary:
            print("Running Refine Summary")
            refine_results_df = run_refine_summary(docs)
            refine_results_df.to_csv(os.path.join(experiment_dir,'summary_refine_results.csv'),index=False)
        if semantic_all_qa:
            print("Running LDA: All Doc Semantic Search")
            print("THIS IS NOT FULLY IMPLEMENTED YET")
            response, list_of_topicwords, lda_model, dictionary = topics_from_docs(docs)
            

    st.text("")

    build_qa_button = st.button("Step 2: Build Q&A Evaluation Set")
    if build_qa_button:
        #check if check and if file exists
        if map_summary:
            if os.path.exists(os.path.join(experiment_dir,"summary_map_results.csv")):
                run_jeopardy(os.path.join(experiment_dir,"summary_map_results.csv"), column_name="intermediate_steps", template=SUMMARY_QA_GEN, max_subsample=10, full_response=True)
                print("Map Reduce Summary Complete")
            else:
                st.error("Map Reduce Summary Results Not Found")
        if refine_summary:
            if os.path.exists(os.path.join(experiment_dir,"summary_refine_results.csv")):
                run_jeopardy(os.path.join(experiment_dir,"summary_refine_results.csv"), column_name="intermediate_steps", template=SUMMARY_QA_GEN, max_subsample=10, full_response=True)
                print("Refine Summary Complete")
            else:
                st.error("Refine Summary Results Not Found")
        if semantic_all_qa:
            if os.path.exists(os.path.join(experiment_dir,"all_semantic_results.csv")):
                print("All Doc Semantic Search")
            else:
                st.error("All Doc Semantic Search Results Not Found")
        if per_chunk_qa:
            if os.path.exists(os.path.join(experiment_dir,"input_documents.csv")):
                    # MAYBE ADD CODE TO BREAK APART DOCUMENTS INTO CHUNKS HERE ... GRAB A RANDOM CHUNK FROM EACH DOCUMENT ?

                run_jeopardy(os.path.join(experiment_dir,"input_documents.csv"), column_name="page_content", template=BASIC_QA_GEN, suffix="_basic_qa", max_subsample=MAX_QA_PER_TEST, full_response=True)
                run_jeopardy(os.path.join(experiment_dir,"input_documents.csv"), column_name="page_content", template=COMPLEX_QA_GEN, suffix="_complex_qa",max_subsample=MAX_QA_PER_TEST, full_response=True)
                print("Input Documents QA Complete")
            else:
                st.error("Input Documents Not Found")
        if grouped_chunk_qa:
            if os.path.exists(os.path.join(experiment_dir,"input_documents_grouped.csv")):
                run_jeopardy(os.path.join(experiment_dir,"input_documents_grouped.csv"), column_name="page_content", template=COMPARISON_QA_GEN, suffix="_compare_qa",max_subsample=MAX_QA_PER_TEST, full_response=True)
                run_jeopardy(os.path.join(experiment_dir,"input_documents_grouped.csv"), column_name="page_content", template=MULTIHOP_QA_GEN, suffix="_multihop_qa",max_subsample=MAX_QA_PER_TEST, full_response=True)
                print("Grouped Input Documents QA Complete")
            else:
                st.error("Grouped Input Documents Not Found")
        if granular_qa:
            if os.path.exists(os.path.join(experiment_dir,"input_documents_granular.csv")):
                run_jeopardy(os.path.join(experiment_dir,"input_documents_granular.csv"), column_name="page_content", template=SMALL_CONTEXT_QA_GEN, suffix="_qa", max_subsample=MAX_QA_PER_TEST, full_response=True)
                print("Granular QA Complete")
            else:
                st.error("Granular Documents Not Found")
    
    for ii in range(3):
        st.text("")

    uploaded_file = st.file_uploader("Add Doc to Repo")
    if uploaded_file is not None and st.session_state["uploaded_file"] != uploaded_file.name:
        with open(os.path.join("../data/raw/",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["uploaded_file"] = uploaded_file.name
        print(f"File Uploaded: {uploaded_file.name}")

    clear_button = st.button("Clear Repo")
    if clear_button:
        delete_files_from_dir("../data/raw/")
        # delete_files_from_dir("../data/processed/")
        # delete_files_from_dir("../data/temp/")
        # delete_files_from_dir("../data/reports/")




def analyzer_window():
    """This function runs the upload and settings container"""
    experiment_dir = st.session_state["experiment_dir"]
    
    general_context = st.session_state["general_context"]
    # brief_description = st.text_input("Please provide a brief description of the knowledge base (e.g. These files contain transcripts from a science-based podcast)", "")
    # if len(brief_description)>0:
    #         general_context = general_context + "The following brief description of the file was provided: "+ brief_description + "\n"
    #         st.session_state["general_context"] = general_context

    model_list = []
    emb_model_list = []
    chunk_list = []
    system_prompt = ""
    prompt_template = ""   

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("Generation Models")
        # for model_name in MODELS:
        #     check_temp = st.checkbox(model_name, value=False)
        #     model_list.append(check_temp)
        this_model = st.selectbox("Select Model", ["Please Select"]+MODELS)
        model_list.append(this_model)
        if model_list[0] != "Please Select":
            system_prompt = st.text_area("System Prompt", "You are an assistant that is helpful, creative, clever, and very friendly.")
    with col2:
        st.write("Embedding Models")
        # for emb_model_name in EMBEDDING_MODELS:
        #     check_temp = st.checkbox(emb_model_name, value=False)
        #     emb_model_list.append(check_temp)
        this_emb_model = st.selectbox("Select Model", ["Please Select"]+EMBEDDING_MODELS)
        emb_model_list.append(this_emb_model)
        if emb_model_list[0] != "Please Select":
            if emb_model_list[0] == NO_RETRIEVER_NAME:
                st.text("Prompt Template must include {prompt}")
                prompt_template = st.text_area("Prompt Template", """{prompt}""")
                top_k = [0]
            else:
                st.text("Prompt Template must include {context} and {prompt}")
                prompt_template = st.text_area("Prompt Template", """Answer based on the following context \n\n{context}\n\nQuestion: {prompt}""")
                top_k = st.multiselect("Number of Docs to Pull", [4,5,6,7,8,16], default=[5])
    with col3:
        st.write("Chunking Size")
        # for chunk_size in CHUNK_SIZE:
        #     check_temp = st.checkbox(chunk_size, value=False)
        #     chunk_list.append(check_temp)
        if emb_model_list[0] == NO_RETRIEVER_NAME:
            st.text("Chunking not needed")
            this_chunk = "0,0"
        else:
            this_chunk = st.selectbox("Select Chunk Size", ["Please Select"]+CHUNK_SIZE)
        chunk_list.append(this_chunk)

    with col4:
        st.write("Advance Pipeline Settings")
        use_multihop = st.selectbox('Use with Query Splitting', ['No','Entity Extraction', 'LLM-based'])
        use_summaries = st.selectbox("Use with Summaries?", ['No','Add to Knowledge Base'])

    col1, col2, _, _, _, _, _, _ = st.columns(8)
    with col1:
        save_button = st.button("Save Pipeline")
    

    
    if save_button and (model_list[0] == "Please Select" or emb_model_list[0] == "Please Select" or chunk_list[0] == "Please Select"):
        st.error("Please select a model, embedding model, and chunk size")
    elif save_button:
        # save experiment settings
        experiment_desc_dict = {}
        experiment_desc_dict['model'] = model_list[0]
        experiment_desc_dict['emb_model'] = emb_model_list[0]
        experiment_desc_dict['chunk'] = chunk_list[0]
        experiment_desc_dict['k'] = top_k[0]
        experiment_desc_dict['use_multihop'] = use_multihop
        experiment_desc_dict['use_summaries'] = use_summaries
        experiment_desc_dict['system_prompt'] = system_prompt
        experiment_desc_dict['prompt_template'] = prompt_template
        #check is experiment_list.csv exists
        if os.path.exists(os.path.join(experiment_dir,"experiment_list.csv")):
            experiment_list = pd.read_csv(os.path.join(experiment_dir,"experiment_list.csv"))
            experiment_num = experiment_list["experiment_num"].max()+1
            experiment_desc_dict['experiment_num']=experiment_num
            this_exp_df = pd.DataFrame([experiment_desc_dict])
            experiment_list = pd.concat([experiment_list, this_exp_df],ignore_index=True)
            experiment_list.to_csv(os.path.join(experiment_dir,"experiment_list.csv"),index=False)
        else:
            experiment_desc_dict['experiment_num']=1
            experiment_list = pd.DataFrame([experiment_desc_dict])
            #make experiment_num the first column
            cols = experiment_list.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            experiment_list = experiment_list[cols]
            experiment_list.to_csv(os.path.join(experiment_dir,"experiment_list.csv"),index=False) 

    if not os.path.exists(os.path.join(experiment_dir,"experiment_list.csv")):
        st.header("No Pipelines Saved - Please Save a Pipeline First")
        return
    
    with col2:
        if os.path.exists(os.path.join(experiment_dir,"experiment_list.csv")):
            experiment_list = pd.read_csv(os.path.join(experiment_dir,"experiment_list.csv"))
            show_experiments = st.checkbox("Show Pipelines", value=False)
    if show_experiments:
        st.dataframe(experiment_list)
    
    experiment_list = pd.read_csv(os.path.join(experiment_dir,"experiment_list.csv"))
    experiment_numnbers = experiment_list["experiment_num"].unique()

    #select experiments to run
    with col1:
        experiments_to_run = st.multiselect("Select Pipeline to Run (experiment_num)", experiment_numnbers, default=None)

    #check if all_results.csv exists
    if os.path.exists(os.path.join(experiment_dir,"all_results.csv")):
        all_results = pd.read_csv(os.path.join(experiment_dir,"all_results.csv"))
        ran_experiments = all_results["experiment_num"].unique()
        # if experiment is already run give a warning (ovelapping experiments)
        if len(set(experiments_to_run).intersection(set(ran_experiments)))>0:
            st.warning("You selected experiments already run. Please change selection or delete from all_results.csv")
            return
         
    run_experiments_list = experiment_list[experiment_list["experiment_num"].isin(experiments_to_run)]
    
    #get all files ending in _qa.csv
    qa_files = [file for file in os.listdir(experiment_dir) if file.endswith("_qa.csv")]

    model_list = [model for model, check in zip(MODELS, model_list) if check]
    emb_model_list = [emb_model for emb_model, check in zip(EMBEDDING_MODELS, emb_model_list) if check]
    chunk_list = [chunk for chunk, check in zip(CHUNK_SIZE, chunk_list) if check]

    run_exp_button=None
    if len(run_experiments_list)>0:
        run_exp_button = st.button("Step 1: Run Experiments")
    elif not os.path.exists(os.path.join(experiment_dir,"all_results.csv")):
        st.header("Please select experiments to run")
        return

    col1, col2, _, _, _, _ = st.columns(6)
    with col1:
        metric_to_use = st.selectbox("Metric to Use", ["Max Similarity", "Similarity 1", "Similarity 2", "ROUGE", "LLM-eval", "Answer Relevancy", "Critique", "Faithfulness*","Context Relevancy*", "Context Recall*"])
    with col2:
        if metric_to_use.find("*")>-1:
            st.warning("Evaluation only performed on pipelines with retrieval")
    eval_rag_button = st.button("Step 2: Evaluate RAG Experiments")

    if run_exp_button:
        print("Running RAG Experiments")
        df_input = pd.read_csv(os.path.join(experiment_dir, "input_documents.csv"))
        df_input["page_content"] = df_input["page_content"].astype(str)
        df_input["metadata"] = df_input["metadata"].apply(lambda x: safe_loads(x))

        doc_list = []
        for index, row in df_input.iterrows():
            doc_list.append(Document(page_content=str(row['page_content']), metadata=row['metadata']))

        #check if "all_results.csv exists"
        if os.path.exists(os.path.join(experiment_dir,"all_results.csv")):
            all_results_orig = pd.read_csv(os.path.join(experiment_dir,"all_results.csv"))
        else:
            all_results_orig=None

        #loop over the model_list, emb_model_list, chunk_list when True
        all_results=[]
        for idx, row in run_experiments_list.iterrows():
            
            experiment_num = row['experiment_num']
            model = row['model']
            emb_model = row['emb_model']
            chunk = row['chunk']
            k = row['k']
            use_multihop = row['use_multihop']
            use_summaries = row['use_summaries']
            system_prompt = row['system_prompt']
            prompt_template = row['prompt_template']
            if use_multihop=="No":
                experiment_desc =f"model: {model}; emb_model: {emb_model}; chunk: {chunk}; k: {k}"
                st.write("Running: "+ experiment_desc)

                if emb_model!=NO_RETRIEVER_NAME:
                    # split chunk into chunk_size and overlap_size
                    chunk_size = int(chunk.split(",")[0])
                    overlap_size = int(chunk.split(",")[1])

                    #if too big skip iteration
                    if k*chunk_size > MAX_TOKENS:
                        st.write(f"k={k} too large for chunk size {chunk_size}, skipping: "+ experiment_desc)
                        continue

                    # NEED TO ADD SUMMARIES OPTION HERE
                    #split data into chunks
                    create_knowledge_base(doc_list,faiss_dir="../data/temp-faiss-db",embedding_model=emb_model,\
                                        chunk_size=chunk_size,overlap_size=overlap_size)

                #run RAG
                for qa_file in qa_files:
                    rag_results = run_rag(model_name=model, emb_model_name=emb_model, file_path=os.path.join(experiment_dir, qa_file),\
                            question_column="question", system_prompt= system_prompt, template=prompt_template, faiss_dir="../data/temp-faiss-db",temperature=0, k=k)
                    rag_results['experiment_num'] = experiment_num 
                    # rag_results['model'] = model
                    # rag_results['emb_model'] = emb_model 
                    # rag_results['chunk'] = chunk 
                    # rag_results['k'] = k 
                    # rag_results['use_multihop'] = use_multihop 
                    # rag_results['use_summaries'] = use_summaries 
                    # rag_results['system_prompt'] = system_prompt 
                    # rag_results['prompt_template'] = prompt_template 
                    rag_results["experiment_description"] = experiment_desc
                    rag_results["experiment_category"] = get_experiment_category(qa_file)
                    rag_results["qa_file"] = qa_file
                    rag_results.to_csv(os.path.join(experiment_dir, f"Exp_{experiment_num}_results.csv"), index=False)
                    all_results.append(rag_results)

        
        
            elif use_multihop == "LLM-based":
                #Run experiment with multi-hop
                experiment_desc =f"model: {model}; emb_model: {emb_model}; chunk: {chunk}; k: {k}; Expert special: LLM-based Multi-hop"
                if emb_model!=NO_RETRIEVER_NAME:
                    # split chunk into chunk_size and overlap_size
                    chunk_size = int(chunk.split(",")[0])
                    overlap_size = int(chunk.split(",")[1])

                    #if too big skip iteration
                    if k*chunk_size > MAX_TOKENS:
                        st.write(f"k={k} too large for chunk size {chunk_size}, skipping: "+ experiment_desc)
                        continue

                    # NEED TO ADD SUMMARIES OPTION HERE

                    #split data into chunks
                    create_knowledge_base(doc_list,faiss_dir="../data/temp-faiss-db",embedding_model=emb_model,\
                                        chunk_size=chunk_size,overlap_size=overlap_size)
                
                for qa_file in qa_files:                
                    rag_results = run_rag_multihop(model_name=model, emb_model_name=emb_model, file_path=os.path.join(experiment_dir, qa_file),\
                            question_column="question", faiss_dir="../data/temp-faiss-db",temperature=0, k=k)

                    rag_results["experiment_description"] = experiment_desc
                    rag_results["experiment_category"] = get_experiment_category(qa_file)
                    rag_results["experiment_num"] = experiment_num
                    rag_results["qa_file"] = qa_file
                    rag_results.to_csv(os.path.join(experiment_dir, f"Exp_{experiment_num}_results.csv"), index=False)

                    all_results.append(rag_results)

        if all_results:
            all_results = pd.concat(all_results,ignore_index=True)
            if all_results_orig is not None:
                all_results = pd.concat([all_results_orig, all_results],ignore_index=True)
            all_results.to_csv(os.path.join(experiment_dir, "all_results.csv"), index=False)

        if use_multihop == "Run All" or use_multihop == "Entity Extraction":
            print("Running Entity Extraction Multi-hop")
    

    if eval_rag_button and not os.path.exists(os.path.join(experiment_dir, "all_results.csv")): 
        st.error("Please run an experiment first")
        return
    elif eval_rag_button and os.path.exists(os.path.join(experiment_dir, "all_results.csv")):

        #check if all_results.csv is newer than all_similarities.csv and rerun rag_eval if so
        if os.path.exists(os.path.join(experiment_dir, "all_similarities.csv")) and (os.path.getmtime(os.path.join(experiment_dir, "all_results.csv")) < os.path.getmtime(os.path.join(experiment_dir, "all_similarities.csv"))):
            st.warning("Not re-calculating: all_similarities.csv is newer than all_results.csv")
        else:
            run_rag_eval(input_path=os.path.join(experiment_dir, "all_results.csv"), output_path=os.path.join(experiment_dir, "all_similarities.csv"))

    if os.path.exists(os.path.join(experiment_dir, "all_results.csv")) and os.path.exists(os.path.join(experiment_dir, "all_similarities.csv")) and \
            (os.path.getmtime(os.path.join(experiment_dir, "all_results.csv")) < os.path.getmtime(os.path.join(experiment_dir, "all_similarities.csv"))):
        # read in all_similarities.csv
        df_all = pd.read_csv(os.path.join(experiment_dir, "all_similarities.csv"))

        #measure the std of all similarity scores to drop extreme questions
        results_mean = df_all['similarity'].mean()
        results_std = df_all['similarity'].std()

        st.write(f"Results length: {len(df_all)}")
        st.write(f"Results Mean: {results_mean} and Std: {results_std}")
        df_all = df_all[df_all['similarity_avg'] > results_mean - 3*results_std]
        df_all = df_all[df_all['similarity_max'] > results_mean - 2.5*results_std]
        st.write(f"Results length after dropping questions: {len(df_all)}")        

        #create a drop down to decde which leaderboard to show
        st.write("Leaderboard Options")
        leaderboard_options = ["Leaderboard Overall", "Leaderboard by Category", "Leaderboard by File"]
        leaderboard_option = st.selectbox("Select Leaderboard", leaderboard_options)

        def write_boxplot(df_all_subset, x_var="similarity", y_var="experiment_description"):
            # Create the boxplot
            fig = px.box(df_all_subset, x=x_var, y=y_var, points="all")

            #for 'question', 'answer_full', 'rag_response', create new lines every 100 characters if <br> doesn't exist
            df_all_subset['question'] = df_all_subset['question'].apply(lambda x: "<br>".join(textwrap.wrap(x, width=100)) if "<br>" not in x else x)
            df_all_subset['answer_full'] = df_all_subset['answer_full'].apply(lambda x: "<br>".join(textwrap.wrap(x, width=100)) if "<br>" not in x else x)
            df_all_subset['rag_response'] = df_all_subset['rag_response'].apply(lambda x: "<br>".join(textwrap.wrap(x, width=100)) if "<br>" not in x else x)

            fig.update_traces(hovertemplate='<b>Question:</b> %{customdata[0]}<br><b>Answer:</b> %{customdata[1]}<br><b>RAG Response:</b> %{customdata[2]}',
                            customdata=df_all_subset[['question', 'answer_full', 'rag_response']].values)

            # Increase the size of the plot
            fig.update_layout(autosize=True, width=1600, height=800)

            # Increase font size of labels and title
            fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=14))
            fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=14))
            fig.update_layout(title_font=dict(size=20))

            # Show the plot
            st.plotly_chart(fig)

            # Add CSS to wrap y-axis labels
            st.markdown("""
                <style>
                .yaxislayer-above {
                    text-align: left !important;
                    white-space: pre-line !important;
                }
                </style>
            """, unsafe_allow_html=True)

        if leaderboard_option == "Leaderboard Overall":
            leaderboard_total = df_all.groupby(["experiment_description"]).agg({"similarity_1": ["mean", "std"], "similarity_2": ["mean", "std"], "similarity": ["mean", "std"]}).reset_index()
            leaderboard_total["count"] = df_all.groupby(["experiment_description"]).size().values
            leaderboard_total = leaderboard_total.sort_values(by=[("similarity","mean")], ascending=False)
            st.dataframe(leaderboard_total)
            #create a boxplot of the results by experiment_description using seabord
            write_boxplot(df_all)
            
            
        elif leaderboard_option == "Leaderboard by Category":
            leaderboard_cat = df_all.groupby(["experiment_description","experiment_category"]).agg({"similarity_1": ["mean", "std"], "similarity_2": ["mean", "std"], "similarity": ["mean", "std"]}).reset_index()
            leaderboard_cat["count"] = df_all.groupby(["experiment_description","experiment_category"]).size().values
            st.dataframe(leaderboard_cat)
            for category in leaderboard_cat["experiment_category"].unique():
                st.write(category)
                df_all_subset = df_all[df_all["experiment_category"]==category]
                write_boxplot(df_all_subset)
            

        elif leaderboard_option == "Leaderboard by File":
            leaderboard_details = df_all.groupby(["experiment_description","experiment_category", "qa_file"]).agg({"similarity_1": ["mean", "std"], "similarity_2": ["mean", "std"], "similarity": ["mean", "std"]}).reset_index()
            leaderboard_details["count"] = df_all.groupby(["experiment_description","experiment_category", "qa_file"]).size().values
            st.dataframe(leaderboard_details)
            for qa_file in leaderboard_details["qa_file"].unique():
                st.write(qa_file)
                df_all_subset = df_all[df_all["qa_file"]==qa_file]
                write_boxplot(df_all_subset)
            
  

        