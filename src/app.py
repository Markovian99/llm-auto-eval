import streamlit as st
from datetime import date, datetime
import pandas as pd
import numpy as np
from io import StringIO
import json
import os

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from config import MODELS, TEMPERATURE, MAX_TOKENS, APP_NAME, EMBEDDING_MODELS, PARSE_METHOD, BASIC_QA_GEN, SUMMARY_QA_GEN, COMPLEX_QA_GEN, COMPARISON_QA_GEN, MULTIHOP_QA_GEN, SMALL_CONTEXT_QA_GEN, PROCESSED_DOCUMENTS_DIR
from app_utils import initialize_session_state
from app_sections import run_sidebar, analyzer_window

# default session state variables
initialize_session_state()

st.set_page_config(layout='wide')

col1, col2 = st.columns([4,1])
with col1:
    # App layout
    st.title(APP_NAME)
with col2:
    page = st.radio("Choose a view", ["Preview/Download Files", "Run Experiments"])



# Inject custom CSS to set the width of the sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 300px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

#general context for prompts
with st.sidebar:
    st.sidebar.title("Generate Evaluation Data")

    #list folder in processed folder
    exp_list = os.listdir(PROCESSED_DOCUMENTS_DIR)
    exp_list = [exp for exp in exp_list if os.path.isdir(os.path.join("../data/processed/",exp))]
    #add select option 
    exp_list.append("Create New Experiment")

    #find index of experiment name in list
    if st.session_state['experiment_name'] != "":
        exp_index = exp_list.index(st.session_state['experiment_name'])
    else:
        exp_index = 0

    #add streamlit dropdown for folder selection based on list of folders in processed folder
    experiment_name = st.selectbox("Select Experiment", exp_list, index=exp_index)
    if experiment_name == "Create New Experiment":
        #add streamlit select folder name click to browse    
        experiment_name = st.text_input("Experiment Name", "")

    if experiment_name != "Create New Experiment" and experiment_name != "":
        # create directory for experiment. first change experiment name to lowercase and replace spaces and special with underscores
        experiment_name = experiment_name.replace(" ", "_").replace(".", "_").replace("(", "_").replace(")", "_")
        experiment_dir = os.path.join(PROCESSED_DOCUMENTS_DIR, experiment_name)
        #create directory for experiment
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        st.session_state['experiment_dir'] = experiment_dir
        st.session_state['experiment_name'] = experiment_name

        run_sidebar()    

with st.expander("*Expand for Instructions*"):
    st.markdown("""
    ### Building Q&A Evaluation Sets (sidebar)
                
    1. Select an experiment from the dropdown menu or create a new experiment.
       - Further down the sidebar, one can add pdfs to the experiment or delete all pdf from the experiment.
       - You cannot delete an experiment, but you can delete all pdfs from an experiment.
    2. Select the list of LLM to generate Q&A pairs (selecting 2-3 reduces bias). 
       - **Note:** If you select more than one LLM, the summaries and themes will be built using the first LLM selected.
       - For building summaries, select an LLM that is less expensive do to high cost.
    3. Select the parsing method for the documents (applies to pfds only). **Preview input_documents.csv to see how parsing method affects what is extracted.**
       - **MAX_TOKENS**: The documents are split into chunks of 3500 tokens.
       - **spacing**: The documents are split based on spacing in the pdfs.
       - **font-change**: The documents are split based on font-change in the pdfs.
       - **timestamp**: The documents are primarily when a new timestamp is seen (used for transcripts).
       - **line**: The documents are split based on lines.
       - **page**: The documents are split by page.
       - **elements**: The documents are split based on elements in pdfs.
       - **Note**: For non-pdf documents, the parsing method is set to "MAX_TOKENS".
    4. Select Q&A pair types to generate. **Summary Q&A pairs are VERY expensive to generate.**
       - **Map Reduce Summary**: Generate Q&A pairs based on the map reduce summary of each document. Note these Q&A pairs are very expensive to generate.
       - **Refine Summary**: Generate Q&A pairs based on the refine summary of each document. Note these Q&A pairs are very expensive to generate.
       - **All Doc Semantic Search**: Generate Q&A pairs using LDA + LLM to create themes for each documents.
       - **Per Doc Chunk**: Generate Q&A pairs based on just grabbing individual chunks of text from the document. Note these chunks may be much bigger than embedding model chunks, but the idea is to generate relatively straight forward Q&A pairs.
       - **Grouped Doc Chunk**: Generate Q&A pairs based on grouping chunks of text disparate documents. This is intended to create Q&A pairs requiring a more challenging retrieval step (and LLM to combine disparate info in the context).
       - **Granular**: Generate Q&A pairs based on grabbing individual sentences from the document. This is intended to challenge the retrieval step (nd LLM) to grab very granular information.
    5. Click the button to "Build Docs and Summaries". This will create a csv files with the documents and summaries which can be previewed. **CHECK PARSING METHOD BEFORE RUNNING SUMMARIES**
    6. Click the button to "Build Q&A Evaluation Set" to generate Q&A pairs. **CHECK PARSING METHOD BEFORE RUNNING Q&A PAIRS**
       - This will create a csv files ending in "_qa.csv" for each question type.
       - "answer" is the answer to the question from initial Q&A generation.
       - "answer_full" is the answer to the question from re-asking the question to the LLM proving the original context. This is a more apple to apples comparison with your RAG pipeline.
    
    ### Running RAG Experiments
    1. Select an LLMs, embedding model, and chunking size to run your experiments.
       - A cross-product of your selections will be run so be careful with your selections.
       - You can run one at a time as well and results will be appended to the csv file.
    """)

if page == "Preview/Download Files":
    #find all csv in experiment directory and sort files by name and remove files starting with "Exp_"
    csv_files = sorted([f for f in os.listdir(st.session_state['experiment_dir']) if f.endswith(".csv") and not f.startswith("Exp_")])

    #add select option to preview a specific file
    this_csv_file = st.selectbox("Select .csv File", csv_files)
    if this_csv_file:
        #read qa file into dataframe
        this_df = pd.read_csv(os.path.join(st.session_state['experiment_dir'], this_csv_file))
        st.dataframe(this_df)
        # add streamlit button to download file
        st.download_button(label="Download File", data=this_df.to_csv(index=False), file_name=this_csv_file, mime="text/csv")

    #add upload button to csv files
    st.header("Modify or Add to Evaluation Set")
    uploaded_file = st.file_uploader("Upload .csv File ending in '_qa.csv' to modify or add to existing evaluation set.", type=["csv"])
    #save uploaded file to experiment directory
    if uploaded_file:
        #read uploaded file into dataframe
        uploaded_df = pd.read_csv(uploaded_file)
        #save uploaded file to experiment directory
        uploaded_df.to_csv(os.path.join(st.session_state['experiment_dir'], uploaded_file.name), index=False)
    
elif page == "Run Experiments":
    analyzer_window()


# elif app_task == "General Chatbot":
#     run_full_chatbot()


