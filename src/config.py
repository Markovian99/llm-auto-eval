# Define some dummy data
MODELS = ["OpenAI: gpt-3.5-turbo-1106", "OpenAI: gpt-3.5-turbo", "OpenAI: gpt-4", "OpenAI: gpt-4-1106-preview", "OpenAI: gpt-4-32k", 
          "Anthropic: claude-3-opus-20240229","Google: gemini-pro", "HF: Llama-2-70B"]
TEMPERATURE = 0
MAX_TOKENS = 3500 # Enough for response if 4k context

NO_RETRIEVER_NAME = "NO RETRIEVER - LLM ONLY"
EMBEDDING_MODELS = ["BAAI/bge-base-en-v1.5", "all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-mpnet-base-v2","thenlper/gte-base", NO_RETRIEVER_NAME]

USE_AZURE = False

ANSWER_EVAL_MODEL = EMBEDDING_MODELS[0]

PARSE_METHOD = ["MAX_TOKENS", "spacing", "font-change", "timestamp", "line", "page", "elements"]
CHUNK_SIZE = ["64,16", "128,32", "128,64", "256,64", "256,128", "512,128", "512,256", "1024,256"]

MIN_QUESTION_LENGTH = 10 #also used for min context size
MIN_QUESTION_SIMILARITY = 0.5 # minimum similarity between question and context to be considered a valid question. we set this low to not filter out questions that are similar to context 
MIN_ANSWER_SIMILARITY = 0.5 # minimum similarity between answer and answer_full to be considered a valid test

MAX_QA_PER_TEST = 4 #maximum number of questions to generate per test

APP_NAME = "LLM Pipeline Auto-Eval"

# make sure to include the trailing slash
PROCESSED_DOCUMENTS_DIR = "../data/processed/"
REPORTS_DOCUMENTS_DIR = "../data/reports/"

# we will use these to filter out the generated questions and answers. 
NO_RESPONSE_EXCEPTION = "Exception: OpenAI Response Issue"
BAD_FORMAT_EXCEPTION = "Exception: Invalid Q&A Format"
BAD_QUESTION_EXCEPTION = "Exception: Invalid Question"
BAD_ANSWER_SIMILARITY = "Exception: Answers too dissimilar"
BAD_CONTEXT = "Exception: Context is too short"

#PROMPT TEMPLATES

SUMMARY_QA_GEN = """
You will be provided with a summary or a list of themes from a document of set of documents. Please generate a question and answer pair from the provided context. Please include facts, named-entities, or subject matter in the question itself. DO NOT ASK GENERAL QUESTIONS ABOUT CONTEXT OR THEMES. An example is given below. Please follow the same format for your answer.
Context:\nJohn is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
Question:\nWhat is John's major at XYZ University and what is a summary of what John does with his time?
Answer:\nJohn is majoring in Computer Science at XYZ University and spends a significant amount of time studying and completing assignments.

Now, it's your turn. Please generate a question and answer pair. In the question, do not refer to the context and do not use the words 'document', 'context', or any similar words.
Context:\n{context}
Question:
"""  

BASIC_QA_GEN = """
Generate a very specific question and answer pair from the provided context. Please include facts, named-entities, or subject matter in the question. DO NOT ASK GENERAL QUESTIONS. An example is given below. Please follow the same format for your answer.

Context:
John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.

Question:
What is John's major at XYZ University?

Answer:
John is majoring in Computer Science at XYZ University.

Now, it's your turn. Using the follwong context, please generate a very specific (non-generic) question and answer pair following the pre-specified format above. In the question, do not refer to the context and do not use the words 'document', 'context', or any similar words.

Context:
{context}

Question:
"""

COMPLEX_QA_GEN = """
Generate a complex question and answer pair that requires some thought to answer the question. An example is given below. Please follow the same format for your answer.

Context:
John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.

Question:
Are the courses John is taking aligned with his stated major?

Answer:
John is majoring in Computer Science and is enrolled in Data Structures, Algorithms, and Database Management. Therefore, these course John is taking are aligned with his major.

Now, it's your turn. Please generate a question and answer pair requires at least 2 pieces of information from the given context. In the question, do not refer to the context and do not use the words 'document', 'context', or any similar words.

Context:
{context}

Question:
"""  

SMALL_CONTEXT_QA_GEN = """
Generate a specific question and answer pair from the provided context. DO NOT ASK GENERAL QUESTIONS. An example is given below. Please follow the same format for your answer.

Context:
John Smith is a student at XYZ University, majoring in Computer Science, and taking Data Structures, Algorithms, and Database Management. 

Question:
What is John Smith's major?

Answer:
John is majoring in Computer Science at XYZ University.

Now, it's your turn. Please generate a very specific (non-generic) question and answer pair. In the question, do not refer to the context and do not use the words 'document', 'sentence','context', or any similar words.

Context:
{context}

Question:
"""

COMPARISON_QA_GEN = """
Generate a question and answer pair that compares information from multiple entities in the provided context. An simple example is given below. Please follow the same format for your answer.

Context:
John is a student at XYZ University. He is pursuing a degree in Computer Science. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management.\n
Steve is a student at ABC University. He is pursuing a degree in Computer Science. He spends a significant amount of time studying, but often fails to complete assignments on time. He is enrolled in several courses this semester, including Operating Systems, Algorithms, and Artificial Intelligence. \n
Mark is a student at DEF University pursuing a degree in Mathematics. He enjoys walking around campus thinking about math problems. He is enrolled in several courses this semester, including Calculus, Linear Algebra, and Differential Equations.

Question:
What are the key differences in the coursework John and Steve are currently taking?

Answer:
John is enrolled in Data Structures, Algorithms, and Database Management and Steve is enrolled in Operating Systems, Algorithms, and Artificial Intelligence. Therefore, the key difference in the coursework John and Steve are currently taking is that only John is taking Data Structures and Database Management while only Steve is taking Artificial Intelligence.

Now, it's your turn. Please generate a question and answer pair that compares information from multiple entities in the given context. In the question, do not refer to the context and do not use the words 'document', 'context', or any similar words.

Context:
{context}

Question:
"""  

MULTIHOP_QA_GEN = """
Generate question and answer pair that requires at least 2 separate pieces of information from different sections of the provided context. An simple example is given below. Please follow the same format for your answer.

Context:
John is a student at XYZ University. He is pursuing a degree in Computer Science. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management.\n
Lucy is a student at XYZ University. She is pursuing a degree in Physics and Computer Science. She spends a significant amount of time studying and thinking about the universe. She is enrolled in Quantum Mechanics in the Physics Department and Algorithms in the Computer Science Department.\n
Steve is a student at ABC University. He is pursuing a degree in Computer Science. He spends a significant amount of time studying, but often fails to complete assignments on time. He is enrolled in several courses this semester, including Operating Systems, Algorithms, and Artificial Intelligence.\n
Mark is a student at DEF University pursuing a degree in Mathematics. He enjoys walking around campus thinking about math problems. He is enrolled in several courses this semester, including Calculus, Algorithms, Linear Algebra, and Differential Equations.

Question:
What is Mark's major and what course is he taking that likely aligns with a different major? 

Answer:
Mark's major is Mathematics. Although he is a Mathematics major, his Algorithms course likely aligns with Computer Science as Steve and Lucy are majoring in Computer Science and taking Algorithms. 

Now, it's your turn. Please generate a question and answer pair requires at least 2 pieces of information from different sections of the context. Please do not make a generic reference to the context and do not use the words 'document', 'context', or any similar words in the question.

Context:
{context}

Question:
"""  


# https://github.com/explodinggradients/ragas/tree/main/src/ragas/metrics

CONTEXT_PRECISION = """\
Please extract relevant sentences from the provided context that can potentially help answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information".  While extracting candidate sentences you're not allowed to make any changes to sentences from given context.

question:{question}
context:\n{context}
candidate sentences:\n"""  # noqa: E501


CONTEXT_RECALL_RA = """
Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not.
Think in steps and reason before coming to conclusion. 

context: Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist,widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.
answer: Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895 
classification
1. Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. The date of birth of Einstein is mentioned clearly in the context. So [Attributed]
2. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. The exact sentence is present in the given context. So [Attributed]
3. He published 4 papers in 1905. There is no mention about papers he wrote in given the context. So [Not Attributed]
4. Einstein moved to Switzerland in 1895. There is not supporting evidence for this in the given the context. So [Not Attributed]

context:{context}
answer:{ground_truth}
classification:
"""  # noqa: E501
