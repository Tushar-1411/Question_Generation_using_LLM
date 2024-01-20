# Importing necessary libraries
import streamlit as st
from dotenv import load_dotenv
import PyPDF2 as pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAI

from src.exceptions import CustomException
from src.logger import logging
import sys

# Load Environment variables
load_dotenv()

# Instantiate your choice of llm for different outputs (Experiment with different temperature values, for this use case I'm using the default one).
llm_model = GoogleGenerativeAI(model="models/text-bison-001")


# Prompt for the llm to generate the list of questions based on the provided document
map_prompt = """You are a helpful AI bot that aids a teacher in question selection for standardized tests.
Below is information about the chapter.
Information will include contents about the chapter that a student is taught by a teacher.
Your goal is to generate question which can be asked to a student learning this chapter in a test.

% START OF INFORMATION ABOUT chapter:
{text}
% END OF INFORMATION ABOUT chapter:

Please respond with list of a few questions based on the topics above

YOUR RESPONSE:"""

# Takes input from the first prompt and outputs a list of questions
combine_prompt = """
You are a helpful AI bot that aids a teacher in question selection for standardized tests.
You will be given a list of potential questions that we can ask students.

Please consolidate the questions and return a list of questions.

% QUESTIONS
{text}
"""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])


# Text Splitter to split up the documents into smaller and meaningful chunks.
# (Can experiment with different chunksizes and overlaps values for better outputs)
class Text_Splitter:
    def __init__(self, chunk_size = 2000, chunk_overlap = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_text_splitter(self, chunk_size, chunk_overlap):
        try:
            logging.info("Initializing text splitter ...")
            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size = chunk_size, 
                                chunk_overlap = chunk_overlap
                                )
            logging.info("Text Splitter initiaized ...")
            return text_splitter
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def split_docs(self, document):  
        try:
            # Splitting the document into chunks
            text_splitter = self.load_text_splitter(self.chunk_size, self.chunk_overlap) 
            split_docs = text_splitter.split_documents(document)
            logging.info("Split Successfull...")
            return split_docs

        except Exception as e:
            raise CustomException(e, sys)
        
# PDF Loader class to read the in the pdf document
class PDFLoader:
    def __init__(self):
        pass       

    def read_pdf_document(self, document):
        try: 
            logging.info("Reading the pdf file...")
            loader = pdf.PdfReader(document)
            all_pages_doc = []
            for page_number in range(len(loader.pages)):
                text = loader.pages[page_number].extract_text()
                all_pages_doc.append(Document(page_content = text, metadata={"source": document.name, "page" : page_number + 1}))
            logging.info("File read successfully...")
            return all_pages_doc
        except Exception as e:
            raise CustomException(e, sys) 
        
# Main function where the chain is initialized and invoked for output (Outputs a list of questions).
class LLM_Chain:
    def __init__(self, llm):
        self.llm = llm

    def initialize_chain(self, llm):
        try:
            logging.info("Initializing Chain ...")
            chain = load_summarize_chain(llm,
                                chain_type="map_reduce",
                                map_prompt=map_prompt_template,
                                combine_prompt=combine_prompt_template,
                                #verbose=True
                                )
            logging.info("Chain initialized successfully ...")
            return chain
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_chain(self, chain, input_documents):
        try:
            logging.info("Invoking the chain for processing ...")
            output = chain({"input_documents": input_documents})
            return output['output_text']
        except Exception as e:
            raise CustomException(e, sys)

# Final class for the initialization of other classes and run the respective functions
class Generate_Ques:
    def __init__(self):
        self.text_splitter = Text_Splitter()
        self.pdf_loader = PDFLoader()
        self.llm_chain = LLM_Chain(llm_model)
        # logging.info("All classes initialized successfully !!!")

    def generate_questions(self, pdf_file):
        try:
            # Read the file
            document = self.pdf_loader.read_pdf_document(pdf_file)
            # split the document
            splitted_docs = self.text_splitter.split_docs(document)
            # Pass the splitted documents to the llm chain for processing
            chain = self.llm_chain.initialize_chain(llm_model)
            question_list = self.llm_chain.run_chain(chain, splitted_docs)
            return question_list
        
        except Exception as e:
            raise CustomException(e, sys)
        
# Initialize the final class 
output_class = Generate_Ques()

### Code for streamlit app
# Page configuration
st.set_page_config(
    page_title="Question Generation App",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("Question Generation App")
st.write(
    "This app aids teachers in generating questions for unit tests using Langchain."
    "Users can upload the document of their chapters or units and it will generate the questions relevant to that document."
)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


if uploaded_file is not None:

    # Button to trigger question generation
    if st.button("Generate Questions"):
        # Processing the PDF and generating questions
        questions = output_class.generate_questions(uploaded_file)
        # Display the generated questions
        st.subheader("Generated Questions:")
        # for i, question in enumerate(questions, 1):
        st.write(questions)

# App footer
st.sidebar.markdown(
    """
    ## About
    
    This app uses Langchain to generate questions from a PDF document. Upload a PDF, click the 'Generate Questions' button, 
    and the app will provide you with a set of questions suitable for unit tests.
    """
)


