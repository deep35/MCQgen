import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
# from src.mcqGenerator.utils import read_file, get_table_data
# from src.mcqGenerator.logger import logging

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2
import groq


# Load environment variables
load_dotenv()

# load api key
key = os.getenv("GROQ_API_KEY")

# Load LLM model
llm = ChatGroq(temperature=0.7, groq_api_key=key, model="mixtral-8x7b-32768")

# Template for the making QUIZ questions generating prompt
template = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quize of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs
{response_json}

"""
# Quiz generation prompt execution
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template,
)

# chaining both prompt and llm to generate quiz questions
quiz_chain = LLMChain(
    llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True
)

# Prompt for the Analyzing the generated quiz and improve it.
template2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity 
if the quiz questions which needs to be changed and change the tone such that it perfectly fits the students ability.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

# Prompt execution for the analyzing the quiz
quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"], template=template2
)

# chaining both prompt and llm to evaluate generated quiz questions
review_chain = LLMChain(
    llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True
)

# Sequencing both chains to generate and evaluate quiz questions
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True,
)


