import os
import regex as re
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqGenerator.utils import read_file, get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqGenerator.MCQGenerator import generate_evaluate_chain
from src.mcqGenerator.logger import logging

# loading json response file
with open("./Response.json", "r") as file:
    RESPONSE_JSON = json.load(file)

# Creating a title for the app
st.title("MCQ Creator Application with LangChain")

# create a form using st.form
with st.form("user_inputes"):
    # file Upload
    uploaded_file = st.file_uploader("Upload a PDF or txt file")

    # input fields
    mcq_count = st.number_input("Numbers Of MCQs", min_value=3, max_value=15)

    # subject
    subject = st.text_input("Insert Subject Name", max_chars=20)

    # Quiz tone
    tone = st.text_input(
        "Complexity Level of Questions(For now use Simple only)", max_chars=20, placeholder="Simple"
    )

    # Add button
    button = st.form_submit_button("Create MCQs")

    # check if the button is clicked and all fields have input
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Wait While We Create MCQs..."):
            try:
                text = read_file(uploaded_file)
                # count tokens and the cost of API call
                with get_openai_callback() as cb:
                    respone = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON),
                        }
                    )
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                print(f"Error : {e}")
                st.text_area(f"Regenerate quiz by clicking the button")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(respone, dict):
                    print(respone.get('quiz',None))
                    try:
                        quiz = str(respone.get('quiz',None))                         
                        matches = re.findall(r"({.*})", quiz)
                        quiz = matches[0]  # Extract the first match (assuming one valid match)
                    except Exception as e:
                        print(f"Could't generate quiz: {e}")
                        st.text_area(f"Regenerate quiz by clicking the button")
                        quiz = None  # Set quiz to None to avoid further processing
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            print(table_data)
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                            # Display the review in a text box as  well
                            st.text_area(label="Review", value=respone["review"])
                        else:
                            st.text_area(f"Regenerate quiz by clicking the button")
                            st.error("error")

                else:
                    st.write(respone)
