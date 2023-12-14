from src.llm_helper import get_few_shot_db_chain

import streamlit as st

st.title('T Shirts Shop: RetailChat SQL Engine :👕 ')

question = st.text_input('Question: ')

if question:
    chain = get_few_shot_db_chain()
    answer = chain.run(question)
    st.header('Answer:')
    st.write(answer)

