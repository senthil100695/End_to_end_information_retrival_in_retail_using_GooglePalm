from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from dotenv import load_dotenv
from src.few_shots import few_shots, mysql_prompt
import os
load_dotenv()

def get_few_shot_db_chain():
    llm = GooglePalm(google_api_key = os.environ['GOOGLE_API_KEY'],temperature = 0.1)
    db_user = "root"
    db_password = "Root"
    db_host = "localhost"
    db_name = 'atliq_tshirts'
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info = 3)
    table_info = db.table_info

    #embedding
    embeddings = GooglePalmEmbeddings(google_api_key = os.environ['GOOGLE_API_KEY'])
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectordb = FAISS.from_texts(to_vectorize,embedding=embeddings,metadatas=few_shots)

    #vectordbsearch
    example_selector = SemanticSimilarityExampleSelector(
                            vectorstore = vectordb,
                            k=2)
    
    #Prompt(structure) to LLM
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",)
    

    #Using Few shot why means if LLM fails to query looki for the similar query and write the corret query

    few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=mysql_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info", "top_k"],) #These variables are used in the prefix and suffix)

    #SQL Chain
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)

    return chain

if __name__ == "__main__":
    chain = get_few_shot_db_chain()
    print(chain.run("tell me levi small size t-shirts price"))



    
