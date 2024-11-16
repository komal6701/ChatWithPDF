from langchain_community.vectorstores import Cassandra
from langchain_community.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings

#support for dataset retrieval with Hugging Face
from datasets import load_dataset

#with CassIO, the engine powering the Astra DB integeration in langchain
#you will initalize the DB connection
import cassio
from PyPDF2 import PdfReader
import openai
import os
from dotenv import load_dotenv
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN=os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID=os.getenv('ASTRA_DB_ID')

OPEN_API_KEY = os.getenv('OPENAI_API_KEY')

pdfreader=PdfReader('Path to the PDF')

from typing_extensions import Concatenate
#readtext from pdf
raw_text=''
for i, page in enumerate(pdfreader.pages):
  content=page.extract_text()
  if content:
    raw_text+=content

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

from langchain_openai import OpenAI
llm=OpenAI(open_api_key=OPEN_API_KEY)
embeddings=OpenAIEmbeddings(open_api_key=OPEN_API_KEY)

astra_vector_store=Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,

)

from langchain.text_splitter import CharacterTextSplitter
#we need to split the text using Character Text plit such that it should not increase token size
text_splitter=CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,

)
texts=text_splitter.split_text(raw_text)

astra_vector_store.add_texts(texts[:50])
print("Inserted %i headlines." % len(texts[:50]))
astra_vector_index=VectorStoreIndexWrapper(vectorstore=astra_vector_store)

first_question=True
while True:
    if first_question:
      query_text:input("\nEnter your question(or type 'quit' to exit): ").strip()
    else:
      query_text=input("\nWhat isypur next question (or type 'quit' to exit): ").strip()

    if query_text.lower()=="quit":
      break
    if query_text=="":
      continue

    first_question=False

    print("\nQuestion: \"%s\"" % query_text)
    answer=astra_vector_index.query(query_text, llm=llm).strip()
    print("ANSWER :\"%s\"\n" %answer)

    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
      print("     [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))


