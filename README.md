# ChatWithPDF
# PDF to Vector Store with OpenAI and Astra DB

## Description
This project demonstrates how to extract text from a PDF file, split the text into chunks, and store these chunks in a Cassandra vector store powered by Astra DB. It also shows how to query the vector store using OpenAI's language models to retrieve relevant information based on user queries.

## Requirements
To run this project, you need to install the following dependencies:
- `python-dotenv`
- `PyPDF2`
- `openai`
- `langchain_community`
- `cassio`
- `datasets`

You can install the required packages using:
```sh
pip install -r requirements.txt

1. Clone the repository
  git clone https://github.com/your-repository.git
  cd your-repository

2. Set the environment variables in .env file
  OPENAI_API_KEY=your_openai_api_key
  ASTRA_DB_APPLICATION_TOKEN=your_astra_db_application_token
  ASTRA_DB_ID=your_astra_db_id

3. Run the script
  python app.py
