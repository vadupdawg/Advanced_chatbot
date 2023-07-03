import os
from flask import Flask, render_template, request, jsonify
import logging
import nest_asyncio
import sys
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

openai_api_key = os.getenv("OPENAI_API_KEY")

nest_asyncio.apply()

app = Flask(__name__)

app.debug = True

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

chat_histories = {}

# create a memory object
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Adding initial chat history
initial_message = "Hi ik ben de chatbot van GroeimetAi. GroeimetAi is een bedrijf dat middel en klein bedrijf helpt met het implementeren van AI mogelijkheden en gebruik te maken van deze geweldige trend. Ik kan je helpen met al je vragen over AI implementatie, ik ben echter alleen getrained op de data die OpenAI mij heeft gevoerd tot September 2021."
memory.chat_memory.messages.append((memory.ai_prefix, initial_message))

#create a sitemap loader
loader = WebBaseLoader("https://smaller-united-759551.framer.app/")
docs = loader.load()

#create a text splitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=8000, chunk_overlap=0)
documents = text_splitter.split_documents(docs)

#create an embeddings object
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#create a vectorstore object
vectorstore = Chroma.from_documents(documents, embeddings)

global qa
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(
        temperature=0, 
        model="gpt-3.5-turbo-16k", 
        openai_api_key=openai_api_key), 
        vectorstore.as_retriever(),
        condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=openai_api_key, verbose=True), 
        memory=memory,
        verbose=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data['user_message']

    result = qa({"question": query})

    return jsonify(chatbot_response=result["answer"])


if __name__ == '__main__':
    app.run(debug=True)
