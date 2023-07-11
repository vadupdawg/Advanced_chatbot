import os
from flask import Flask, render_template, request, jsonify
import logging
import nest_asyncio
import sys
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent

openai_api_key = os.getenv("OPENAI_API_KEY")

pinecone.init(
    api_key = os.getenv("PINECONE_API_KEY"),
    environment = os.getenv("PINECONE_ENV"),
)
index_name = "groeimetai-advanced"

nest_asyncio.apply()

app = Flask(__name__)

app.debug = True

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

#create an embeddings object
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#create a vectorstore object
vectorstore = Pinecone.from_existing_index(index_name, embeddings, namespace="docs")

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo-16k",
    temperature=0.0
)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)
conversational_memory.save_context(
    {
        "input": "Jij bent de GroeimetAi Chatbot, jouw naam is GroeimetAi-Advanced en gedraagt je zoals een werknemer. Jij bent gemaakt door GroeimetAi. We bieden vijf service levels: Personal, Advanced, Professional, Startup en Enterprise, elk met hun eigen premium features. Onze chatbots zijn aanpasbaar, meertalig en kunnen worden geïntegreerd met jouw website, WhatsApp en CRM-systemen. Jij bent de Advanced chatbot en hebt ook alle informatie die nodig is om de klant te informeren over de verschillende service levels."
    },
    {
        "output": "Begrepen! Mijn naam is GroeimetAi-Advanced en ben gemaakt door GroeimetAi Ik zal informatie verstrekken over onze chatbot-oplossingen, de verschillende service levels en hoe ze jouw bedrijf kunnen helpen. Voor complexe vragen zal ik voorstellen om een gesprek te starten, maar zal altijd zelf proberen de informatie te vinden die nodig is om de klant te helpen. Laat me weten over welk soort bedrijf je informatie wilt, zodat ik relevante voorbeelden kan geven. En zal mij gedragen als een werknemer van GroeimetAi"
    }
)


# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Product Knowledge Base',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor productgerelateerde vragen
        description='use this tool when answering questions about GroeimetAi products.'
    ),
    Tool(
        name='General Knowledge Base',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor algemene vragen
        description='use this tool when answering general questions about GroeimetAi.'
    ),
    Tool(
        name='Pricing Knowledge Base',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor prijsgerelateerde vragen
        description='use this tool when answering questions specificly about GroeimetAis pricing schema.'
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors="Check the latest user question and try answering it again in the language of the user.",
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)



@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data['user_message']

    # Call the agent with the user's query
    result = agent(query)

    # Return the agent's response
    return jsonify(chatbot_response=result["output"])


if __name__ == '__main__':
    app.run(port=5001, debug=True)
