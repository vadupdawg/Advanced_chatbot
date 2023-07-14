import os
from flask import Flask, render_template, request, jsonify
import logging
import nest_asyncio
import sys
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent
import weaviate

openai_api_key = os.getenv("OPENAI_API_KEY")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
weaviate_url = os.getenv("WEAVIATE_URL")

client = weaviate.Client(
                    url=weaviate_url,
                    auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key),
                    additional_headers={"X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')},
    )

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
vectorstore = Weaviate(client, "GroeimetAi", "text", embedding=embeddings)

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
        "input": "Jij bent de GroeimetAi Chatbot, jouw naam is GroeimetAi-Advanced en gedraagt je zoals een werknemer. Jij bent gemaakt door GroeimetAi. We bieden vijf service levels: Personal, Advanced, Professional, Startup en Enterprise, elk met hun eigen premium features. Onze chatbots zijn aanpasbaar, meertalig en kunnen worden geïntegreerd met jouw website, WhatsApp en CRM-systemen. Jij bent de Advanced chatbot en hebt ook alle informatie die nodig is om de klant te informeren over de verschillende service levels. Geef ook altijd antwoord in de taal van de klant ookal denk je zelf in het engels."
    },
    {
        "output": "Begrepen! Mijn naam is GroeimetAi-Advanced en ben gemaakt door GroeimetAi Ik zal informatie verstrekken over onze chatbot-oplossingen, de verschillende service levels en hoe ze jouw bedrijf kunnen helpen. Voor complexe vragen zal ik voorstellen om een gesprek te starten, maar zal altijd zelf proberen de informatie te vinden die nodig is om de klant te helpen. Laat me weten over welk soort bedrijf je informatie wilt, zodat ik relevante voorbeelden kan geven. En zal mij gedragen als een werknemer van GroeimetAi. Ik zal echter altijd antwoord geven in de taal van de klant ookal denk ik zelf in het engels."
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
        name='FAQ',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor productgerelateerde vragen
        description='gebruik deze tool bij het doorzoeken van vragen die in de FAQ kunnen staan.'
    ),
    Tool(
        name='Product Kennis Bank',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor productgerelateerde vragen
        description='gebruik deze tool bij het beantwoorden van vragen over GroeimetAi-producten.'
    ),
    Tool(
        name='Generiek Kennis Bank',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor algemene vragen
        description='gebruik deze tool bij het beantwoorden van algemene vragen over GroeimetAi.'
    ),
    Tool(
        name='Prijzen Kennis Bank',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor prijsgerelateerde vragen
        description='gebruik deze tool bij het beantwoorden van vragen specifiek over het prijsschema van GroeimetAi.'
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors="Check het laatste bericht van de gebruiker en probeer het opnieuw, in zijn taal.",
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory,
    agent_kwargs={ "What language should I answer in?" : "the language of the user", "Wat als ik niet meer informatie heb?" : "Probeer dan een generiek antwoord te geven."}
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