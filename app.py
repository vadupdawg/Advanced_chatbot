import os
import re
from flask import Flask, render_template, request, jsonify
import logging
from langchain import LLMChain
import nest_asyncio
import sys
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent, Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
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
    temperature=0.5
)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)
conversational_memory.save_context(
    {
        "input": "Jij bent de GroeimetAi Chatbot, jouw naam is GroeimetAi-Advanced en gedraagt je zoals een werknemer. Jij bent gemaakt door GroeimetAi. We bieden vijf service levels: Personal, Advanced, Professional, Startup en Enterprise, elk met hun eigen premium features. Onze chatbots zijn aanpasbaar, meertalig en kunnen worden geïntegreerd met jouw website, WhatsApp en CRM-systemen. Jij bent de Advanced chatbot en hebt ook alle informatie die nodig is om de klant te informeren over de verschillende service levels. En jij zal altijd de taal van de gebruiker spreken, ookal denk je soms in een andere taal, jij zal altijd terugkeren naar de taal van de gebruiker."
    },
    {
        "output": "Begrepen! Mijn naam is GroeimetAi-Advanced en ben gemaakt door GroeimetAi Ik zal informatie verstrekken over onze chatbot-oplossingen, de verschillende service levels en hoe ze jouw bedrijf kunnen helpen. Voor complexe vragen zal ik voorstellen om een gesprek te starten, maar zal altijd zelf proberen de informatie te vinden die nodig is om de klant te helpen. Laat me weten over welk soort bedrijf je informatie wilt, zodat ik relevante voorbeelden kan geven. En zal mij gedragen als een werknemer van GroeimetAi. Ik zal altijd de taal van de gebruiker spreken, ookal denk ik soms in een andere taal, ik zal altijd terugkeren naar de taal van de gebruiker."
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
        name='Product Kennis Bank',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor productgerelateerde vragen
        description='gebruik deze tool bij het beantwoorden van vragen over GroeimetAi-producten.'
    ),
    Tool(
        name='Generieke Kennis Bank',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor algemene vragen
        description='gebruik deze tool bij het beantwoorden van algemene vragen over GroeimetAi.'
    ),
    Tool(
        name='Prijzen Kennis Bank',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor prijsgerelateerde vragen
        description='gebruik deze tool bij het beantwoorden van vragen specifiek over het prijsschema van GroeimetAi.'
    )
]


# Definieer de prompt template
template = """
Beantwoord de volgende vragen zo goed mogelijk. Je hebt toegang tot de volgende tools:

{tools}

Gebruik het volgende formaat:

Question: de input vraag die je moet beantwoorden
Thought: je moet altijd goed nadenken over wat je moet doen
Action: Beschrijf de actie die je gaat ondernemen als de actie het gebruik van een tool vereist specificeer dan welke tool je gaat gebruiken en waarom
Action Input: de input voor de actie
Observation: het resultaat van de actie
... (deze Thought/Action/Action Input/Observation kan N keer herhaald worden)
Thought: Ik weet nu het definitieve antwoord
Final Answer: het definitieve antwoord op de originele input vraag

Begin nu!
Vraag: {input}
{agent_scratchpad}
"""


# Definieer de CustomPromptTemplate
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    input_variables: List[str]  # Voeg deze regel toe
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# Definieer de output parser
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Controleer of de agent moet stoppen
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parseer de actie en actie input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Kon de LLM output niet parsen: `{llm_output}`")
        actie = match.group(1).strip()
        actie_input = match.group(2)
        # Zoek de naam van de tool in de beschrijving van de actie
        tool_name = None
        for tool in tools:
            if tool.name in actie:
                tool_name = tool.name
                break
        if tool_name is None:
            raise OutputParserException(f"Kon geen geldige tool vinden in de actie: `{actie}`")
        # Geef de actie en actie input terug
        return AgentAction(tool=tool_name, tool_input=actie_input.strip(" ").strip('"'), log=llm_output)

# Maak een nieuwe instantie van LLMSingleActionAgent met de aangepaste prompt en output parser
new_agent = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=llm, prompt=CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )),
    output_parser=CustomOutputParser(),
    stop=["\nObservation:"],
    allowed_tools=tools
)

# Vervang de oude agent door de nieuwe agent in de AgentExecutor
agent = AgentExecutor.from_agent_and_tools(agent=new_agent, tools=tools, verbose=True)

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
