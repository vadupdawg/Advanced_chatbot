import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import Flask, render_template, request, jsonify
import logging
from langchain import LLMChain
import nest_asyncio
import sys
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.callbacks import get_openai_callback
from typing import List, Union
import re
from qdrant_client import QdrantClient

openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

nest_asyncio.apply()

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

app.debug = True

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

#create an embeddings object
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# create a vectorstore object
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
collection_name = "groeimetai-final2"
qdrant = Qdrant(client, collection_name, embeddings=embeddings)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo-16k",
    temperature=0.0
)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
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
    retriever=qdrant.as_retriever()
)

tools = [
    Tool(
        name='Similarity search database',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor productgerelateerde vragen
        description='Gebruik deze tool om de database te doorzoeken naar de informatie waar de gebruiker naar op zoek is omtrent alles wat met GroeimetAi te maken heeft.'
    )
]

template = """Complete the objective as best you can. You have access to the following tools:

{tools}

Use the following format:
  
Question: the input question you must answer or a simple greeting
Thought: je moet altijd nadenken over wat te doen wanneer je een vraag krijgt
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: het resultaat van de action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: het finale antwoord op de input question

These were previous tasks you completed:



Begin!

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
    
output_parser = CustomOutputParser()
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    data = request.get_json()
    query = data['user_message']

    # Add the context manager here
    with get_openai_callback() as cb:
        # Call the agent with the user's query
        result = agent(query)

    # Print token usage information
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

    # Return the agent's response
    return jsonify(chatbot_response=result["output"])

if __name__ == '__main__':
    app.run(port=5001, debug=True)