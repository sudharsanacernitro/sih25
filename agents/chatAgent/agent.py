from langgraph.graph import START,END,StateGraph
from langchain_core.messages import HumanMessage,AIMessage
from typing import TypedDict,List,Union
from langchain_ollama import ChatOllama


llm = ChatOllama( model=  'qwen2.5-coder:3b' )


class AgentState(TypedDict):
    messages : List[Union[HumanMessage,AIMessage]]



def process( state: AgentState ) -> AgentState:
    """ This node is used to answer the user """

    response = llm.invoke( state["messages"] )

    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI : {response.content}")

    return state

graph = StateGraph(AgentState)

graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)

agent = graph.compile()


conversationHistory = []


while True:

    userInput = input("you :")

    conversationHistory.append( HumanMessage(content=userInput))

    result = agent.invoke({"messages" : conversationHistory})

    conversationHistory = result["messages"]

