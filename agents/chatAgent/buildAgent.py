from typing import TypedDict,List,Union
from langgraph.graph import START,END,StateGraph
from langchain_core.messages import HumanMessage,AIMessage
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_core.tools import tool


llm = ChatOllama( model=  'qwen2.5-coder:3b' )

class AgentState(TypedDict):
    messages : List[Union[HumanMessage,AIMessage]]

    
def process( state: AgentState ) -> AgentState:
    """ This node is used to answer the user queries"""

    
    embeddings = OllamaEmbeddings(model = "qwen2.5-coder:3b")

    db = FAISS.load_local(
        "./agents/chatAgent/RAG_documents/vectorEmbeddings",
        embeddings,
        allow_dangerous_deserialization=True  # required sometimes for pickle
        )
    
    try:

        query = state["messages"][-1].content


        docs = db.similarity_search(query, k=1)
        docs_page_content = " ".join([d.page_content for d in docs])

        prompt =f"""you are an Agri AI assistant
            Answer the following question: {query}

            by searching the below contents for reference : {docs_page_content}

            or by using previous messages

            give only main keypoints in "one line"
            """
        
        state["messages"][-1].content = prompt

        response = llm.invoke( state["messages"] )

        state["messages"].append(AIMessage(content=response.content))
        print(f"\nAI : {response.content}")

        return state

    except Exception as e:

        print(f"Exception occured in RAG {e}")


def builder():
    graph = StateGraph(AgentState)

    graph.add_node("process",process)
    graph.add_edge(START,"process")
    graph.add_edge("process",END)

    agent = graph.compile() 

    return agent


if __name__ == "__main__" :

    conversationHistory = []
    conversationHistory.append( HumanMessage(content="what is my previous question"))

    agent = builder()

    result = agent.invoke({"messages" : conversationHistory})

    print(result["messages"][-1].content)

    conversationHistory = result["messages"]
