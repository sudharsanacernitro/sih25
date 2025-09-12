
from langchain_core.messages import HumanMessage,AIMessage
from agents.chatAgent.buildAgent import builder



def runAgent(conversationHistory , message):

    conversationHistory.append( HumanMessage(content=message))

    agent = builder()


    result = agent.invoke({"messages" : conversationHistory})

    

    conversationHistory = result["messages"]

    return (result["messages"][-1].content)


if __name__ == "__main__":
    runAgent([])


