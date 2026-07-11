import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
load_dotenv()

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)


def booking_handler(request: str)-> str:
    """Simulates the booking agent handling a request."""
    print("\n---Deligating to booking agent---")
    return f"Booking agent received the request: {request}. Booking confirmed!"
def info_handler(request: str)-> str:
    """Simulates the information agent handling a request."""
    print("\n---Deligating to information agent---")
    return f"Information agent received the request: {request}. Here is the information you requested!"
def unclear_handler(request: str)-> str:
    """Simulates the unclear agent handling a request."""
    print("\n---Deligating to unclear agent---")
    return f"Unclear agent received the request: {request}. Please clarify your request!"

cordinator_router_prompt=ChatPromptTemplate.from_messages([
        ("system","You are a routing agent that will route the request to the appropriate agent based on the request type. The request types are 'booking', 'information', and 'unclear'. If the request is related to booking, route it to the booking agent. If the request is related to information, route it to the information agent. If the request is unclear, route it to the unclear agent."),
        ("user","Route the following request to the appropriate agent: {request}")
]
)

cordinator_router_chain=cordinator_router_prompt | llm | StrOutputParser()

branches={
    "booking":RunnablePassthrough.assign(output=lambda x: booking_handler(x['request']['request'])),
    "information":RunnablePassthrough.assign(output=lambda x: info_handler(x['request']['request'])),
    "unclear":RunnablePassthrough.assign(output=lambda x: unclear_handler(x['request']['request'])),
}
delegation_branch=RunnableBranch(
    (lambda x: x['decision'].strip()== "booking", branches["booking"]),
    (lambda x: x['decision'].strip()== "information", branches["information"]),
    (lambda x: x['decision'].strip()== "unclear", branches["unclear"]),
    branches["unclear"]
)
coordinator_agent={
    "decision":cordinator_router_chain,
    "request":RunnablePassthrough()
} | delegation_branch | (lambda x: x['output'])
def main():
    request="I would like to book a flight to New York."
    result=coordinator_agent.invoke({"request":request})
    print(result)
if __name__=="__main__":
    main()