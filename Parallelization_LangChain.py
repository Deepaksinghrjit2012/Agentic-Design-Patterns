import os
import asyncio
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable,RunnableParallel
load_dotenv()
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)


summerization_chain:Runnable=(
    ChatPromptTemplate.from_messages(
        [
            ("system","you are an agent that will summerize a topic concisely:"),
            ("user","{topic}")
        ]
    )
    |llm
    |StrOutputParser()
)

questions_chain: Runnable=(
    ChatPromptTemplate.from_messages(
        [
            ("system","you are an agent that will generate three intresting questions about the following topic:"),
            ("user","{topic}")
        ]
    )
    |llm
    |StrOutputParser()
)


terms_chain:Runnable=(
    ChatPromptTemplate.from_messages(
        [
            ("system","you are an agent that will identify 5-10 key terms from the following topic, seperated bt commas:"),
            ("user","{topic}")
        ]
    )
    |llm
    |StrOutputParser()
)


map_chain=RunnableParallel(
    {
        "summary": summerization_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "topic":RunnablePassthrough()
    }
)

systhesis_prompt=ChatPromptTemplate.from_messages(
    [
        ("system","""Based on the followinf information:
         Summary:{summary}
         Related Questions:{questions}
         Key Terms:{key_terms}
         Synthesize a comprehensive answer."""),
        ("user","Origina topic: {topic}")
    ])
full_parallel_chain=map_chain|systhesis_prompt|llm|StrOutputParser()


async def run_parallel_chain(topic:str)->None:
    response= await full_parallel_chain.ainvoke(topic)
    print(response)

if __name__=="__main__":
    test_topic="the history of space exploration"
    asyncio.run(run_parallel_chain(test_topic))

