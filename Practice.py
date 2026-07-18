import io
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough , RunnableParallel,Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import asyncio
load_dotenv()

llm=ChatGoogleGenerativeAI(
    model=os.getenv("Deployment_Model"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

question_chain:Runnable=(
    ChatPromptTemplate.from_messages(
    [
        ("system","you are an assistant make question the topic given below"),
        ("user","{topic}")
    ]
    )
    |llm
    |StrOutputParser()
)

summerization_chain:Runnable=(
    ChatPromptTemplate.from_messages(
        [
            ("system","you are an summerization agent who crete a summery on the given topic below"),
            ("user","{topic}")

        ]
    )
    |llm
    |StrOutputParser()

)

map_prompt=RunnableParallel(
    {
        "summary":summerization_chain,
        "questions":question_chain,
        "topic":RunnablePassthrough()
    }
)

systhasyzed_prmpt=ChatPromptTemplate.from_messages(
    [
        ("system","""Based on the followinf information:
         Summary:{summary}
         Related Questions:{questions}
         Synthesize a comprehensive answer."""),
         ("user","Origina topic:{topic}")
    ]
)

full_parallel_chain=map_prompt|systhasyzed_prmpt|llm|StrOutputParser()

async def run_parallel(topic:str)->None:
    responce= await full_parallel_chain.ainvoke(topic)
    print(responce)

if __name__=="__main__":
    topic="Indian History"
    asyncio.run(run_parallel(topic))


