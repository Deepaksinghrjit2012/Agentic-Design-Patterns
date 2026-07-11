import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

extraction_prompt=ChatPromptTemplate.from_template(
    "You are an extraction agent that will extract specification from the give text:\n\n{input_text}")

specification_prompt=ChatPromptTemplate.from_template(
    "you are a specification text conversion to a json format in given key only 'RAM','Memory' and 'Model' from given specifications \n\n{specification}"
)

specification= extraction_prompt | llm | StrOutputParser()


full_chain=(
    {"specification":specification}
    |specification_prompt
    |llm
    |StrOutputParser()
)

input_text="The new iPhone 14 Pro comes with a powerful A16 Bionic chip, 6GB of RAM, and storage options ranging from 128GB to 1TB. It features a Super Retina XDR display, Face ID, and runs on iOS 16."
result=full_chain.invoke({"input_text":input_text})
print(result)
