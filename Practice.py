import os
import asyncio
import json
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

load_dotenv()

def extract_specification(input_text:str)->str:
    """Extracts specifications from the given input text."""
    # Simulate extraction logic
    return f"Extracted specifications from the input text: {input_text}"
def convert_to_json(specification:str)->str:
    """Converts the given specification text to JSON format."""
    # Simulate JSON conversion with a real structured payload.
    normalized = specification.lower()
    payload = {
        "RAM": "6GB" if "6gb" in normalized else None,
        "Model": "iPhone 14 Pro" if "iphone 14 pro" in normalized else None,
        "Storage": "128GB to 1TB" if "128gb" in normalized or "1tb" in normalized else None,
    }
    return json.dumps(payload, indent=2)


extraction_tool=FunctionTool(extract_specification)
JsonTool=FunctionTool(convert_to_json)

ExtractionAgent=Agent(
    name="extraction_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are an extraction agent. Extract the specification details from the given text "
        "and return only the extracted specification summary."
    ),
    description="This agent is to extract specification details from plain text",
    tools=[extraction_tool],
)
JsonAgent=Agent(
    name="Json_convertor_Agnet",
    model="gemini-2.5-flash",
    instruction=(
        "You are a JSON conversion agent. Take the extracted specification text and convert it "
        "into JSON with exactly these keys: RAM, Model, Storage. Return only valid JSON."
    ),
    description="This agent is to convert specification into json output",
    tools=[JsonTool],
    output_schema={
        "type": "object",
        "properties": {
            "RAM": {"type": ["string", "null"]},
            "Model": {"type": ["string", "null"]},
            "Storage": {"type": ["string", "null"]},
        },
    },
)

RouterAgent=Agent(
    name="Orchestrator_Agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are an orchestrator agent. First transfer the request to extraction_agent. "
        "After extraction_agent returns the extracted specification, transfer that result to "
        "Json_convertor_Agnet so the final response is valid JSON."
    ),
    description="Orchestrator Agent that will decide which agent to call",
    sub_agents=[JsonAgent,ExtractionAgent],
)


agent_runner=InMemoryRunner(RouterAgent)

if __name__ =="__main__":
    asyncio.run(
        agent_runner.session_service.create_session(
            app_name="InMemoryRunner",
            user_id="local-user",
            session_id="local-session",
        )
    )
    input_text="The new iPhone 14 Pro comes with a powerful A16 Bionic chip, 6GB of RAM, and storage options ranging from 128GB to 1TB. It features a Super Retina XDR display, Face ID, and runs on iOS 16."
    message=types.Content(
        role="user",
        parts=[types.Part.from_text(text=input_text)]
    )
    for event in agent_runner.run(
        user_id="local-user",
        session_id="local-session",
        new_message=message,

    ): 
        print(event)
