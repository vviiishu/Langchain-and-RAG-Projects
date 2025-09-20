from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
from langchain_ollama import OllamaLLM  

# Create FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# Load Ollama model (make sure ollama is running locally)
llm = OllamaLLM(model="llama2")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} for a 5 year old child with 100 words"
)

# Chain prompt + model
chain = prompt | llm

# ✅ Explicitly define input/output types to avoid Pydantic error
chain = chain.with_types(input_type=dict, output_type=str)

# Add API route
add_routes(
    app,
    chain,
    path="/poem"
)

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
