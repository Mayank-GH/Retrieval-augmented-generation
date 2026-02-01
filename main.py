import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api  
from inngest.experimental import ai 
from dotenv import load_dotenv
import uuid 
import os 
import datetime

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="RAG",
    api_base_url="http://localhost:8288",
    logger=logging.getLogger("uvicorn"),
    serializer=inngest.PydanticSerializer(),
    is_production=False,
)

@inngest_client.create_function(
    fn_id="RAG: ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf")
)
async def rag_ingest_pdf():
    return {"hello","world"}

app = FastAPI()

inngest.fast_api.serve(app,inngest_client,functions=[rag_ingest_pdf])