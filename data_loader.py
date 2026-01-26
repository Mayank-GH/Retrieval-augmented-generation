from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

##openAI embedding model with 3072 dimensions
client = OpenAI()
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

splitter  = SentenceSplitter(chunk_size =1000,chunk_overlap=200)

def load_and_chunk_pdf(path:str): 
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d,"text",None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model = EMBED_MODEL,
        input = texts)
    return [item.embedding for item in response.data]

chunks = load_and_chunk_pdf("Max_Verstappen.pdf")
embed_texts(chunks)