import json
from langchain_core.documents import Document
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings
import numpy as np
import os
from tqdm import tqdm



if __name__ == "__main__":
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

    data_path = "data/bird_train.json"
    bird_train = json.load(open(data_path, "r"))
    bulk_size = 500
    docs = []

    for example in bird_train:
        example_doc = Document(page_content=example["skeleton"], metadata={"question" : example["question"], "sql" : example["sql"]})
        docs.append(example_doc)

    num_batches = int(np.ceil(len(docs) / bulk_size))
    embedder = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")
    opensearch_vs = OpenSearchVectorSearch(
        embedding_function=embedder,
        http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASS),
        **settings.opensearch
    )

    for i in tqdm(range(num_batches)):
        chunks = docs[i * bulk_size : (i + 1) * bulk_size]
        _ = opensearch_vs.add_documents(chunks)

    

    
