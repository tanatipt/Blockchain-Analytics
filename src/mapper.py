from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from typing_extensions import Any, Literal

llm_map = {
    "openai": ChatOpenAI
}

vectorstore_map  = {
    "opensearch" : OpenSearchVectorSearch
}

embedding_map= {
    "gemini" : GoogleGenerativeAIEmbeddings,
}


def get_class(map_type: Literal['llm', 'vectorstore', 'embedding'], name: str) -> Any:
    """
    Retrieves the class corresponding to the given mapping type and name.
    Args:
        map_type (Literal['llm', 'vectorstore', 'embedding']): Type of the mapping
        name (str): Name of the class to retrieve

    Raises:
        Exception: Mapping type does not exist
        Exception: Mapping name does not exist in mapping type

    Returns:
        Any: The class corresponding to the specified mapping type and name.
    """
    map_dict = {
        "llm" : llm_map,
        "vectorstore" : vectorstore_map,
        "embedding" : embedding_map
    }

    if map_type not in map_dict:
        raise Exception('ERROR: Mapping type does not exist')

    map_type_dict = map_dict[map_type]

    if name not in map_type_dict:
        raise Exception('ERROR: Mapping name does not exist in mapping type')
    
    cls = map_type_dict[name]

    return cls