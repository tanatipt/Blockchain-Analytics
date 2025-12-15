from src.modules.schemas import State, Step
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
import os
from config import settings


class SkeletonOutput(BaseModel):
    """
    A Pydantic model representing the full output of the question-skeleton extraction process.
    
    This model includes:
    1. A structured chain of reasoning that details how database literals and columns were identified.
    2. The final extracted question skeleton produced from that reasoning.
    """
    chain_of_thought: list[Step] = Field(
        ...,
        description=(
            "A structured sequence of reasoning steps used to analyze the user question, "
            "identify database literals and columns, and extract the final question skeleton."
        )
    )
    question_skeleton: str = Field(
        ...,
        description="The final extracted skeleton of the question, derived from the reasoning process."
    )

extract_skeleton_prompt = """
# Persona
You are a text-to-SQL expert who excels at analysing natural-language questions and abstracting their underlying logical structure,
with a strong ability to distinguish database literals from database columns.

# Instruction
Given a user question, your task is to **comprehensively analyse the question** by:
1. Identifying all database literals present in the question or its evidence. A database literal is a specific value belonging to a database column (e.g., "Japan", "Chinese Grand Prix") and is typically used in the WHERE clause for filtering.
2. Extracting a question skeleton. The skeleton represents the structural form of the question while omitting detailed database content such as entity names and column names.

When extracting the question skeleton, you **must**:
1. Replace **all database literals** with the database column to which they belong.
2. Replace **all database columns** with the placeholder <COLUMN>.
3. Preserve SQL-related keywords such as "average", "total", "difference", "count".

# Examples
<question>: Name movie titles released in year 1945. Sort the listing by the descending order of movie popularity.
<skeleton>: Name <COLUMN> released in <YEAR>. Sort the listing by the descending order of <COLUMN>.

<question>: In August of 1996, how many orders were placed by the customer with the highest amount of orders?
<skeleton>: In <MONTH> of <YEAR>, how many <COLUMN> were placed by the <COLUMN> with the highest amount of <COLUMN>?

<question>: Calculate the total production for each product which were supplied from Japan.
<skeleton>: Calculate the total <COLUMN> for each <COLUMN> which were supplied from <COUNTRY>.

<question>: Calculate the difference in the average number of low-priority orders shipped by truck in each month of 1995 and 1996.
<skeleton>: Calculate the difference in the average number of <COLUMN> in each month of <YEAR> and <YEAR>.
"""

async def select_examples(state : State, opensearch : OpenSearchVectorSearch, llm : BaseChatModel):
    question = state.question
    extract_skeleton_pt = ChatPromptTemplate(
        [
            ('system', extract_skeleton_prompt),
            ('user', '<question>: {question}\n<skeleton>:')
        ]
    )

    extract_skeleton_chain = extract_skeleton_pt | llm.with_structured_output(SkeletonOutput)
    response = await extract_skeleton_chain.ainvoke({"question" : question})
    question_skeleton = response.question_skeleton
    docs = await opensearch.asimilarity_search(query = question_skeleton, k = 5)
    return {"few_shot_examples" : [(doc.metadata['question'], doc.metadata['sql']) for doc in docs]}


if __name__ == "__main__":
    import os
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
    print(settings.LANGSMITH_API_KEY, settings.LANGSMITH_TRACING, settings.LANGSMITH_PROJECT)
    

    embedder = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")
    print(type(embedder))
    opensearch = OpenSearchVectorSearch(
        embedding_function=embedder,
        http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASS),
        **settings.opensearch
    )
    state = State(
        question = "What was the address that had the most transactions on 20-11-2025?", 
        dataset_id='bigquery-public-data.crypto_ethereum', 
        selected_schema = {},
        few_shot_examples=[]
    )

    llm = ChatOpenAI(model = "gpt-4.1-mini", temperature = 0.0, top_p = 0.0)
    few_shot_examples = select_examples(state, opensearch, llm)

    print(few_shot_examples)