from src.modules.schemas import State, Step
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field


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

async def select_examples(state : State, opensearch : OpenSearchVectorSearch, llm : BaseChatModel) -> State:
    """

    Select the most relevant few-shot text-to-SQL examples for a given question by using vector search over a vector database.

    Args:
        state (State): State of the Langchain graph
        opensearch (OpenSearchVectorSearch): OpenSearch vector search instance
        llm (BaseChatModel): Language model instance

    Returns:
        State: State of the Langchain graph
    """
    question = state.question
    extract_skeleton_pt = ChatPromptTemplate(
        [
            ('system', extract_skeleton_prompt),
            ('user', '<question>: {question}\n<skeleton>:')
        ]
    )
    # Extracting the question skeleton
    extract_skeleton_chain = extract_skeleton_pt | llm.with_structured_output(SkeletonOutput)
    response = await extract_skeleton_chain.ainvoke({"question" : question})
    question_skeleton = response.question_skeleton
    # Performing vector search using the question skeleton and retrieving top 5 relevant examples
    docs = await opensearch.asimilarity_search(query = question_skeleton, k = 5)
    return {"few_shot_examples" : [(doc.metadata['question'], doc.metadata['sql']) for doc in docs]}
