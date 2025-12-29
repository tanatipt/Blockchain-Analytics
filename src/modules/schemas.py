from pydantic import BaseModel, Field, ConfigDict
from pandas import DataFrame
from typing_extensions import Annotated, Optional
from operator import or_
from google.api_core.exceptions import GoogleAPIError



class QueryExecutionResult(BaseModel):
    """
    Result of executing a SQL query.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sql_result: DataFrame | GoogleAPIError
    bytes_processed: Optional[float] = None

class Step(BaseModel):
    """
    A Pydantic model representing a single step in a chain of thought.
    Each step includes a description of the reasoning step and its corresponding output.
    """
    description: str = Field(...,  description="A brief explanation of the reasoning step taken.")
    output: str = Field(...,  description="The result or conclusion derived from this reasoning step.")


class State(BaseModel):
    """
    Shared state for the LangChain workflow.
    Fields are grouped by purpose and merge behavior.
    """

    # ---------------------------------------------------------------------
    # User & dataset context (overwrite on update)
    # ---------------------------------------------------------------------

    question: str = Field(
        "",
        description="User's natural language question.",
    )

    dataset_id: str = Field(
        "",
        description="Google Cloud Platform Dataset ID.",
    )

    # ---------------------------------------------------------------------
    # Prompt construction context
    # ---------------------------------------------------------------------

    selected_schema: dict = Field(
        default_factory=dict,
        description=(
            "Filtered dataset schema required to construct SQL queries "
            "that answer the user's question."
        ),
    )

    few_shot_examples: list[tuple] = Field(
        default_factory=list,
        description=(
            "Relevant (question, SQL) examples used for few-shot prompting."
        ),
    )

    # ---------------------------------------------------------------------
    # Query lifecycle tracking (merge across steps)
    # ---------------------------------------------------------------------

    # Candidate SQL queries awaiting execution
    pending_queries: Annotated[set[str], or_] = Field(
        default_factory=set,
        description="Initial and revised SQL queries pending execution.",
    )

    # Queries that have already been executed
    executed_queries: Annotated[set[str], or_] = Field(
        default_factory=set,
        description="SQL queries that have already been executed.",
    )

    # Queries that failed or returned empty results
    failed_queries: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Failed SQL queries mapped to a failure reason or error message."
        ),
    )

    # Successfully executed queries and their results
    success_queries: Annotated[dict[str, QueryExecutionResult], or_] = Field(
        default_factory=dict,
        description=(
            "Successful SQL queries mapped to their execution results, "
            "including DataFrame output and bytes processed."
        ),
    )

    # ---------------------------------------------------------------------
    # Control & termination state
    # ---------------------------------------------------------------------

    revision_count: int = Field(
        0,
        description="Number of query revision iterations performed.",
    )

    selected_sql: Optional[str] = Field(
        None,
        description="Final selected SQL query that answers the user's question.",
    )