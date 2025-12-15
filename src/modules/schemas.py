from pydantic import BaseModel, Field, ConfigDict
from pandas import DataFrame
from typing_extensions import Annotated, Optional
from operator import or_

class Step(BaseModel):
    """
    A Pydantic model representing a single step in a chain of thought.
    Each step includes a description of the reasoning step and its corresponding output.
    """
    description: str = Field(...,  description="A brief explanation of the reasoning step taken.")
    output: str = Field(...,  description="The result or conclusion derived from this reasoning step.")

class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    question : str = Field("", description = "User's question")
    dataset_id : str = Field("", description = "Google Cloud Platform Dataset ID")
    selected_schema : dict = Field(
        {}, 
        description = "A filtered schema that will be neccessary for constructing the SQL query to answer the user's question."
    )
    few_shot_examples: list[tuple] = Field(
        [], 
        description = "A list of relevant question-SQL examples to be used for few-shot prompting."
    )

    pending_queries : Annotated[set, or_] = Field({}, description = "A set of initial candidate SQL queries to be executed")
    executed_queries : Annotated[set, or_] = Field({}, description = "A set of candidate SQL queries that have been executed")
    failed_queries : dict = Field({}, description = "A set of candidate SQL queries that have failed to be executed or produced an empty result.")
    success_queries : Annotated[dict[str, DataFrame], or_] = Field({}, description = "A set of successful candiate SQL queries and its execution results.")
    revision_count : int = Field(0, description = "The number of times query revisions have been performed.")

    selected_sql : Optional[str] = Field(None, description = "The final selected SQL query to answer the user's question.")