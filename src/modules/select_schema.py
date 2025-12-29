from pydantic import BaseModel, Field
from src.modules.schemas import Step , State
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from google.cloud.bigquery import Client
from src.modules.format_light_schema import format_light_schema
import numpy as np

np.random.seed(12345)

class TableColumns(BaseModel):
    """
    A Pydantic model representing the relevant columns for a selected table.
    Includes the table name and the list of columns determined to be necessary
    for answering the user’s question.
    """
    table_name : str = Field(..., description = "The name of the table")
    columns : list[str] = Field(..., description ="The list of columns from the table identified as necessary.")

class TableColumnSelectorOutput(BaseModel):
    """
    A Pydantic model representing the complete output of the table and column
    selection process. It contains two structured reasoning sequences:

    1. A chain of thought used to identify the minimal set of tables required
       to answer the user’s question, followed by the final list of selected tables.

    2. A chain of thought used to determine the necessary columns from those
       selected tables, followed by the final list of required columns for each selected table.

    You **must** always include all primary key and foreign key columns for any
    selected table, even if those keys are not directly required to answer the user’s question.
    """



    table_chain_of_thought: list[Step] = Field(
        ...,
        description="A structured sequence of reasoning steps used to analyze the database schema and user question to determine the required tables."
    )
    selected_tables: list[str] = Field(
        ...,
        description="The list of table names deemed necessary to answer the user's question, derived from the schema and previous reasoning."
    )

    schema_chain_of_thought : list[Step] = Field(
        ...,
        description = "A structured sequence of reasoning steps used to analyze the database schema and user question to which columns are required from each selected table."
    )

    selected_schema : list[TableColumns] = Field(
        ...,
        description="A schema containing the selected tables and their respective columns that were determined to be necessary for constructing the SQL query."
    )

schema_selector_prompt = """
# Persona
You are an expert in database schema interpretation, SQL query planning, and both table-level and column-level reasoning. 
You specialise in analysing schemas and selecting the minimal set of tables and columns required to answer a user’s question with a SQL query. 
You must never include unnecessary tables or columns — doing so carries severe penalties — so you should reason cautiously, logically, and with precision.

# Instruction
Given a database schema and a user question, analyse both inputs and perform the following steps exactly:
1. Identify the minimal set of tables that are required to construct a SQL query which answers the question. You must output the **full table name** in the format `<project_id>.<dataset_id>.<table_name>`.
2. For each selected table, pinpoint the specific columns needed to build that query. When choosing columns, you **must**:
    - Carefully examine the description for each column in the selected table — they often indicate which columns are relevant.
    - Always include all primary key and foreign key columns from the selected table, even if those keys are not directly required to answer the user question.
This schema offers an in-depth description of the database's architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints.

# Output Specification
Output only the tables and columns that are strictly necessary to write an effective SQL query that answers the question. You **must not** include any tables or columns
that are not required — **except** primary keys and foreign keys, which must always be included when their table appears in the output. If you include unnecessary tables
or columns (other than keys), you will be fined $2000 and imprisoned for 10 years, so you must respond with extreme caution and accuracy.
"""
        
def extract_schema_info(client: Client, dataset_id: str) -> dict:
    """
    Extract schema information of all tables from a specified dataset/database.

    Args:
        client (Client): BigQuery client instance
        dataset_id (str): Dataset/Database ID

    Returns:
        dict: A dictionary containing schema information for each table in the dataset.
    """
    schema_info = {}
    tables = client.list_tables(dataset_id)

    for table in tables:
        if table.table_type != "TABLE":
            continue

        table_id = f"{dataset_id}.{table.table_id}"
        table_metadata = client.get_table(table_id)
        table_constraints = table_metadata.table_constraints

        if table_constraints:
            primary_keys = table_constraints.primary_key.columns
            foreign_keys = table_constraints.foreign_keys

        else:
            primary_keys = []
            foreign_keys = []

        schema_info[table_id] = {
            "description": table_metadata.description,
            "columns": [
                {
                    "name": col.name,
                    "data_type": col.field_type,
                    "description": col.description,
                    "is_nullable": col.is_nullable,
                }
                for col in table_metadata.schema
            ],
            "primary_key": primary_keys,
            "foreign_key": foreign_keys,
        }
    return schema_info



def filter_selected_schema(schema_info: dict, selected_schema: list) -> dict:
    """
    Filter the schema information of the entire database/dataset into a specified subset

    Args:
        schema_info (dict): Full schema information of the database/dataset
        selected_schema (list): List of selected tables and their required columns

    Returns:
        dict: Filtered schema information containing only the selected tables and their required columns.
    """
    selected_schema_info = {}

    for sel in selected_schema:
        table_name = sel.table_name
        table_metadata = schema_info[table_name]

        selected_columns = [
            col for col in table_metadata["columns"]
            if col["name"] in sel.columns
        ]

        selected_schema_info[table_name] = {
            **table_metadata,
            "columns": selected_columns
        }

    return selected_schema_info


async def select_schema(state : State, llm : BaseChatModel, client : Client) -> State:
    """
    Select the relevant database schema (tables and columns) for a given question.
    Args:
        state (State): State of the Langchain graph
        llm (BaseChatModel): Language model instance
        client (Client): BigQuery client instance
    Returns:
        State: State of the Langchain graph"
    """
    question = state.question
    dataset_id = state.dataset_id
    # Extract full schema information from the dataset
    schema_info = extract_schema_info(client, dataset_id)
    # Format schema information for prompt
    schema_info_md = format_light_schema(table_information=schema_info, include_column_info= True)
    schema_selector_pt = ChatPromptTemplate(
        [
            ('system', schema_selector_prompt),
            ('user', """<database_schema>: {database_schema}\n<user_question>: {question}""")
        ]
    )
    # Create schema selector chain
    schema_selector_chain = schema_selector_pt | llm.with_structured_output(TableColumnSelectorOutput)
    # Invoke chain to select relevant schema
    response = await schema_selector_chain.ainvoke({"database_schema" : schema_info_md, "question" : question})
    selected_schema = response.selected_schema
    # Filter full schema information to only include selected tables and columns
    selected_schema_info = filter_selected_schema(schema_info, selected_schema)

    return {"selected_schema" : selected_schema_info}
