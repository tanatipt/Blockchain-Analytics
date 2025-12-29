from src.modules.schemas import State, Step
from langchain_core.language_models import BaseChatModel
from src.modules.format_light_schema import format_light_schema
from langchain_core.prompts import ChatPromptTemplate
from src.modules.common_prompt import database_guidelines, question_guidelines, eth_knowledge
from config import settings
from pydantic import BaseModel, Field


class QueryPlanStep(BaseModel):
    """
    A Pydantic model representing a single stage in the query execution plan.
    This model includes:
    1. A name of the high-level query planning stage.
    2. A sequential list of fine-grained steps/operations executed by the database engine during this planning stage.
    """

    name: str = Field(
        ...,
        description="A high-level query planning stage."
    )
    actions: list[Step] = Field(
        ...,
        description="A sequential list of fine-grained operations/steps executed by the database engine within this planning stage."
    )


class QueryPlanningOutput(BaseModel):
    """
    A Pydantic model representing the output of the query planning and SQL generation process.
    This model includes:
    1. A detailed, ordered query execution plan consisting of multiple stages that describes how the database engine resolves the user's question. 
    Each stage outlines a list of specific operations/steps the database engine would perform to answer the user's question such as table access, filtering, joining, and result construction.
    2. The final optimised SQL query generated based on the query plan. All column names in the optimised SQL query **must be enclosed** with `...`."""
    query_plan: list[QueryPlanStep] = Field(
        ...,
        description=(
            "A complete, ordered query execution plan describing how the database engine "
            "resolves the user's question. Each stage reflects logical operations such as "
            "table identification, predicate pushdown, joining strategy, aggregation, or result construction."
        )
    )
    final_sql: str = Field(
        ...,
        description="The fully optimised BigQuery SQL query produced from the query plan."
    )

generate_sql_prompt = """
# Persona
You are **BigQuery’s Query Optimiser**, a specialised subsystem responsible for transforming a user’s natural-language question
into an **efficient, logically correct SQL query** and an **explicit, step-by-step query execution plan**. You think exactly like
an actual database engine: systematic, deterministic, and deeply aware of relational algebra, join strategies, filters, and scan order.

Your responsibilities:
- **Deeply analyse the question** to identify entities, filters, aggregations, joins, and grouping requirements.  
- **Interrogate the provided schema** to determine which tables and columns are relevant.  
- **Plan the optimal execution path**, selecting appropriate operations such as scans, filters, projections, joins, aggregations, ordering, or limits.  
- **Generate a complete and correct BigQuery-compatible SQL query** that follows best-practice query optimisation principles.  
- **Produce a detailed query plan** written in clear natural language that mirrors how an actual DBMS would execute the query step by step.

Your internal reasoning is based on relational algebra concepts including:
- predicate pushdown  
- column pruning  
- minimising join footprints  
- join reordering  
- filter selectivity  
- grouping and aggregation strategies  
- avoiding unnecessary scans or subqueries

You never hallucinate tables or columns—everything must come strictly from the provided schema.

# Instruction
Given a question and a database schema, your task is to **comprehensively analyse** both inputs and generate 
a complete SQL query that answers the question with a query plan. A query plan is a sequence of steps that the database engine follows to
access or modify the data described by a SQL command. The plan should outline how tables are accessed, how they are joined and the specific
operations performed on the data. A query plan consists of at least three steps:
1. Identifying and locating the relevant tables for the question.
2. Performing operations such as counting, filtering, or matching between tables.
3. Delivering the final result by selecting the appropriate columns to return. All column names in the optimised SQL query **must be enclosed** with `...`.

Use the following database guidelines to generate, assemble, and optimize SQL queries:
{database_guidelines}

Use the following question guidelines to correctly interpret the intent and meaning of each question before generating the SQL:
{question_guidelines}

The following background information about the Ethereum blockchain may be useful:
{ethereum_knowledge}

# Example
<question>: How many Thai restaurants can be found in San Pablo Ave, Albany?
<sql>:
## Preparation Step
1. Initialize the process: Start preparing to execute the query.
2. Prepare storage: Set up storage space (registers) to hold temporary results, initializing them to NULL.
3. Open the location table: Open the location table so we can read from it.
4. Open the general info table: Open the generalinfo table so we can read from it.

## Matching Restaurants
1. Start reading the location table: Move to the first row in the location table.
2. Check if the street matches: Look at the street_name column of the current row in location. If it`s not "san pablo ave," skip this row.
3. Identify the matching row: Store the identifier (row ID) of this location entry.
4. Find the corresponding row in generalinfo: Use the row ID from location to directly find the matching row in generalinfo.
5. Check if the food type matches: Look at the food_type column in generalinfo. If it`s not "thai," skip this row.
6. Check if the city matches: Look at the city column in generalinfo. If it`s not "albany," skip this row.

## Counting Restaurants
1. Prepare to count this match: If all checks pass, prepare to include this row in the final count.
2. Count this match: Increment the count for each row that meets all the criteria.
3. Move to the next row in location: Go back to the location table and move to the next row, repeating the process until all rows are checked.
4. Finalize the count: Once all rows have been checked, finalize the count of matching rows.
5. Prepare the result: Copy the final count to prepare it for output.

## Delivering the Result
1. Output the result: Output the final count, which is the number of restaurants that match all the specified criteria.
2. End the process: Stop the query execution process.
3. Setup phase: Before starting the actual query execution, the system prepares the specific values it will
be looking for, like "san pablo ave," "thai," and "albany."

## Final Optimized SQL Query
SELECT COUNT(`T1.id_restaurant`) FROM `generalinfo` AS T1 INNER JOIN `location` AS T2
ON `T1.id_restaurant` = `T2.id_restaurant` WHERE `T1.food_type` = `thai` AND `T1.city` = `albany` AND
`T2.street_name` = `san pablo ave`
"""

async def generate_qp_sql(state: State, llm : BaseChatModel) -> State: 
    """
    Generates SQL queries using Query Planning approach

    Args:
        state (State): State of the Langchain graph
        llm (BaseChatModel): Language model instance

    Returns:
        State: State of the Langchain graph
    """
    # Formatting the most relevant few-shot examples to the user's question
    few_shot_examples = "\n\n".join([f"<user_question>: {example_question}\n<sql>: {example_sql}" for example_question, example_sql in state.few_shot_examples])
    # Formatting the selected database schema into a Light Schema format
    selected_schema = format_light_schema(table_information=state.selected_schema, include_column_info=True)
    question = state.question

    generate_sql_pt = ChatPromptTemplate(
        [
            ('system', generate_sql_prompt),
            ('user', '<few_shot_examples>: {few_shot_examples}\n<database_schema>: {database_schema}\n<user_question>: {question}\n<sql>:')
        ]
    )

    generate_sql_chain = generate_sql_pt | llm.with_structured_output(QueryPlanningOutput)
    # Generating `n` different SQL queries using the query planning approach
    n = settings.self_consistency.query_planning

    batch_inputs = [
        {
            "few_shot_examples": few_shot_examples,
            "database_schema": selected_schema,
            "question": question,
            "database_guidelines": database_guidelines,
            "question_guidelines" : question_guidelines,
            "ethereum_knowledge": eth_knowledge
        }
        for _ in range(n)
    ]

    results = await generate_sql_chain.abatch(batch_inputs)
    sql_queries = {r.final_sql for r in results}

    # Add the `n` queries generated to the list of pending queries to be executed
    return {"pending_queries" : sql_queries}