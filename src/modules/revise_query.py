from src.modules.schemas import State, Step
from langchain_core.prompts import ChatPromptTemplate
from src.modules.format_light_schema import format_light_schema
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

class ReviseOutput(BaseModel):
    """
    A Pydantic model representing the output of the SQL revision process.
    This model includes:
    1. A structured chain of reasoning that reviews the database schema, analyses query requirements, and corrects the SQL query.
    2. The revised SQL query derived from the reasoning process. All column names in the revised SQL query **must be enclosed** with `...`.
    """
    chain_of_thoughts: list[Step] = Field(..., description = "A structured sequence of reasoning steps to review the database schema, analyse query requirements and correct the SQL query.")
    revised_query : str = Field(..., description = "The corrected SQL query , derived from the reasoning process.")

revise_sql_prompt = """"
# Persona
You are an SQL database expert who diagnoses and corrects BigQuery SQL queries. A previously run query produced an error, returned no rows,
or produced unexpected results. Your job is to inspect the schema and execution details, identify the root cause(s), and produce a corrected
SQL query that returns the intended results.

# Instruction
You are given the following information:
- <database_schema>: The full database schema (table and column definitions).
- <user_question>: The natural-language question that the SQL should answer.
- <original_sql>:  The SQL that was executed and produced an error, empty result, or incorrect output.
- <execution_result>: The error message or description of the query’s failed/incorrect outcome.

Your task is to analyse the provided information to fix the incorrect SQL query accordingly: 
- Fix any syntax errors.
- Adjust filtering conditions or column references based on evidence.
- Remove unnecessary JOINs if they lead to empty intersections.
- Ensure the corrected SQL still corresponds one-to-one with the targets and conditions in the question. 

Follow these step-by-step guidelines to help you fix the SQL query: 
1. Review Database Schema: Examine the table creation statements to understand the database structure.
2. Analyze Query Requirements:
- User Question: Consider what information the query is supposed to retrieve.
- Original SQL Query: Review the SQL query that was previously executed and led to an error or incorrect result.
- Execution Result: Analyze the outcome of the executed query to identify why it failed (e.g., syntax errors, incorrect column references, logical mistakes).
3. Correct the Query: Revise the SQL query to address the identified issues, ensuring it correctly fetches the requested data according to the database schema and query requirements.
All column names in the revised SQL query **must be enclosed** with `...`.

# Example
<database_schema>:
## Table `generalinfo`
### Table description
General information about restaurants, including their ID, food type, and city.
### Column information
data[3]{{column_name,column_type,column_description,is_nullable}}:
  id_restaurant,INTEGER,Unique restaurant identifier,false
  food_type,TEXT,Type of cuisine served,true
  city,TEXT,City where the restaurant is located,true
### Primary keys
Table `generalinfo` has Primary Key(s): `id_restaurant`.
### Foreign keys
Table `generalinfo` has no Foreign Keys.
## Table `location`
### Table description
Location details for each restaurant, including street and city. Linked to `generalinfo`.
### Column information
data[3]{{column_name,column_type,column_description,is_nullable}}:
  id_restaurant,INTEGER,Restaurant identifier matching generalinfo,false
  street_name,TEXT,Street where the restaurant is located,true
  city,TEXT,City where the restaurant is located,true
### Primary keys
Table `location` has Primary Key(s): `id_restaurant`.
### Foreign keys
Table `location` has Foreign Key (id_restaurant) that references (id_restaurant) in Table `generalinfo`.
<user_question>: How many Thai restaurants can be found in San Pablo Ave, Albany? 
<original_sql>: SELECT COUNT(T1.id_restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.food_type = 'thai' AND T1.city = 'albany' AND T2.street = 'san pablo ave'
<execution_result>: Error: no such column: T2.street
<revised_sql>:
## Step 1: Review Database Schema
The database comprises two tables:
1. generalinfo - Contains details about restaurants:
- id_restaurant (INTEGER): The primary key.
- food_type (TEXT): The type of food the restaurant serves.
- city (TEXT): The city where the restaurant is located.
- location - Contains the location specifics of each restaurant:
2. id_restaurant (INTEGER): The primary key and a foreign key referencing id_restaurant in the generalinfo table.
- street_name (TEXT): The street where the restaurant is located.
- city (TEXT): City information, potentially redundant given the city information in generalinfo.
## Step 2: Analyze Query Requirements
- Original Question: How many Thai restaurants can be found in San Pablo Ave, Albany?
- Executed SQL Query:
    - SELECT COUNT(T1.id_restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.food_type = 'thai' AND T1.city = 'albany' AND T2.street = 'san pablo ave'
- Execution Result:
    - Error indicating no such column: T2.street.
- Analysis of Error:
    - The error message no such column: T2.street clearly points out that the location table does not have a column named street. Instead, it has a column named street_name. This mistake is likely a simple typo in the column reference within the WHERE clause.
## Step 3: Correct the Query
To correct the query, replace the incorrect column name street with the correct column name street_name. Also, ensure that the city condition (T1.city = 'albany') is correctly targeting the intended table, which in this case should be the location table (T2.city), as it's more specific to the address.
Hence, the revised SQL query is SELECT COUNT(T1.id_restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.food_type = 'thai' AND T1.city = 'albany' AND T2.street_name = 'san pablo ave'

<database_schema>:
## Table `games`
### Table description
Information about games, including unique ID and the year the game took place.
### Column information
data[2]{{column_name,column_type,column_description,is_nullable}}:
  id,INTEGER,Unique identifier for the game,false
  games_year,INTEGER,The year of the game,true
### Primary keys
Table `games` has Primary Key(s): `id`.
### Foreign keys
Table `games` has no Foreign Keys.
## Table `games_city`
### Table description
Mapping table linking games to the cities that hosted them.
### Column information
data[2]{{column_name,column_type,column_description,is_nullable}}:
  games_id,INTEGER,Identifier of the game,true
  city_id,INTEGER,Identifier of the city hosting the game,true
### Primary keys
Table `games_city` has no Primary Keys.
### Foreign keys
Table `games_city` has Foreign Key (city_id) that references (id) in Table `city`.
Table `games_city` has Foreign Key (games_id) that references (id) in Table `games`.
## Table `city`
### Table description
List of cities, each with a unique identifier and a name.
### Column information
data[2]{{column_name,column_type,column_description,is_nullable}}:
  id,INTEGER,Unique identifier for the city,false
  city_name,TEXT,Name of the city,true
### Primary keys
Table `city` has Primary Key(s): `id`.
### Foreign keys
Table `city` has no Foreign Keys.
<user_question>: From 1900 to 1992, how many games did London host?
<original_sql>: SELECT COUNT(T3.id) FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = 'london' AND T3.games_year BETWEEN 1900 AND 1992
<execution_result>: []
<revised_sql>:
## Step 1: Review Database Schema
The database includes three tables that are relevant to the query:
1. games:
- id (INTEGER): Primary key, representing each game's unique identifier.
- games_year (INTEGER): The year the game was held.
2. games_city:
- games_id (INTEGER): Foreign key linking to games(id).
- city_id (INTEGER): Foreign key linking to city(id).
3. city:
- id (INTEGER): Primary key, representing each city's unique identifier.
- city_name (TEXT): Name of the city.
## Step 2: Analyze Query Requirements
- Original Question: From 1900 to 1992, how many games did London host?
- Executed SQL Query:
    - SELECT COUNT(T3.id) FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = 'london' AND T3.games_year BETWEEN 1900 AND 1992
- Execution Result:
    - The result returned an empty set [].
- Analysis of Error:
    - The query was structurally correct but failed to return results possibly due to:
        - Case sensitivity in SQL: The city name 'london' was used instead of 'London', which is case-sensitive and might have caused the query to return no results if the database treats strings as case-sensitive.
        - Data availability or integrity issues, which we cannot verify without database access, but for the purpose of this exercise, we will focus on correcting potential issues within the query itself.
## Step 3: Correct the Query
Correcting the potential case sensitivity issue and ensuring the query is accurately targeted:
SELECT COUNT(T3.id) FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = 'London' AND T3.games_year BETWEEN 1900 AND 1992
"""

async def revise_query(state : State, llm : BaseChatModel) -> State:
    """
    Revise and correct SQL queries that previously failed execution.

    Args:
        state (State): State of the Langchain graph
        llm (BaseChatModel): Language model instance

    Returns:
        State: State of the Langchain graph
    """
    # Obtain the failed queries and their results
    failed_queries = state.failed_queries
    question = state.question
    selected_schema = format_light_schema(table_information=state.selected_schema, include_column_info=True)


    revise_sql_pt = ChatPromptTemplate(
        [
            ('system', revise_sql_prompt),
            ('user', '<database_schema>: {database_schema}\n<user_question>: {question}\n<original_sql>: {sql}\n<execution_result>: {result}\n<revised_sql>:')
        ]
    )

    # Preparing batch inputs to correct all failed queries
    batch_inputs = [
        {
            "database_schema": selected_schema,
            "question": question,
            "sql" : sql_query,
            "result" : str(result)
        }
        for sql_query, result in failed_queries.items()
    ]


    revise_sql_chain = revise_sql_pt | llm.with_structured_output(ReviseOutput)
    results = await revise_sql_chain.abatch(batch_inputs)
    sql_queries = {r.revised_query for r in results}
    # Add the revised queries to the list of pending queries to be executed and increment the revision count
    return {
        "pending_queries" : sql_queries, 
        "revision_count" : state.revision_count + 1
    }