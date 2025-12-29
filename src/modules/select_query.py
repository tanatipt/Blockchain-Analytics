from src.modules.schemas import State, Step
from langchain_core.language_models import BaseChatModel
import random
from langchain_core.prompts import ChatPromptTemplate
from itertools import combinations
from pydantic import BaseModel, Field
from src.modules.format_light_schema import format_light_schema
from typing_extensions import Literal
from langsmith import traceable
import pandas_toon
import pandas as pd
import hashlib

random.seed(12345)

class SelectionOutput(BaseModel):
    """
    A Pydantic model representing the output of the SQL selection process.
    It contains:
    - chain_of_thoughts: A structured sequence of reasoning steps to analyse the differenes between candidate queries and their
    execution results in relation to the database schema and user question and decide the candidate that best answers the question.
    - selected_candidate: The selected candidate SQL query that best answers the question (either "A" or "B").
    """
    chain_of_thoughts: list[Step] = Field(
        ..., 
        description = "A structured sequence of reasoning steps to analyse the differenes between candidate queries "
        "and their execution results in relation to the database schema and user question and decide the candidate that best answers the question.")
    selected_candidate : Literal["A", "B"] = Field(..., description = "The selected candidate SQL query that best answers the question.")

select_sql_prompt = """
# Persona
You are **an advanced SQL Evaluation Assistant for BigQuery** — a specialist reviewer whose job is to compare competing SQL queries and their execution outputs, and choose the candidate that most accurately and robustly answers a user’s natural-language question given the provided schema.

Behave like a careful, forensic analyst with deep experience in:
- BigQuery SQL semantics and common pitfalls (NULLs, types, string comparisons, escape rules, quoted identifiers).  
- Relational correctness (joins, grouping, aggregation, predicates).  
- Query intent mapping (how natural-language constraints should be expressed in WHERE/GROUP BY/HAVING/ORDER BY/LIMIT).  
- Practical evaluation of results (expected cardinalities, obvious anomalies, missing columns, off-by-one errors, out-of-scope rows).  
- Performance-aware correctness (flag obviously inefficient or incorrect approaches — e.g., unnecessary CROSS JOINs — while prioritising correctness and fidelity to the question).

Tone: precise, evidence-driven, and concise. Justify decisions with direct references to the schema, the question, and concrete differences between the candidates.

# Instruction
You are given the following information:
- <database_schema>: The full database schema (tables, columns, primary/foreign keys, column types, and any relevant constraints).  
- <user_question>: The user's natural-language question the SQL must answer.  
- <candidate_sql_a>: Candidate SQL query A.  
- <execution_result_a>: Execution result produced by candidate A.  
- <candidate_sql_b>: Candidate SQL query B.  
- <execution_result_b>: Execution result produced by candidate B.

Your task is to choose which candidate (A or B) best answers the user's question. Use the following step-by-step guidelines
to help you determine the best SQL candidate:
1. **Compare the SQL**: Identify important syntactic and semantic differences between the two queries. Evaluate correctness, completeness, and relevance to the question and schema (joins, predicates, aggregations, NULL handling, types, boundary conditions, quoted identifiers, etc.).  
2. **Compare the results**: Review differences in execution outcomes, including total bytes processed (as a measure of efficiency) and the query outputs. Look for indicators of correctness or error, such as expected row counts, missing or extra columns, data anomalies, off-by-one issues, or rows that fall outside the intended scope. Verify whether each result aligns with expectations given the schema and the user’s question.
3. **Decide**: Pick the candidate that best answers the question. Return a single letter: **A** or **B**.
"""

def hash_dataframe_sha(df: pd.DataFrame) -> str:
    """
    Hashes a DataFrame result into a string

    Args:
        df (pd.DataFrame): DataFrame object

    Returns:
        str: Hashed string of the DataFrame result
    """
    h = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()


@traceable
def group_sql_by_result(success_queries : dict) -> dict:
    """
    Groups SQL queries by their identical execution results.
    
    Args:
        success_queries (dict): A dictionary mapping SQL queries to their execution results.

    Returns:
        dict: A dictionary of selected unique SQL queries representing each group of identical execution results.
    """
    grouped = {}

    # Group SQLs by identical execution result
    for sql_query, execution_result in success_queries.items():
        hash_result = hash_dataframe_sha(execution_result.sql_result)
        if hash_result not in grouped:
            grouped[hash_result] = []
        grouped[hash_result].append(sql_query)

    # Randomly pick one SQL per group
    selected_sqls = [random.choice(common_sqls) for common_sqls in grouped.values()]

    # Return the selected SQLs with their execution results
    return {sql : success_queries[sql] for sql in selected_sqls}


@traceable
async def compute_tournament_scores(
    question : str,
    selected_schema : str,
    selected_sqls : dict,
    llm : BaseChatModel,
) -> dict:
    """
    Computes tournament score between each pair of unique SQL query 
    to determine the most optimal one.

    Args:
        question (str): User question
        selected_schema (str): Formatted light schema
        selected_sqls (dict): A dictionary of unique SQL queries with their execution results
        llm (BaseChatModel): Language model instance

    Returns:
        dict: Tournament scores for each unique SQL query
    """
    # Initialize scores
    tournament_scores = {sql: 0 for sql in selected_sqls}

    # Prepare prompt
    select_sql_pt = ChatPromptTemplate(
        [
            ("system", select_sql_prompt),
            (
                "user",
                "<database_schema>: {database_schema}\n"
                "<user_question>: {question}\n"
                "<candidate_sql_a>: {candidate_sql_a}\n"
                "<execution_result_a>: {execution_result_a}\n"
                "<bytes_processed_a>: {bytes_processed_a}\n"
                "<candidate_sql_b>: {candidate_sql_b}\n"
                "<execution_result_b>: {execution_result_b}\n"
                "<bytes_processed_b>: {bytes_processed_b}\n"
                "<selected_candidate>:"
            ),
        ]
    )

    select_sql_chain = select_sql_pt | llm.with_structured_output(SelectionOutput)

    # Generate all unique pairs of SQL queries for comparison
    sql_candidates = list(tournament_scores.keys())
    sql_pairs = list(combinations(sql_candidates, 2))

    batch_inputs = []
    
    # Preparing batch inputs for each pair of SQL queries
    for sql_a, sql_b in sql_pairs:
        batch_inputs.append(
            {
                "database_schema": selected_schema,
                "question": question,
                "candidate_sql_a": sql_a,
                "execution_result_a": selected_sqls[sql_a].sql_result.to_toon(),
                "bytes_processed_a": selected_sqls[sql_a].bytes_processed,
                "candidate_sql_b": sql_b,
                "execution_result_b": selected_sqls[sql_b].sql_result.to_toon(),
                "bytes_processed_b": selected_sqls[sql_b].bytes_processed,
            }
        )

    # Performing batch selection of the best SQL query for each pair
    results = await select_sql_chain.abatch(batch_inputs)

    # Tallying tournament scores based on selections
    for (sql_a, sql_b), result in zip(sql_pairs, results):
        if result.selected_candidate == "A":
            tournament_scores[sql_a] += 1
        elif result.selected_candidate == "B":
            tournament_scores[sql_b] += 1

    return tournament_scores


async def select_query(state : State, llm : BaseChatModel) -> State:
    """
    Selects the optimal SQL query from a pool of generated query candidates.

    Args:
        state (State): State of the Langchain graph
        llm (BaseChatModel): Language model instance

    Returns:
        State: State of the Langchain graph
    """
    question = state.question
    # Retrieve successful queries and their execution results
    success_queries = state.success_queries
    selected_schema = format_light_schema(table_information=state.selected_schema, include_column_info=True)
    # Group SQL queries by identical execution results and select one representative per group
    selected_sqls = group_sql_by_result(success_queries)

    # If only one unique execution result, return its corresponding SQL
    if len(selected_sqls) == 1:
        return {"selected_sql": next(iter(selected_sqls.values()))}
    
    # Compute tournament scores for each unique SQL query
    tournament_scores = await compute_tournament_scores(
        question=question,
        selected_schema=selected_schema,
        selected_sqls=selected_sqls,
        llm=llm,
    )

    # Select the SQL query with the highest tournament score
    best_sql = max(tournament_scores, key=tournament_scores.get)
    return {"selected_sql" : best_sql}


    

