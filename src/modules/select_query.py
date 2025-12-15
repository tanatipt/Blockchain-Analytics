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
2. **Compare the results**: Inspect differences in the execution outputs for signs of correctness or error: expected row counts, missing/extra columns, data anomalies, off-by-one issues, or rows that appear out of scope. Validate whether each result aligns with expectations given the schema and question.  
3. **Decide**: Pick the candidate that best answers the question. Return a single letter: **A** or **B**.
"""

def hash_dataframe_sha(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()


@traceable
def group_sql_by_result(success_queries):
    grouped = {}

    # Group SQLs by identical execution result
    for sql_query, sql_result in success_queries.items():
        hash_result = hash_dataframe_sha(sql_result)
        if hash_result not in grouped:
            grouped[hash_result] = []
        grouped[hash_result].append(sql_query)

    # Randomly pick one SQL per group
    selected_group = {
        sql_result: random.choice(sql_list)
        for sql_result, sql_list in grouped.items()
    }

    return selected_group


@traceable
async def compute_tournament_scores(
    question : str,
    selected_schema : str,
    sql_group : dict,
    success_queries : dict,
    llm : BaseChatModel,
):
    # Initialize scores
    tournament_scores = {sql_query: 0 for sql_query in sql_group.values()}

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
                "<candidate_sql_b>: {candidate_sql_b}\n"
                "<execution_result_b>: {execution_result_b}\n"
                "<selected_candidate>:"
            ),
        ]
    )

    select_sql_chain = select_sql_pt | llm.with_structured_output(SelectionOutput)

    sql_candidates = list(tournament_scores.keys())
    sql_pairs = list(combinations(sql_candidates, 2))

    batch_inputs = []
    for sql_a, sql_b in sql_pairs:
        batch_inputs.append(
            {
                "database_schema": selected_schema,
                "question": question,
                "candidate_sql_a": sql_a,
                "execution_result_a": success_queries[sql_a].to_toon(),
                "candidate_sql_b": sql_b,
                "execution_result_b": success_queries[sql_b].to_toon(),
            }
        )

    results = await select_sql_chain.abatch(batch_inputs)

    for (sql_a, sql_b), result in zip(sql_pairs, results):
        if result.selected_candidate == "A":
            tournament_scores[sql_a] += 1
        elif result.selected_candidate == "B":
            tournament_scores[sql_b] += 1

    return tournament_scores


async def select_query(state : State, llm : BaseChatModel):
    question = state.question
    success_queries = state.success_queries
    selected_schema = format_light_schema(table_information=state.selected_schema, include_column_info=True)
    sql_group = group_sql_by_result(success_queries)

    if len(sql_group) == 1:
        return {"selected_sql": next(iter(sql_group.values()))}
    
    tournament_scores = await compute_tournament_scores(
        question=question,
        selected_schema=selected_schema,
        sql_group=sql_group,
        success_queries=success_queries,
        llm=llm,
    )

    best_sql = max(tournament_scores, key=tournament_scores.get)
    return {"selected_sql" : best_sql}


    

