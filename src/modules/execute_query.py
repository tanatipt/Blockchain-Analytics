from src.modules.schemas import State, QueryExecutionResult
from google.cloud.bigquery import Client
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
from concurrent.futures import ThreadPoolExecutor
import asyncio
from config import settings

# Configure BigQuery job with a maximum bytes billed limit
job_config = bigquery.QueryJobConfig(
    maximum_bytes_billed=549755813888
)

def run_query_sync(query: str, client: Client) -> QueryExecutionResult:
    """
    Executes an SQL query using BigQuery client synchronously

    Args:
        query (str): SQL query string
        client (Client): BigQuery client instance

    Returns:
        QueryExecutionResult: Result of the query execution
    """
    try:
        job = client.query(query, job_config=job_config)
        return QueryExecutionResult.model_validate({
            "sql_result": job.to_dataframe(),
            "bytes_processed": job.total_bytes_processed,
        })
    except GoogleAPIError as exc:
        return QueryExecutionResult.model_validate({
            "sql_result": exc,
            "bytes_processed": None,
        })

async def run_query_async(
    executor: ThreadPoolExecutor,
    query: str,
    client: Client,
) -> QueryExecutionResult:
    """
    Executes an SQL query using BigQuery client asynchronously

    Args:
        executor (ThreadPoolExecutor): Executor for running the query
        query (str): SQL query string
        client (Client): BigQuery client instance

    Returns:
        QueryExecutionResult: Result of the query execution
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, run_query_sync, query, client)


def execute_router(state: State) -> str:
    """
    Router function to determine the next action based on query execution results.

    Returns:
        str: Next node to visit in the Langchain graph.
    """
    has_success = bool(state.success_queries)
    has_failed = bool(state.failed_queries)

    if state.revision_count >= settings.max_revision_count:
        return "select_query" if has_success else "__end__"

    if has_success and not has_failed:
        return "select_query"

    return "revise_query"

async def execute_query(state: State, client: Client) -> State:
    """
    Executes a list of pending queries that have not been executed in previous iterations

    Args:
        state (State): State of the Langchain graph
        client (Client): BigQuery client instance

    Returns:
        State: State of the Langchain graph
    """
    pending = state.pending_queries - state.executed_queries
    success_queries = {}
    failed_queries = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        tasks = {
            query: run_query_async(executor, query, client)
            for query in pending
        }

        results = await asyncio.gather(*tasks.values())

    for query, result in zip(tasks, results):
        sql_result = result.sql_result

        if isinstance(sql_result, GoogleAPIError):
            failed_queries[query] = sql_result.message
        elif sql_result.empty:
            failed_queries[query] = "[]"
        else:
            success_queries[query] = result

    return {
        "success_queries": success_queries,
        "failed_queries": failed_queries,
        "executed_queries": pending,
    }