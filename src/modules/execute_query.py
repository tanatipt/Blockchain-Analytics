from src.modules.schemas import State
from google.cloud.bigquery import Client
from google.api_core.exceptions import GoogleAPIError
from concurrent.futures import ThreadPoolExecutor
import asyncio
from config import settings


def run_query_sync(query: str, client : Client):
    """Regular blocking BigQuery call."""

    try:
        results = client.query(query).to_dataframe()
    except GoogleAPIError as e:
        results = e

    return results 

async def run_query_async(executor, query: str, client : Client):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, run_query_sync, query, client)


def execute_router(state : State):
    failed_sql_queries = state.failed_queries
    success_sql_queries = state.success_queries

    if state.revision_count >= settings.max_revision_count:
        if len(success_sql_queries) > 0:
            return "select_query"
        else:
            return "__end__"
    else:
        if len(failed_sql_queries) == 0 and len(success_sql_queries) > 0:
            return "select_query"
        else:
            return "revise_query"

async def execute_query(state : State, client : Client):
    pending_queries = state.pending_queries
    executing_queries = []
    success_queries = {}
    failed_queries = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        tasks = []

        for sql_query in pending_queries:
            if sql_query not in state.executed_queries:
                tasks.append(run_query_async(executor, sql_query, client))
                executing_queries.append(sql_query)
    
        results = await asyncio.gather(*tasks)

    for sql_query, result in zip(executing_queries, results):
        if isinstance(result, GoogleAPIError):
            failed_queries[sql_query] = result
        else:
            if result.empty:
                failed_queries[sql_query] = []
            else:
                success_queries[sql_query] = result

    return {
        "success_queries" : success_queries, 
        "failed_queries" : failed_queries,
        "executed_queries" : set(executing_queries)
    }