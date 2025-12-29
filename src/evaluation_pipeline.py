import pandas as pd
from langgraph.graph.state import CompiledStateGraph
from src.modules.schemas import QueryExecutionResult
from tqdm import tqdm
from src.graph_constructor import GraphConstructor
from src.baseline_constructor import BaselineConstructor
from google.cloud.bigquery import Client
from google.api_core.exceptions import GoogleAPIError
from src.modules.execute_query import run_query_sync
from langchain_community.callbacks import get_openai_callback
from src.mapper import get_class
from config import settings
import numpy as np
import os
import asyncio
import time

BYTES_TO_GB = 1024 ** 3


def calculate_row_match(predicted_row : tuple, ground_truth_row : tuple) -> tuple:
    """
    Calculate the number of matches and mismatches between two rows.

    Args:
        predicted_row (tuple): Row from the predicted dataframe
        ground_truth_row (tuple): Row from the ground truth dataframe

    Returns:
        tuple: Number of matches, number of elements only in predicted row,
               number of elements only in ground truth row
    """
    matches = 0
    element_in_pred_only = 0
    element_in_truth_only = 0

    for pred_val in predicted_row:
        if pd.isna(pred_val):
            continue

        if pred_val in ground_truth_row:
            matches += 1
        else:
            element_in_pred_only += 1

    for truth_val in ground_truth_row:

        if pd.isna(truth_val):
            continue

        if truth_val not in predicted_row:
            element_in_truth_only += 1

    return matches, element_in_pred_only, element_in_truth_only


def calculate_soft_f1(predicted: pd.DataFrame, ground_truth : pd.DataFrame) -> float:
    """
    Calculate the soft F1-score between the GT SQL and predicted SQL results

    Args:
        predicted (pd.DataFrame): Predicted SQL query results
        ground_truth (pd.DataFrame): Ground truth SQL query results

    Returns:
        float: Soft F1-score between the two dataframes
    """
    predicted = list(predicted.itertuples(index = False, name = None))
    ground_truth = list(ground_truth.itertuples(index = False, name = None))
    # if both predicted and ground_truth are empty, return 1.0 for f1_score
    if not predicted and not ground_truth:
        return 1.0

    # Calculate matching scores for each possible pair
    match_scores = []
    pred_only_scores = []
    truth_only_scores = []

    # Rows only in the ground truth results
    for i, gt_row in enumerate(ground_truth):
        if i >= len(predicted):
            match_scores.append(0)
            truth_only_scores.append(1)
            continue

        pred_row = predicted[i]
        match_score, pred_only_score, truth_only_score = calculate_row_match(
            pred_row, gt_row
        )
        match_scores.append(match_score)
        pred_only_scores.append(pred_only_score)
        truth_only_scores.append(truth_only_score)

    # Rows only in the predicted results
    for i in range(len(predicted) - len(ground_truth)):
        match_scores.append(0)
        pred_only_scores.append(1)
        truth_only_scores.append(0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )
    return f1_score

def round_numbers(df : pd.DataFrame, sig_figs: int = 5) -> pd.DataFrame:
    """
    Round numbers to the nearest signifcant figures.

    Args:
        df (pd.DataFrame): Dataframe containing numeric values.
        sig_figs (int, optional): Significant figures . Defaults to 5.

    Returns:
        pd.DataFrame: Dataframe with rounded numeric values.
    """
    # Create a copy to avoid SettingWithCopy warnings if df is a slice
    df = df.copy()
    
    # Identify numeric columns
    num_cols = df.select_dtypes(include='number').columns

    # Round each numeric column
    for col in num_cols:
        s = df[col].to_numpy(dtype=float)

        mask = s != 0
        mags = np.zeros_like(s)
        mags[mask] = sig_figs - np.floor(np.log10(np.abs(s[mask]))) - 1

        s[mask] = np.round(s[mask] * 10**mags[mask]) / 10**mags[mask]

        df[col] = s

    return df

def format_df_to_table(df: pd.DataFrame) -> str:
    """
    Safely formats a dataframe to a pretty-printed table.
    Args:
        df: The dataframe to format.
    Returns:
        A string representing the formatted table."""
    
    return df.head(10).to_markdown(index = False, tablefmt="github")


def process_query_result(execution: QueryExecutionResult, gt_df: pd.DataFrame) -> dict:
    """
    Handles the extraction of metrics and formatting from a query execution.
    Args:
        execution (QueryExecutionResult): The result of the query execution.
        gt_df (pd.DataFrame): The ground truth dataframe for comparison.
    Returns:
        dict: A dictionary containing the result string, F1 score, validity, and bytes processed"""
    
    if isinstance(execution.sql_result, GoogleAPIError):
        return {
            "result_str": f"ERROR: {execution.sql_result.message}",
            "f1": 0.0,
            "validity": 0.0,
            "bytes": None
        }

    pred_df = round_numbers(execution.sql_result)
    pred_df = pred_df.astype("string").fillna("<PRED N/A>")

    return {
        "result_str": format_df_to_table(pred_df),
        "f1": calculate_soft_f1(pred_df, gt_df),
        "validity": 1.0,
        "bytes" : execution.bytes_processed
    }

class EvaluationPipeline:
    def __init__(
            self, 
            input_path : str, 
            output_path : str,
            dataset_id : str ,
            client : Client
    ):
        """
        Initalises the evaluation pipeline

        Args:
            input_path (str): Input path for evaluation dataset
            output_path (str): Output path for evaluation results
            dataset_id (str): Dataset ID for BigQuery
            client (Client): BigQuery client instance
        """
        self.input_path = input_path
        self.dataset_id = dataset_id
        self.output_path = output_path
        self.client = client
        self.eval_dataset = pd.read_csv(input_path)

    async def evaluate(self, graph: CompiledStateGraph) -> None:
        """
        Evaluate the provided text-to-SQL graph on the evaluation dataset

        Args:
            graph (CompiledStateGraph): Compiled Langgraph state graph instance
        """
        eval_dataset = self.eval_dataset

        # Initialize results dictionary
        results = {
            "gt_sql_result": [],
            "pred_sql": [],
            "pred_sql_result": [],
            "soft_f1_score": [],
            "sql_validity": [],
            "gigabytes_processed": [],
            "time_elapsed": [],
            "token_usage": []
        }

        # Evaluate each question in the dataset
        for row in tqdm(eval_dataset.itertuples(), total=len(eval_dataset)):
            question = row.question
            # Execute GT SQL to obtain ground truth results
            gt_df = round_numbers(self.client.query(row.gt_sql).to_dataframe())
            gt_df = gt_df.astype("string").fillna('<GT N/A>')
            gt_sql_result = format_df_to_table(gt_df)

            # Execute the graph to obtain predicted SQL and results
            start_time = time.perf_counter() 
            with get_openai_callback() as cb:
                response = await graph.ainvoke({
                    "question": question,
                    "dataset_id": self.dataset_id,
                })
                total_tokens = cb.total_tokens

            elapsed_time = time.perf_counter() - start_time
            pred_sql = response.get("selected_sql")
            success_queries = response.get("success_queries") or {}

            eval_data = {
                "result_str": "ERROR: No Suitable Queries Found",
                "f1": 0.0,
                "validity": 0.0,
                "bytes": None
            }

            if pred_sql:
                execution = success_queries.get(pred_sql) or run_query_sync(query=pred_sql, client=self.client)
                eval_data.update(process_query_result(execution, gt_df))

            # Append results to the dictionary
            results["gt_sql_result"].append(gt_sql_result)
            results["pred_sql"].append(pred_sql)
            results["pred_sql_result"].append(eval_data["result_str"])
            results["soft_f1_score"].append(eval_data["f1"])
            results["sql_validity"].append(eval_data["validity"])
            results["time_elapsed"].append(elapsed_time)
            results["gigabytes_processed"].append(
                eval_data["bytes"] / BYTES_TO_GB if eval_data["bytes"] is not None else None
            )
            results["token_usage"].append(total_tokens)

        # Save results to CSV
        evaluation_df = pd.concat(
            [eval_dataset, pd.DataFrame(results)],
            axis=1,
        )
        evaluation_df.to_csv(self.output_path, index=False)


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

    # Initialize components
    embedder = get_class("embedding", settings.embedder.provider)(**settings.embedder.params)
    opensearch = get_class("vectorstore", settings.vectorstore.provider)(
        embedding_function = embedder, 
        http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASS),
        **settings.vectorstore.params
    )
    factual_llm = get_class("llm", settings.llm.factual.provider)(**settings.llm.factual.params)
    creative_llm = get_class("llm", settings.llm.creative.provider)(**settings.llm.creative.params)
    bigquery_client = Client()

    # Construct graphs
    graph_builder = GraphConstructor(
        factual_llm=factual_llm,
        creative_llm=creative_llm,
        opensearch=opensearch,
        bigquery_client=bigquery_client
    )

    # Construct baseline
    baseline_builder = BaselineConstructor(
        factual_llm=factual_llm,
        bigquery_client=bigquery_client
    )
    graph = graph_builder.compile()
    baseline = baseline_builder.compile()

    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(
        input_path = settings.input_path,
        output_path = settings.output_path,
        dataset_id = settings.dataset_id,
        client = bigquery_client
    )

    # Evaluate the graph and baseline
    #asyncio.run(pipeline.evaluate(baseline))
    asyncio.run(pipeline.evaluate(graph))




