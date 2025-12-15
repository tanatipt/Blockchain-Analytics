import pandas as pd
from langgraph.graph.state import CompiledStateGraph
from tqdm import tqdm
from src.graph_constructor import GraphConstructor
from src.baseline_constructor import BaselineConstructor
from google.cloud.bigquery import Client
from google.api_core.exceptions import GoogleAPIError
from src.modules.execute_query import run_query_sync
from src.mapper import get_class
from config import settings
import os
import asyncio
import time

def calculate_row_match(predicted_row : tuple, ground_truth_row : tuple):
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


def calculate_soft_f1(predicted: pd.DataFrame, ground_truth : pd.DataFrame):
    predicted = list(predicted.itertuples(index = False, name = None))
    ground_truth = list(ground_truth.itertuples(index = False, name = None))
    # if both predicted and ground_truth are empty, return 1.0 for f1_score
    if not predicted and not ground_truth:
        return 1.0

    # Calculate matching scores for each possible pair
    match_scores = []
    pred_only_scores = []
    truth_only_scores = []
    for i, gt_row in enumerate(ground_truth):
        # rows only in the ground truth results
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

    # rows only in the predicted results
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

class EvaluationPipeline:
    def __init__(
            self, 
            input_path : str, 
            output_path : str,
            dataset_id : str ,
            client : Client
        ):
        self.input_path = input_path
        self.dataset_id = dataset_id
        self.output_path = output_path
        self.client = client
        self.eval_dataset = pd.read_csv(input_path)

    async def evaluate(self, graph: CompiledStateGraph):
        eval_dataset = self.eval_dataset
        additional_columns = {
            "gt_sql_result" : [],
            "pred_sql" : [],
            "pred_sql_result" : [],
            "soft_f1_score" : [],
            "time_elapsed" : []
        }

        for row in tqdm(eval_dataset.itertuples(), total = len(eval_dataset)):
            print(row)
            question = row.question
            gt_sql = row.gt_sql
            gt_sql_result = self.client.query(gt_sql).to_dataframe()

            start_time = time.time()
            response = await graph.ainvoke({
                "question" : question,
                "dataset_id" : self.dataset_id
            })
            end_time = time.time()

            elapsed_time = end_time - start_time
            pred_sql = response.get("selected_sql")
            soft_f1_score = 0.0
            pred_sql_result = None

            if pred_sql is None:
                pred_sql_result = "ERROR: No Suitable Queries Found"
            else:
                success_queries = response.get("success_queries")

                if success_queries:
                    pred_sql_result = success_queries[pred_sql]

                else:
                    pred_sql_result = run_query_sync(
                        query=pred_sql,
                        client=self.client,
                    )

                if not isinstance(pred_sql_result, GoogleAPIError):
                    soft_f1_score = calculate_soft_f1(
                        pred_sql_result,
                        gt_sql_result,
                    )

            additional_columns["gt_sql_result"].append(gt_sql_result)
            additional_columns["pred_sql"].append(pred_sql)
            additional_columns["pred_sql_result"].append(pred_sql_result)
            additional_columns["soft_f1_score"].append(soft_f1_score)
            additional_columns["time_elapsed"].append(elapsed_time)


        additional_df = pd.DataFrame.from_dict(additional_columns)
        evaluation_df = pd.concat([eval_dataset, additional_df], axis = 1)
        evaluation_df.to_csv(self.output_path, index = False)



if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY


    embedder = get_class("embedding", settings.embedder.provider)(**settings.embedder.params)
    opensearch = get_class("vectorstore", settings.vectorstore.provider)(
        embedding_function = embedder, 
        http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASS),
        **settings.vectorstore.params
    )
    factual_llm = get_class("llm", settings.llm.factual.provider)(**settings.llm.factual.params)
    creative_llm = get_class("llm", settings.llm.creative.provider)(**settings.llm.creative.params)
    bigquery_client = Client()


    graph_builder = GraphConstructor(
        factual_llm=factual_llm,
        creative_llm=creative_llm,
        opensearch=opensearch,
        bigquery_client=bigquery_client
    )

    baseline_builder = BaselineConstructor(
        factual_llm=factual_llm,
        bigquery_client=bigquery_client
    )
    graph = graph_builder.compile(save_path='graph_structure.png')
    baseline = baseline_builder.compile(save_path='baseline_structure.png')
    pipeline = EvaluationPipeline(
        input_path = settings.input_path,
        output_path = settings.output_path,
        dataset_id = settings.dataset_id,
        client = bigquery_client
    )

    #asyncio.run(pipeline.evaluate(baseline))
    asyncio.run(pipeline.evaluate(graph))




