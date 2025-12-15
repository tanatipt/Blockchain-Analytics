from src.modules.select_schema import extract_schema_info
from src.modules.format_light_schema import format_light_schema
from src.modules.common_prompt import database_optimisation
from google.cloud.bigquery import Client
from src.modules.schemas import Step
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph.state import StateGraph, CompiledStateGraph, START, END
import inspect

class State(BaseModel):
    question : str = Field("", description = "User's question")
    dataset_id : str = Field("", description = "Google Cloud Platform Dataset ID")
    schema_info : dict = Field({}, description = "A schema of the database")
    selected_sql : str = Field("", description = "The generated SQL that will answer the user's question")

class SqlOutput(BaseModel):
    """
    A Pydantic model representing the output of the baseline SQL generation process.
    This model includes:
    1. A structured chain of reasoning that details how the SQL query is generated from the question and schema.
    2. The final optimised SQL query generated to answer the user's question
    """
    chain_of_thoughts: list[Step] = Field(..., description = "A structured sequence of reasoning steps used to analyse the question and schema to generate the SQL query.")
    final_sql : str = Field(..., description = "The final and fully optimised BigQuery SQL query derived from the reasoning process.")

generate_sql_prompt = """
# Persona
You are an expert in relational databases, SQL query construction, and schema reasoning for BigQuery. You specialise in
analysing complex database schemas and translating natural-language questions into accurate, efficient SQL queries.

# Instruction
Given a user question and the provided database schema, your task is to **thoroughly analyse** both the schema and the question, 
then generate an SQL query that correctly answers the user’s request. Follow these guidelines to construct and optimise 
the SQL queries: {database_optimisation}
"""

def extract_schema(state : State, client : Client):
    schema_info = extract_schema_info(client = client, dataset_id=state.dataset_id)
    return {"schema_info" : schema_info}

async def generate_sql(state : State, llm : BaseChatModel):
    schema_info = format_light_schema(state.schema_info, include_column_info=True)
    question = state.question

    generate_sql_pt = ChatPromptTemplate(
        [
            ('system', generate_sql_prompt),
            ('user', '<database_schema>: {database_schema}\n<question>: {question}')
        ]
    )

    generate_sql_chain = generate_sql_pt | llm.with_structured_output(SqlOutput)
    response = await generate_sql_chain.ainvoke(
        {
            "database_schema" : schema_info, 
            "question" : question, 
            "database_optimisation" : database_optimisation
        })

    return {"selected_sql" : response.final_sql}

class BaselineConstructor:
    def __init__(self, bigquery_client : Client, factual_llm : BaseChatModel):
        self.extract_schema_info = self.init_node(extract_schema, client = bigquery_client)
        self.generate_sql = self.init_node(generate_sql, llm = factual_llm)
      
    def init_node(self, node_function : callable, **kwargs : dict) -> callable:
        """
        Initializes a node function with additional keyword arguments.
        Args:
            node_function (callable): The node function to be wrapped.

        Returns:
            callable: The wrapped node function with additional arguments.
        """
        if inspect.iscoroutinefunction(node_function):

            async def async_wrapped_node(state: State):
                return await node_function(state, **kwargs)

            return async_wrapped_node

        else:

            def sync_wrapped_node(state: State):
                return node_function(state, **kwargs)

            return sync_wrapped_node
    
    def connect_nodes(self) -> StateGraph:
        """        
        Connects the nodes to form the RAG architecture graph.
        Returns:
            StateGraph: The constructed RAG architecture graph.
        """
        workflow = StateGraph(State)
        workflow.add_node("extract_schema_info", self.extract_schema_info)
        workflow.add_node("generate_sql", self.generate_sql)

        workflow.add_edge(START, "extract_schema_info")
        workflow.add_edge("extract_schema_info", "generate_sql")
        workflow.add_edge("generate_sql", END)

        return workflow
    
    def compile(self, save_path : str = None) -> CompiledStateGraph:
        """ Compiles the RAG architecture graph.
        Args:
            save_path (str, optional): Path to save the graph visualization. Defaults to None.
        Returns:
            StateGraph: The compiled RAG architecture graph."""
        workflow = self.connect_nodes()
        graph = workflow.compile()

        if save_path is not None:
            png_graph = graph.get_graph().draw_mermaid_png()

            with open(save_path, "wb") as f:
                f.write(png_graph)

        return graph