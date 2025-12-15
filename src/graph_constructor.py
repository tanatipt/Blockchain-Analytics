from langchain_core.language_models import BaseChatModel
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from google.cloud.bigquery import Client
from src.modules.schemas import State
from langgraph.graph.state import CompiledStateGraph, StateGraph, START, END
from src.modules.select_examples import select_examples
from src.modules.select_schema import select_schema
from src.modules.generate_dac_sql import generate_dac_sql
from src.modules.generate_qp_sql import generate_qp_sql
from src.modules.generate_rp_sql import generate_rp_sql
from src.modules.execute_query import execute_query, execute_router
from src.modules.revise_query import revise_query
from src.modules.select_query import select_query
import inspect

class GraphConstructor:
    def __init__(
        self, 
        factual_llm : BaseChatModel,
        creative_llm : BaseChatModel,
        opensearch: OpenSearchVectorSearch,
        bigquery_client : Client
    ):
        self.select_examples = self.init_node(select_examples, opensearch = opensearch, llm = factual_llm)
        self.select_schema = self.init_node(select_schema, client = bigquery_client, llm = factual_llm)
        self.generate_dac_sql = self.init_node(generate_dac_sql, llm = creative_llm)
        self.generate_qp_sql = self.init_node(generate_qp_sql, llm = creative_llm)
        self.generate_rp_sql = self.init_node(generate_rp_sql, llm = creative_llm)
        self.execute_query = self.init_node(execute_query, client = bigquery_client)
        self.revise_query = self.init_node(revise_query, llm = factual_llm)
        self.select_query = self.init_node(select_query, llm = factual_llm)


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
        # Retrieval workflow
        retrieval_workflow = StateGraph(State)
        retrieval_workflow.add_node('select_examples', self.select_examples)
        retrieval_workflow.add_node('select_schema', self.select_schema)
        retrieval_workflow.add_edge(START, 'select_examples')
        retrieval_workflow.add_edge(START, 'select_schema')
        retrieval_workflow.add_edge('select_schema', END)
        retrieval_workflow.add_edge('select_examples', END)
        retrieval_subgraph = retrieval_workflow.compile()

        # SQL Generation workflow
        generation_workflow = StateGraph(State)
        generation_workflow.add_node('generate_dac_sql', self.generate_dac_sql)
        generation_workflow.add_node('generate_qp_sql', self.generate_qp_sql)
        generation_workflow.add_node('generate_rp_sql', self.generate_rp_sql)
        generation_workflow.add_edge(START, "generate_dac_sql")
        generation_workflow.add_edge(START, "generate_qp_sql")
        generation_workflow.add_edge(START, "generate_rp_sql")
        generation_workflow.add_edge("generate_dac_sql", END)
        generation_workflow.add_edge("generate_qp_sql", END)
        generation_workflow.add_edge("generate_rp_sql", END)
        generation_subgraph = generation_workflow.compile()

        # Retrieval + Generation workflow
        workflow = StateGraph(State)
        workflow.add_node("retrieve_context", retrieval_subgraph)
        workflow.add_node("generate_sql", generation_subgraph)
        workflow.add_node('execute_query', self.execute_query)
        workflow.add_node('revise_query', self.revise_query)
        workflow.add_node('select_query', self.select_query)

        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_sql")
        workflow.add_edge("generate_sql", "execute_query")
        workflow.add_conditional_edges("execute_query", execute_router, {"__end__" : END, "revise_query" : "revise_query", "select_query" : "select_query"})
        workflow.add_edge("revise_query", "execute_query")
        workflow.add_edge("select_query", END)

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