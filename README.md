# Blockchain Analytics
In this project, we developed a text-to-SQL chatbot for blockchain analytics that allows users to interact with and query on-chain data to gain insights into network utilisation and activity. We focus specifically on the Ethereum blockchain, as it is one of the most widely used platforms; however, the system can be easily extended to other blockchains by modifying the prompt and database configuration. Given a user question, the chatbot operates through the following core components:

1. **Schema Linking**: The system filters the database schema to identify only the tables and columns relevant to the user’s question. At the same time, it retrieves semantically similar text-to-SQL examples from a vector database to serve as few-shot demonstrations that guide SQL generation.
2. **SQL Generation**: Using the selected schema, the chatbot generates candidate SQL queries via three complementary strategies: query planning, divide-and-conquer, and role-playing.
3. **SQL Revision**: The generated SQL queries are executed and iteratively revised to fix execution errors or logical issues. This process is repeated for a fixed number of iterations.
4. **SQL Selection**: From the set of successfully executed queries, the system selects the most optimal SQL query to return as the final result.

# Project Structure

Below is an overview of the project's structure, highlighting the most important files and their roles:
```bash
/src/
├── modules/
    ├── common_prompt.py                # Guideline prompts to help guide SQL generation
    ├── execute_query.py                # Node to execute the generated SQL queries
    └── format_light_schema.py          # Utility function to format a given database schema into the Light Schema format
    └── generate_dac_sql.py             # Node to generate the SQL queries using the divide-and-conquer approach
    └── generate_qp_sql.py              # Node to generate the SQL queries using the query planning approach
    └── generate_rp_sql.py              # Node to generate the SQL queries using the roleplaying approach
    └── revise_query.py                 # Node that revises and corrects SQL queries that produce execution errors or return no results.
    └── schemas.py                      # Defines common Pydantic schemas used by different nodes in the chatbot.
    └── select_examples.py              # Node to selct the most relevant text-to-SQL examples to the user question
    └── select_query.py                 # Node to select the most optimal SQL query from a set of successfully executed SQL queries
    └── select_schema.py                # Node to select the table and columns of the database schema that is the most relevant to the user question
├── offline/
    ├── embed_examples.py               # Embed the BIRD training text-to-SQL examples into the Opensearch vector store 
├── graph_constructor.py                # Constructs the text-to-SQL chatbot and connects different nodes together
├── baseline_constructor.py             # Constructs the baseline chatbot
├── mapper.py                           # Returns the appropriate class to instantiate depending on the arguments passed                            
├── evaluation_pipeline.py              # Evaluation script to evaluate the performance of the text-to-SQL and baseline chatbot

/config/
└── settings.yaml                       # Configuration file

/resources/
└── baseline.png                        # Baseline chatbot graph visualisation
└── graph.png                           # Text-to-SQL chatbot graph visualisation
```

# Queryable Ethereum Blockchain Data

Most analytics tasks are best performed on structured data. However, blockchain data is inherently difficult to analyze because it is stored in low-level, protocol-specific formats rather than in tidy, relational tables. In addition, blockchain data is distributed across blocks over time, rather than maintained as a single, centralized representation of the current state. As a result, there is a strong need for a unified, structured view of the Ethereum blockchain. Transforming raw blockchain data into a relational database addresses these challenges by providing a clean, queryable representation suitable for analytical workloads.

Constructing such a relational database requires traversing all blocks in the blockchain and processing the data to derive the current state, which typically necessitates building and maintaining a dedicated ETL pipeline. Fortunately, BigQuery provides a public dataset that performs this aggregation and transformation on our behalf, offering a unified, structured view of Ethereum blockchain data that is updated daily to reflect the latest state of the network.

# Evaluation Dataset

Due to the lack of text-to-SQL datasets specifically designed for blockchain analytics, we constructed our own evaluation dataset consisting of 40 text-to-SQL examples that pose analytical questions about on-chain Ethereum data. All questions are answerable using the provided Ethereum blockchain dataset; therefore, the dataset does not include any negative or unanswerable questions. In future work, the dataset and chatbot could be extended to include negative questions to evaluate the model’s ability to recognize unanswerable queries.

## Dataset Construction

The dataset was constructed using the following process:

1. We prompted Gemini 3.0 and ChatGPT to generate a large set of realistic analytical questions by providing them with the BigQuery schema of the Ethereum blockchain dataset. The models were also instructed to generate corresponding SQL queries for each question.
2. We sampled a diverse subset of the generated questions to ensure variation in difficulty and coverage across different aspects of Ethereum blockchain data.
3. Each question and its corresponding SQL query was manually reviewed and refined to ensure correctness, optimization, executability in BigQuery, and alignment with the intended semantics of the question.

## Sample Questions

<table>
  <thead>
    <tr>
      <th width="70%">Question</th>
      <th width="30%">Ground Truth SQL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>What are the per-day average base fee per gas and total gas used on Ethereum for the last 30 days?</td>
      <td>
<pre lang="sql">
SELECT 
  DATE(timestamp) AS date,
  AVG(base_fee_per_gas) AS avg_base_fee,
  SUM(gas_used) AS total_gas_used
FROM `bigquery-public-data.crypto_ethereum.blocks`
WHERE  
    DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
    AND DATE(timestamp) < CURRENT_DATE()
GROUP BY date
ORDER BY date ASC;</pre>
      </td>
    </tr>
    <tr>
      <td>Which 10 miners have mined the most blocks?</td>
      <td>
<pre lang="sql">
SELECT 
  miner
FROM `bigquery-public-data.crypto_ethereum.blocks`
GROUP BY miner
ORDER BY COUNT(*) DESC
LIMIT 10;</pre>
      </td>
    </tr>
    <tr>
      <td>Find the average gas used by transactions after 1 December 2025 that interacted with ERC-721 (NFT) contracts, compared with non-NFT contracts.</td>
      <td>
<pre lang="sql">
SELECT 
  c.is_erc721, 
  AVG(tx.receipt_gas_used) as avg_gas
FROM `bigquery-public-data.crypto_ethereum.transactions` tx
JOIN `bigquery-public-data.crypto_ethereum.contracts` c 
  ON tx.to_address = c.address
WHERE tx.block_timestamp >= '2025-12-01' AND c.is_erc721 IS NOT NULL
GROUP BY c.is_erc721;</pre>
      </td>
    </tr>
    <tr>
      <td>Which 10 days of the year have historically had the highest average gas used per Ethereum block, and what is the average gas used per block for each of those days?</td>
      <td>
<pre lang="sql">
SELECT 
  EXTRACT(DAYOFYEAR FROM timestamp) as day_of_year,
  AVG(gas_used) as total_gas_used
FROM `bigquery-public-data.crypto_ethereum.blocks`
WHERE gas_used IS NOT NULL
GROUP BY day_of_year
ORDER BY total_gas_used DESC
LIMIT 10;</pre>
      </td>
    </tr>
    <tr>
      <td>Which ERC-20 tokens were created in the last 30 days, and what are their names and symbols? Exclude any tokens with missing name or symbol.</td>
      <td>
<pre lang="sql">
SELECT 
  DISTINCT t.name, t.symbol
FROM `bigquery-public-data.crypto_ethereum.contracts` c
JOIN `bigquery-public-data.crypto_ethereum.tokens` t 
  ON c.address = t.address
WHERE DATE(c.block_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
    AND DATE(c.block_timestamp) < CURRENT_DATE() AND t.name IS NOT NULL and t.symbol IS NOT NULL
    AND c.is_erc20 = TRUE AND t.name IS NOT NULL AND t.symbol IS NOT NULL
ORDER BY t.name, t.symbol;</pre>
      </td>
    </tr>
  </tbody>
</table>

# Methodology

Our text-to-SQL chatbot is heavily inspired by prior work on general-purpose text-to-SQL systems. In particular, we adapt ideas from top-performing approaches on the [BIRD benchmark](https://bird-bench.github.io/). Specifically, our design draws on techniques introduced in the following works:

- [Agentar-Scale-SQL: Advancing Text-to-SQL through Orchestrated Test-Time Scaling](https://arxiv.org/pdf/2509.24403)
- [CHASE-SQL: Multi-Path Reasoning and Preference-Optimized Candidate Selection in Text-to-SQL](https://arxiv.org/pdf/2410.01943)
- [CHESS: Contextual Harnessing for Efficient SQL Synthesis](https://arxiv.org/pdf/2405.16755)

The diagram below illustrates the pipeline of our text-to-SQL chatbot. The first two nodes form subgraphs responsible for schema linking and initial SQL generation, while the subsequent nodes handle SQL revision and candidate selection.

<p align="center">
    <img src='resources/graph.png' width="50%">
</p>

## Context Retrieval

The `retrieve_context` subgraph node performs two primary tasks: **few-shot example selection** and **schema linking**.

### Few-Shot Example Selection

Prior work in text-to-SQL has shown that well-chosen few-shot examples can significantly improve model performance, as they provide effective guidance for SQL generation compared to poorly matched examples. To leverage this, we use the BIRD benchmark training dataset to construct a pool of text-to-SQL examples.

Intuitively, we embed text-to-SQL examples from the training set into a vector database and use the user’s question to perform a similarity search. The most relevant questions are then retrieved along with their corresponding SQL queries and used as few-shot demonstrations.

#### Question Skeleton Indexing

Given the BIRD benchmark training dataset, we first extract the skeleton of each question, capturing its underlying structural pattern while abstracting away domain-specific entities and values. We then embed each question skeleton using Google’s `gemini-001` embedding model.

We embed the skeleton rather than the full question because our goal is to capture the structural intent and logical form of the question, rather than its specific content. This makes the embeddings content-agnostic and allows questions with similar reasoning patterns—but different surface topics—to be represented closely in the embedding space.

After generating embeddings for all question skeletons, we store them in a locally hosted OpenSearch database to enable efficient vector-based retrieval.

#### Skeleton-Based Retrieval

Given a user question, we follow a process similar to the indexing stage. First, we prompt an LLM to extract the question skeleton, abstracting away database-specific entities, values, and column names. We then use this skeleton to perform a vector similarity search over the question skeletons stored in the OpenSearch database, retrieving the most structurally similar examples.

Finally, we use the original question and its corresponding ground-truth SQL from the retrieved skeletons as few-shot examples for downstream query generation.

### Schema Linking

Schema linking is a necessary component of any text-to-SQL system, particularly for scalability to databases with thousands of tables and high-dimensional schemas. Given a user question, only a subset of the database schema is typically relevant. The goal of schema linking is therefore to identify the most relevant tables and columns for downstream SQL generation.

Effective schema linking significantly reduces prompt size during SQL generation. Passing the entire database schema to an LLM is often redundant and incurs unnecessary cost. This motivation is analogous to RAG, where only the most relevant document chunks are provided to the model rather than the entire document.

In the literature, two common approaches to schema linking are widely used. The first approach embeds table and column representations—often including their descriptions—into a vector database, and then performs similarity search using the user question to retrieve the most relevant tables and columns. The second approach provides the full database schema alongside the user question to the LLM, allowing the model to reason over the schema and explicitly select the relevant tables and columns. For this project, we adopt the second approach for two main reasons:

1. **Schema size**: The overall schema of BigQuery’s Ethereum dataset is sufficiently small, making it feasible to pass the entire schema to the LLM without incurring prohibitive cost.
2. **Reasoning robustness**: In the vector-based approach, tables and columns are embedded and retrieved independently, which can ignore important contextual relationships. In contrast, the LLM-based approach enables joint reasoning over the schema, allowing the model to capture relationships between columns within the same table as well as across different tables, resulting in more coherent and reliable schema selection.

## SQL Generation

The `generate_sql` subgraph node is responsible for producing candidate SQL queries based on the selected few-shot examples and the linked database schema. We explore three primary approaches for SQL generation: divide-and-conquer,query planning, and role-playing. Each approach is instructed to generate `n` candidate queries, where `n` is a configurable parameter.

Increasing *n* encourages broader exploration of the solution space, which can improve the likelihood of generating correct and diverse SQL queries, albeit at the cost of higher token usage and longer computation time. To facilitate creative reasoning and diverse query generation, the LLM used in each approach is configured with non-zero `temperature` and `top_p` values.

The SQL queries generated by all three approaches are subsequently merged, and duplicate queries are removed. Across all approaches, the model is provided with a consistent set of inputs: the selected few-shot examples, the relevant database schema, shared best-practice guidelines for SQL query generation, instructions on how to interpret the user question, and domain-specific knowledge of the Ethereum blockchain.


### Divide-and-Conquer CoT
Divide-and-conquer is a problem-solving strategy that decomposes a complex problem into a set of smaller, more manageable sub-problems, solves each sub-problem independently, and then combines the partial solutions to produce the final result.

Following this principle, a user question is first decomposed into multiple sub-tasks, each expressed as a pseudo-SQL query that captures a specific part of the overall logic. In the conquer phase, the intermediate pseudo-SQL solutions are then aggregated and composed to form the final SQL query that answers the original question.

### Query Planning CoT
Given an SQL query, a query plan is a structured sequence of operations that a database management system (DBMS) follows to execute the query. During execution, the DBMS query optimizer translates an SQL statement into a query plan that specifies how tables are accessed, how joins are performed, and which operations (e.g., filtering, aggregation) are applied.

Inspired by this process, we adopt a query-planning–based reasoning strategy to guide the LLM in SQL generation. The model is prompted to reverse-engineer the execution process—reasoning like a database query optimizer—by first constructing a high-level query plan from the user question and then translating that plan into an executable SQL query. This strategy consists of three main steps: 

1. Identifying relevant tables required to answer the question.
2. Specifying operations such as filtering, aggregation, or joins between tables
3. Producing the final output by selecting the appropriate columns and result format.

### Roleplay Prompting
In this approach, instead of letting the LLM reason about the SQL solution to the user question from a first-person perspective, we prompt the LLM to reason from a third-person perspective. More specifically, the LLM simulates a back-and-forth conversation between two personas:

- **PHD Student**: Proposes ideas, hypotheses, and candidate SQL strategies to the professor. The student is the only persona allowed to construct the SQL query step by step.
- **Professor/Expert in Database and Query Optimisation**: Never writes SQL or directly solves the problem. Their role is strictly to review, critique, pressure-test, and refine the student’s reasoning.

The dialogue continues until the PhD student converges on a correct, well-justified SQL solution. This can be viewed as multi-persona self-collaborative prompting. One advantage of this approach, compared with having the LLM directly solve the user’s question, is that it encourages explicit reasoning, systematic validation of assumptions, and early detection of logical or optimisation flaws in the proposed query.

## SQL Revision

Given the set of SQL query candidates generated during the SQL generation stage, this stage iteratively alternates between executing queries against the BigQuery database and revising those that result in errors or empty outputs. This execution–revision cycle repeats for a fixed number of rounds until the stage terminates. In this chatbot, a maximum of five rounds is allowed. Each query progresses through the following lifecycle states:

- Pending Queries: Initial or revised candidate SQL queries that are awaiting execution.
- Executed Queries: Candidate SQL queries that have already been executed against the database. These are retained to prevent duplicate executions and avoid incurring unnecessary costs.
- Failed Queries: Candidate SQL queries that result in an execution error or produce empty outputs in the current revision round. These queries are selected for revision and correction within the same round.
- Success Queries: Candidate SQL queries that execute successfully in any round of this stage.

The pseudo-code for this stage is shown below:

```text
Input:
    pending_queries : Set of SQL query candidates from the SQL generation stage
Output:
    success_queries : Dictionary of successfully executed SQL queries to execution results

revision_count ← 0
# Dictionary of successfully executed queries
success_queries ← {}
# Set of executed queries
executed_queries ← ∅

# Tracking number of revision rounds
WHILE revision_count ≤ 5 DO
    # Dictionary of failed queries for current revision round
    failed_queries ← {}

    # Executing pending queries that haven't been executed before
    FOR each query ∈ pending_queries DO
        IF query ∈ executed_queries THEN
            CONTINUE
        END IF

        sql_result ← Execute(query)
        executed_queries ← executed_queries ∪ {query}

        # Recording whether the query is successfully executed or not
        IF sql_result is error OR sql_result is empty THEN
            failed_queries[query] ← sql_result
        ELSE
            success_queries[query] ← sql_result
        END IF
    END FOR

    # Revising and correcting each failed queries in current round
    FOR each query ∈ failed_queries DO
        revised_query ← LLM_Revise(
            query,
            error_or_result,
            schema,
            user_question
        )

        pending_queries ← pending_queries ∪ {revised_query}
    END FOR

    revision_count ← revision_count + 1
END WHILE

RETURN success_queries
```




## SQL Selection

After the SQL revision stage, we obtain a dictionary of successfully executed SQL queries along with their execution results. The SQL selection stage then determines a single final SQL query from this set. This stage consists of the following steps:

1. **SQL Grouping**: The successfully executed SQL queries are first grouped according to their execution results. Queries that produce identical results are placed in the same group, and one representative query is randomly selected from each group. If all successfully executed queries yield identical results, the representative query from that group is directly returned as the final SQL query.

2. **Tournament Selection**: The representative query from each group participates in a pairwise round-robin tournament. For each unique pair of SQL queries, an LLM evaluates the two queries and selects a winner based on their execution's result correctness and efficiency with respect to the database schema and the user’s question. The winning query receives a score increment. After all pairwise comparisons have been completed, the SQL query with the highest score is selected as the final output.

# Evaluation

## Evaluation Metric


## Baseline Model
## Evaluation Result

# Installation & Project Setup
