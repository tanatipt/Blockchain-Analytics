from src.modules.schemas import State, Step
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from src.modules.common_prompt import database_optimisation
from src.modules.format_light_schema import format_light_schema
from config import settings
from pydantic import BaseModel, Field


class DecompositionOutput(BaseModel):
    """
    A Pydanctic model representing the output of a single sub-question decomposition step in the divide-and-conquer SQL generation process.
    This model includes:
    1. The sub-question derived from the original question.
    2. A structured chain of reasoning that details how the sub-question can be answered using the database schema.
    3. The partial pseudo-SQL query generated to answer the sub-question.
    """

    sub_question: str = Field(
        ...,
        description="A sub-question obtained by decomposing the original question."
    )
    chain_of_thoughts: list[Step] = Field(
        ...,
        description="A structured sequence of reasoning steps used to analyse the sub-question and schema to generate a partial pseudo-SQL query."
    )
    partial_sql: str = Field(
        ...,
        description="The partial pseudo-SQL query that answers the sub-question."
    )

class DivideandConquerOutput(BaseModel):
    """
    A Pydantic model representing the full output of the divide-and-conquer SQL generation process.
    This model includes:
    1. An list of decomposed sub-questions, each with reasoning steps and a partial SQL query to answer each sub-question. 
    2. A structured chain of reasoning that details how the final SQL query will be constructed from the partial SQL queries of the sub-questions.
    3. The final SQL query constructed from all partial SQL queries.
    4. A structured chain of reasoning that details how the final SQL query will be optimised.
    5. The optimised SQL query produced after improving the final SQL. All column names in the optimised SQL query **must be enclosed** with `...`.
    """
    partial_sqls: list[DecompositionOutput] = Field(
        ...,
        description="A list of decomposed sub-questions, each with reasoning steps and a partial SQL query generated through the divide-and-conquer process."
    )
    final_chain_of_thoughts: list[Step] = Field(
        ...,
        description="A structured sequence of reasoning steps used to combine partial SQL queries into a final SQL query, or to derive the final SQL directly when no decomposition is needed."
    )
    final_sql: str = Field(
        ...,
        description="The final SQL query constructed from all partial SQL queries."
    )
    optimised_chain_of_thoughts: list[Step] = Field(
        ...,
        description="A structured sequence of reasoning steps used to analyse and optimise the final SQL query."
    )
    optimised_sql: str = Field(
        ...,
        description="The optimised SQL query produced after improving the final SQL."
    )


generate_sql_prompt = """
# Persona
You are a highly specialised SQL reasoning engine with expert-level knowledge of BigQuery SQL. 
You excel at **divide-and-conquer planning**, multi-step reasoning, and systematic decomposition of complex natural-language questions. 
Your goal is to produce **correct, efficient, and interpretable SQL queries** by:
- Carefully analysing the user question
- Fully understanding the database schema and relationships
- Breaking the problem into logically independent sub-questions
- Generating partial SQL solutions
- Combining and optimising them into a final, clean BigQuery query

You follow a structured reasoning workflow and always explain your thought process explicitly.  
You prioritise correctness, clarity, and optimised SQL structure that follows BigQuery best practices (CTEs, explicit JOINs,
clear aliasing, safe filtering, NULL-handling, and deterministic ordering).

# Instruction
Given a question and a database schema, your task is to **comprehensively analyse** both inputs and generate 
a complete SQL query that answers the question using a rigorous divide-and-conquer approach:
1. **Decompose** the original question into a set of well-defined sub-questions.
2. **Generate a partial pseudo-SQL query** that answers each sub-question.
3. **Assemble** a final SQL query to answer the question by integrating all partial pseudo-SQL queries.
4. **Simplify and optimise** the final SQL query. All column names in the optimised SQL query **must be enclosed** with `...`.

Follow these guidelines to help you generate, assemble and optimise the SQL queries: {database_optimisation}

# Example
<question>: What is the gender of the youngest client who opened account in the lowest average salary
<sql>: 
## 1. Divide and Conquer
**Main Question**: What is the gender of the youngest client who opened account in the lowest average salary branch?
**Analysis**: Question asking about `gender`, and it appears in table `client`. We will use this as the
output column, selecting it from the youngest client in the lowest average salary branch.
**Pseudo SQL**: SELECT `T1.gender` FROM `client` AS `T1` WHERE youngest client in the lowest average salary branch

**Sub-question 1**: youngest client in the lowest average salary branch
**Analysis**: We need to use the `A11` from `district` to get the salary info,
and the youngest client can be obtained from using the `birth_date` column of table `client`. The items
between these two tables can be INNER JOIN using district_id.
**Pseudo SQL**: SELECT `T1.client_id` FROM `client` AS `T1` INNER JOIN `district` AS `T2` ON
`T1.district_id` = `T2.district_id` WHERE lowest average salary branch ORDER BY `T1.birth_date`
DESC NULLS LAST LIMIT 1

**Sub-question 1.1**: lowest average salary branch
**Analysis**: We can get the lowest average salary branch using order by `A11` ASC and pick top 1. The
column `A11` is not NULLABLE, so we do not need to add "IS NOT NULL" filter
**Pseudo SQL**: SELECT `district_id` FROM `district` ORDER BY `A11` ASC LIMIT 1

## 2. Assembling SQL
**Sub-question 1.1 (lowest average salary branch)** : SELECT `district_id` FROM `district` ORDER BY `A11` ASC LIMIT 1
**Sub-question 1 (youngest client in the lowest average salary branch)**: SELECT `T1.client_id` FROM `client` AS `T1` INNER JOIN `district` AS `T2` ON
`T1.district_id` = `T2.district_id` WHERE `T2.district_id` IN (SELECT `district_id` FROM `district` ORDER BY `A11` ASC LIMIT 1) ORDER BY `T1`.`birth_date` DESC NULLS LAST LIMIT 1
**Main Question (gender of the client)***: SELECT `T1.gender` FROM `client` AS `T1` WHERE `T1.client_id` = (SELECT`T1.client_id` FROM `client` AS `T1` INNER JOIN `district` AS `T2` ON `T1.district_id` =
`T2.district_id` WHERE `T2.district_id` IN (SELECT `district_id` FROM `district` ORDER BY `A11` ASC LIMIT 1) ORDER BY `T1.birth_date` DESC NULLS LAST LIMIT 1)

## 3. Simplification and Optimization
**Analysis**: The nested queries can be combined using a single `INNER JOIN` and the filtering can be done within a single `ORDER BY` clause.
**Final Optimized SQL Query**: SELECT `T1.gender` FROM `client` AS `T1` INNER JOIN `district` AS `T2` ON `T1.district`
"""


async def generate_dac_sql(state : State, llm : BaseChatModel):
    few_shot_examples = "\n\n".join([f"<user_question>: {example_question}\n<sql>: {example_sql}" for example_question, example_sql in state.few_shot_examples])
    selected_schema = format_light_schema(table_information=state.selected_schema, include_column_info=True)
    question = state.question

    generate_sql_pt = ChatPromptTemplate(
        [
            ('system', generate_sql_prompt),
            ('user', '<few_shot_examples>: {few_shot_examples}\n<database_schema>: {database_schema}\n<user_question>: {question}\n<sql>:')
        ]
    )

    generate_sql_chain = generate_sql_pt | llm.with_structured_output(DivideandConquerOutput)
    n = settings.self_consistency.divide_and_conquer
    batch_inputs = [
        {
            "few_shot_examples": few_shot_examples,
            "database_schema": selected_schema,
            "question": question,
            "database_optimisation": database_optimisation,
        }
        for _ in range(n)
    ]

    results = await generate_sql_chain.abatch(batch_inputs)
    sql_queries = {r.optimised_sql for r in results}
    return {"pending_queries" : sql_queries}


