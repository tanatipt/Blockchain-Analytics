from src.modules.schemas import State
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from src.modules.format_light_schema import format_light_schema
from src.modules.common_prompt import database_guidelines, question_guidelines, eth_knowledge
from pydantic import BaseModel, Field
from config import settings

class StudentUtterances(BaseModel):
    """
    A Pydantic model representing the PhD-level student utterances in one turn of the conversation.
    """
    student_utterances : list[str] = Field(..., description = "A list of utterances made by the PhD-level student in the conversation.")

class ProfessorUtterances(BaseModel):
    """
    A Pydantic model representing the world-class professor utterances in one turn of the conversation.
    """
    profressor_utterances: list[str] = Field(..., description = "A list of utterances made by the world-class professor in the conversation.")



class RoleplayOutput(BaseModel):
    """
    A Pydantic model representing the structured output of the roleplay conversation and final SQL query.
    The model contains:
    - conversation: A list of turns in the conversation between the PhD-level student and the world-class professor to solve the challenging SQL problem.
    - final_sql: The final SQL query generated from the conversation. All column names in the optimised SQL query **must be enclosed** with `...`.
    """
    conversation : list[StudentUtterances | ProfessorUtterances] =  Field(..., description = "A turn-by-turn dialogue between the PhD-level student and the world-class professor.")
    final_sql : str = Field(..., description = "The final SQL query derived from the conversation.")

generate_sql_prompt = """
# Persona
You are simulating a **two-person collaborative problem-solving session** between:

## The PhD-Level Student  
A highly analytical, motivated researcher with deep knowledge of databases, relational algebra, and SQL query generation.  
They:
- Propose ideas, hypotheses, and potential SQL strategies  
- Analyse the user question and schema from first principles  
- Explore multiple solution pathways and compare them  
- Build the full SQL query step by step  
- Are the *only* persona allowed to generate or assemble the SQL solution  
- Are proactive, curious, and rigorous in their reasoning  
- Think aloud, attempt alternative plans, correct themselves, and refine the query  

## The Professor  
A world-class expert in databases and query optimisation.  
They:
- Never write SQL directly  
- Never solve the problem themselves  
- Only review, critique, pressure-test, and refine the student's reasoning  
- Highlight missing details, logical gaps, inefficiencies, or incorrect assumptions  
- Encourage the student to justify choices like join order, filtering strategy, and scan paths  
- Push for clarity, correctness, and an optimal relational-algebra approach  
- Ensure the student's final SQL is accurate and efficient
Usually, the profressor follows the following guidelines when providing feedback: {database_guidelines}
Both personas are deeply motivated, collaborative, and committed to solving difficult SQL-generation problems with precision.

# Scenario
A PhD-level student and their professor—a renowned expert in database systems—are working together to solve a challenging SQL query problem.  
They maintain a constructive, respectful, technical dialogue.  
The **student drives the reasoning**, while the **professor sharpens the reasoning**.

The conversation should feel like a real academic discussion:
- The student proposes structured ideas, intermediate reasoning, and candidate query fragments.  
- The professor critiques, asks probing questions, and challenges inefficiencies.  
- The student iteratively revises the plan until it converges to a correct and optimised query.  
- The final result must be a **precise SQL query** with a **clear reasoning path**.

# Instruction
Given **a natural-language question** and **a database schema**, your task is to:
1. **Simulate a back-and-forth conversation** between the PhD-level student and the professor.  
2. **Comprehensively analyse** both the question and schema.  
3. Work collaboratively (with the student leading) to construct:
   - A full reasoning process  
   - A clear explanation of the relational operations required  
   - The final optimised BigQuery SQL query. All column names in the optimised SQL query **must be enclosed** with `...`.

The conversation **must always** obey the following rules at all times:
- **The student always proposes the solution direction.**  
- **The professor never writes the SQL.** They *only critique, refine, and guide*.  
- **The student is the only persona allowed to construct the final SQL query.**  
- The conversation continues until the solution reaches a high-quality final SQL query.  
- The tone is academic, focused, and highly technical.

Use the following question guidelines to correctly interpret the intent and meaning of each question before generating the SQL:
{question_guidelines}

The following background information about the Ethereum blockchain may be useful:
{ethereum_knowledge}
"""

async def generate_rp_sql(state : State, llm : BaseChatModel) -> State:
    """
    Generates SQL queries using the Roleplaying approach

    Args:
        state (State): State of the Langchain graph
        llm (BaseChatModel): Language model instance

    Returns:
        State: State of the Langchain graph
    """
    # Formatting the most relevant few-shot examples to the user's question
    few_shot_examples = "\n\n".join([f"<user_question>: {example_question}\n<sql>: {example_sql}" for example_question, example_sql in state.few_shot_examples])
    # Formatting the selected database schema into a Light Schema format
    selected_schema = format_light_schema(table_information=state.selected_schema, include_column_info=True)
    question = state.question

    generate_sql_pt = ChatPromptTemplate(
        [
            ('system', generate_sql_prompt),
            ('user', '<few_shot_examples>: {few_shot_examples}\n<database_schema>: {database_schema}\n<user_question>: {question}\n<sql>:')
        ]
    )

    generate_sql_chain = generate_sql_pt | llm.with_structured_output(RoleplayOutput)
     # Generating `n` different SQL queries using the role playing approach
    n = settings.self_consistency.roleplay

    batch_inputs = [
        {
            "few_shot_examples": few_shot_examples,
            "database_schema": selected_schema,
            "question": question,
            "database_guidelines": database_guidelines,
            "question_guidelines" : question_guidelines,
            "ethereum_knowledge": eth_knowledge
        }
        for _ in range(n)
    ]

    results = await generate_sql_chain.abatch(batch_inputs)
    sql_queries = {r.final_sql for r in results}

    # Add the `n` queries generated to the list of pending queries to be executed
    return {"pending_queries" : sql_queries}
