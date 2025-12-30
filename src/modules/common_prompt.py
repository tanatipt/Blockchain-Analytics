# Prompt for common knowledge of Ethereum
eth_knowledge = """
- 1 ETH equals 1e18 Wei
- 1 Gwei equals 1e9 Wei
- 1 ETH equals 1e9 Gwei
"""

# Prompt for SQL generation guidelines
database_guidelines = """
- **SELECT Clause**
    - Only select columns mentioned in the user's question.
    - Avoid unnecessary columns or values.
- **Aggregation (MAX/MIN)**: Always perform JOINs before using MAX() or MIN().
- **ORDER BY with Distinct Values**: Use `GROUP BY < column >` before `ORDER BY < column > ASC | DESC` to ensure distinct values.
- **Handling NULLs**
    - If a column may contain NULL values, use `JOIN` or `WHERE < column > IS NOT NULL`.
    - When a field is sorted in ascending order , also apply a NOT NULL filter to it.
    - When using the MIN() function on a column , also include a WHERE clause to filter NULL values from that column.
- **FROM / JOIN Clauses**: Only include tables essential to answer the question.
- ** Thorough Question Analysis**: Address all conditions mentioned in the question.
- **DISTINCT Keyword**: Use `SELECT DISTINCT` when the question requires unique values ( e.g. , IDs , URLs).
- **Column Selection**: Carefully analyze column descriptions and evidences to choose the correct column when similar columns exist across tables.
- ** String Concatenation**: Never use `|| ' ' ||` or any other method to concatenate strings in the `SELECT` clause.
- **JOIN Preference**: Prioritize `INNER JOIN` over nested `SELECT` statements .
- ** Date Processing**:  Utilize `STRFTIME ()` for date manipulation ( e.g., `STRFTIME ('%Y', SOMETIME )` to extract the year).
- ** Formatting :**
    - Pay close attention to any formatting requirements in the question , such as
    specific decimal places or percentage representation . These are not just suggestions
    ; they are critical parts of the final answer and must be implemented using
    appropriate SQL functions ( e . g . , ROUND () and multiplying by 100) .
    - If no formatting requirements are specified, return the raw value and do not use `ROUND()` or any other formatting functions.
"""

# Prompt for question answering and interpretation guidelines
question_guidelines = """
- When a question requests an aggregate value over the last X time units (days, hours, weeks, or years), the current day, hour, week, or year must be excluded from the calculation because it is an incomplete time period and would bias the resulting aggregate.
- If the question does not specify a unit for a quantity or measurement, return the value in the unit stored in the database without performing any conversion. If a unit is specified, convert the stored value to the requested unit.
"""