database_optimisation = """
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
    - Use `ROUND ()` to round the result to a specific number.
"""