import pandas as pd
import pandas_toon
from google.cloud.bigquery.table import ForeignKey

# Prompt template for table description
LIGHT_SCHEMA_TEMPLATE = """## Table `{table}`
### Table description
{table_description}"""

# Prompt template for column description
COLUMN_INFORMATION = """### Column information
{column_information}"""

# Prompt template for primary key description
PRIMARY_KEY = """### Primary keys
{primary_key}"""

# Prompt template for foreign key description
FOREIGN_KEY = """### Foreign keys
{foreign_key}"""

def create_pk_description(table_id : str, primary_keys: list[str]) -> str:
    """
    Creates a Primary Key description string

    Args:
        table_id (str): ID of the table
        primary_keys (list[str]): A list of primary key column names.

    Returns:
        str: Primary Key description string
    """
    return f"Table `{table_id}` has Primary Key(s): ({",".join(primary_keys)})."

def create_fk_description(table_id : str, foreign_keys: list[ForeignKey]) -> str:
    """
    Creates a Foreign Key description string

    Args:
        table_id (str): ID of the table
        foreign_keys (list[ForeignKey]): A list of ForeignKey objects.

    Returns:
        str: Foreign Key description string
    """
    foreign_key_str = ""

    for foreign_key in foreign_keys:
        referenced_table = foreign_key.referenced_table
        column_references = foreign_key.column_references

        for column_reference in column_references:
            foreign_key_str += (
                f"Table `{table_id}` has Foreign Key ({column_reference.referencing_column})"
                f"that references ({column_reference.referenced_column}) in Table `{referenced_table}`\n."
            )

    return foreign_key_str

def create_column_information(columns : list) -> str:
    """
    Creates a markdown table string for column information

    Args:
        columns (list): A list of column metadata dictionaries.

    Returns:
        str: Markdown table string representing column information
    """
    pd_data = {
        "column_name": [],
        "column_type": [],
        "column_description": [],
        "is_nullable" : []
    }

    for column in columns:
        pd_data["column_name"].append(column["name"])
        pd_data["column_type"].append(column["data_type"])
        pd_data["column_description"].append(column["description"])
        pd_data["is_nullable"].append(column["is_nullable"])

    column_information = pd.DataFrame(pd_data)
    return column_information.to_toon()

def format_light_schema(table_information : dict, include_column_info : bool = False) -> str:
    """
    Format table information into a Light Schema format

    Args:
        table_information (dict): A dictionary of table information
        include_column_info (bool, optional): Flag to include column description or not. Defaults to False.

    Returns:
        str: A formatted Light Schema string of the table information
    """
    table_schemas = []

    for table in table_information:

        table_description = table_information[table]['description']
        schema_items = []

        schema = LIGHT_SCHEMA_TEMPLATE.format(
            table = table,
            table_description = table_description.strip() if table_description is not None else f"Table `{table}` has no description.",
        )
        schema_items.append(schema)
        
        if include_column_info:
            columns = table_information[table]['columns']
            column_information = create_column_information(columns)
            schema_items.append(COLUMN_INFORMATION.format(column_information = column_information.strip()))

        primary_key = table_information[table]["primary_key"]
        if primary_key:
            schema_items.append(PRIMARY_KEY.format(primary_key=create_pk_description(table, primary_key)))
        else:
            schema_items.append(PRIMARY_KEY.format(primary_key=f"Table `{table}` has no Primary Keys."))

        foreign_key = table_information[table]["foreign_key"]
        if foreign_key:
            schema_items.append(FOREIGN_KEY.format(foreign_key= create_fk_description(table, foreign_key)))
        else:
            schema_items.append(FOREIGN_KEY.format(foreign_key=f"Table `{table}` has no Foreign Keys."))

        table_schemas.append("\n".join(schema_items))

    
    return "\n\n".join(table_schemas)