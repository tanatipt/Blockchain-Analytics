import pandas as pd
import pandas_toon
from google.cloud.bigquery.table import ForeignKey

LIGHT_SCHEMA_TEMPLATE = """## Table `{table}`
### Table description
{table_description}"""

COLUMN_INFORMATION = """### Column information
{column_information}"""

PRIMARY_KEY = """### Primary keys
{primary_key}"""

FOREIGN_KEY = """### Foreign keys
{foreign_key}"""

def create_pk_description(table_id, primary_keys: list[str]):
    return f"Table `{table_id}` has Primary Key(s): ({",".join(primary_keys)})."

def create_fk_description(table_id, foreign_keys: list[ForeignKey]):
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

def shorten(s, width=20, placeholder="..."):
    if len(s) <= width:
        return s
    return s[: width - len(placeholder)] + placeholder

def create_column_information(columns : list):
    pd_data = {
        "column_name": [],
        "column_type": [],
        "column_description": [],
        "is_nullable" : [],
        #"sample_values" : []
    }

    for column in columns:
        pd_data["column_name"].append(column["name"])
        pd_data["column_type"].append(column["data_type"])
        pd_data["column_description"].append(column["description"])
        pd_data["is_nullable"].append(column["is_nullable"])

        #if len(column["sample_values"]) > 0:
        #    pd_data["sample_values"].append(
        #        [
        #            shorten(val, width=20, placeholder="...") if isinstance(val, str) else val
        #            for val in column["sample_values"]
        #        ]
        #    )
        #else:
        #    pd_data["sample_values"].append("Sample Values Unavailable")

    column_information = pd.DataFrame(pd_data)
    return column_information.to_toon()

def format_light_schema(table_information : dict, include_column_info : bool = False) -> str:
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