"""
etl2postgresql.py
All DataFrames are obtained from CSV files. They are transformed, and
all DataFrames are loaded to PostgreSQL (ETL process). CSV files are requested.
"""
from pathlib import Path
from typing import Iterable
import polars as pl
from sqlalchemy import create_engine, text

ROOT_PATH = Path("./4 DB connection/Tables")

# Conexión PostgreSQL
DB_USER = "postgres"
DB_PASSWORD = "zmBWbmxaNwSGVBrP"
DB_HOST = "db.uflkuitzcnrjyqsgbczm.supabase.co"
DB_PORT = "5432"
DB_NAME = "postgres"

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)
POSTGRES_SCHEMA = "public"
# Columnas que deben convertirse a fecha
col_date = ['created_at']

# Primary keys de cada tabla
primary_keys = {
    'order_items': 'id',
    'orders': 'order_id',
    'products': 'product_id',
    'users': 'user_id'
}

# Foreign keys de cada tabla ('table': ['fk_table': 'fk_column'])
foreign_keys = {
    # 'users': [{'orders': 'order_id'}],
    # 'products': [{'order_items': 'product_id'}]
}


def extract_from_file(table_name: str, root_path: Path) -> pl.DataFrame:
    """Read files and construct DataFrames."""
    file_path_csv = ROOT_PATH / f"{table_name}.csv"
    if not file_path_csv.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo para '{table_name}'.")

    quote_char = '"'
    encoding = "utf-8"

    df = pl.read_csv(
        file_path_csv,
        separator=',',
        quote_char=quote_char,
        has_header=True,
        encoding=encoding,
        schema_overrides=None,
        ignore_errors=True,         # Useful if there are damaged rows
        low_memory=True,            # Reduce RAM usage
    )
    return df


def parse_datetime_columns(df: pl.DataFrame, columns: Iterable[str],
                           fmt: str = "%Y-%m-%d %H:%M:%S", strict: bool = False) -> pl.DataFrame:
    """Converts text columns (Utf8) to pl.Datetime using strptime.
    The conversion only applies if the column exists and is Utf8."""
    return df.with_columns(
        [
            pl.col(col)
              .str.strptime(pl.Datetime, strict=False).dt.date()
              .alias(col)
            for col in columns
            if col in df.columns and df.schema[col] == pl.Utf8
        ]
    )


def map_polars_to_postgres(colname: str, dtype: pl.DataType) -> str:
    """Get the PostgreSQL type of every column."""
    # Date
    if dtype == pl.Date:
        return "DATE"
    # Datetime
    if getattr(dtype, "base_type", lambda: None)() == pl.Datetime:
        return "TIMESTAMP"
    mapping = {
        pl.Int8: "SMALLINT",
        pl.Int16: "SMALLINT",
        pl.Int32: "INTEGER",
        pl.Int64: "BIGINT",
        pl.UInt8: "SMALLINT",
        pl.UInt16: "INTEGER",
        pl.UInt32: "BIGINT",
        pl.UInt64: "NUMERIC",
        pl.Boolean: "BOOLEAN",
        pl.Float32: "REAL",
        pl.Float64: "DOUBLE PRECISION",
        pl.Utf8: "TEXT",
        pl.String: "TEXT",
    }
    return mapping.get(dtype, "TEXT")


def create_table_from_df(engine, table_name: str, df: pl.DataFrame,
                         primary_key: str | None = None) -> None:
    """Create table in PostgreSQL based on the DataFrame schema."""
    columns_sql = []

    for col, dtype in df.schema.items():
        sql_type = map_polars_to_postgres(col, dtype)
        # PostgreSQL utiliza comillas dobles para identificadores
        columns_sql.append(
            f'"{col}" {sql_type}'
        )

    # Primary Key
    pk_sql = ""
    if primary_key:
        if primary_key not in df.columns:
            print(
                f"⚠️ La Primary Key '{primary_key}' no existe en la tabla '{table_name}'."
            )
        else:
            pk_sql = (
                f', CONSTRAINT "PK_{table_name}" '
                f'PRIMARY KEY ("{primary_key}")'
            )

    else:
        print(f"⚠️ No se crea Primary Key para la tabla '{table_name}'")

    # Crear tabla
    create_sql = f"""
    DROP TABLE IF EXISTS "{table_name}" CASCADE;

    CREATE TABLE "{table_name}" (
        {', '.join(columns_sql)}
        {pk_sql}
    );
    """

    with engine.begin() as conn:
        conn.execute(text(create_sql))

    print(f"✅ Tabla '{table_name}' creada correctamente.")


def load_table(engine, df: pl.DataFrame, table_name: str) -> None:
    """Load Polars DataFrame into PostgreSQL."""
    # Convertir Polars -> Pandas
    pandas_df = df.to_pandas()
    # Insertar datos
    pandas_df.to_sql(
        name=table_name,
        con=engine,
        if_exists="append",
        index=False,
        method="multi"
    )
    print(f"✅ {len(df):,} registros insertados en '{table_name}'")


# FOREIGN KEYS
def create_foreign_keys(engine, foreign_keys: dict, schema: str = POSTGRES_SCHEMA) -> None:
    """Create foreign keys in PostgreSQL."""
    def qident(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    schema_q = qident(schema)
    with engine.begin() as conn:
        for child_table, fk_list in foreign_keys.items():
            child_q = qident(child_table)
            for fk_def in fk_list:
                for parent_table, fk_column in fk_def.items():
                    parent_q = qident(parent_table)
                    fk_col_q = qident(fk_column)
                    constraint_name = (
                        f"fk_{child_table}_{parent_table}"
                    )
                    constraint_q = qident(constraint_name)

                    alter_sql = f"""
                    ALTER TABLE {schema_q}.{child_q}
                    ADD CONSTRAINT {constraint_q}
                    FOREIGN KEY ({fk_col_q})
                    REFERENCES {schema_q}.{parent_q} ({fk_col_q});
                    """

                    try:
                        conn.execute(text(alter_sql))
                        print(
                            f"✅ FK creada: "
                            f"{child_table}.{fk_column} "
                            f"-> {parent_table}.{fk_column}"
                        )

                    except Exception as e:
                        print(
                            f"❌ Error creando FK "
                            f"{constraint_name}: {e}"
                        )
                        raise


def main():
    """E-T-L process."""
    # Buscar todos los archivos CSV
    csv_files = list(ROOT_PATH.glob("*.csv"))
    if not csv_files:
        print(
            f"❌ No se encontraron archivos CSV en: {ROOT_PATH}"
        )
        return
    print(f"📂 Se encontraron {len(csv_files)} archivos CSV.")

    for csv_file in csv_files:
        # El nombre de la tabla será el nombre del CSV
        table_name = csv_file.stem
        print("\n" + "=" * 60)
        print(f"📊 Procesando tabla: {table_name}")
        print("=" * 60)
        try:
            # 1. Extraction (E)
            df = extract_from_file(table_name, ROOT_PATH)

            # 2. Transformation (T)
            df_trans = parse_datetime_columns(df, col_date)
            print(f"\n📐 Schema: {df_trans.schema}")

            # 3. Create table
            pk = primary_keys.get(table_name)
            create_table_from_df(engine, table_name, df_trans, pk)

            # 4. Load to PostgreSQL and create primary key
            load_table(engine, df_trans, table_name)

        except Exception as e:
            print(f"\n❌ FALLO CRÍTICO para {table_name}.\n'{e}'")
            print("=" * 25)

    # Create foreign keys
    # print("\n" + "=" * 40 + "🔗 CREANDO FOREIGN KEYS" + "=" * 40)
    # create_foreign_keys(engine, foreign_keys)

    print("\n--- PROCESO FINALIZADO ---")


if __name__ == "__main__":
    main()
