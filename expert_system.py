import os
import sqlite3
from typing import Any, Tuple


class KnowledgeBase:
    """
    A simple SQLite database wrapper for executing queries and managing connections.
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize the SQLiteDB instance.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path: str = db_path
        if not os.path.exists(self.db_path):
            open(self.db_path, 'a').close()
        try:
            self.conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            raise Exception(f"Error connecting to database: {e}")
        self.conn.row_factory = sqlite3.Row

    def connect(self) -> None:
        """
        Establish a new connection to the SQLite database.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """
        Close the current database connection.
        """
        if self.conn:
            self.conn.close()

    def execute(self, query: str, params: Tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """
        Execute a single SQL query.

        Args:
            query (str): The SQL query to execute.
            params (Tuple[Any, ...], optional): Parameters for the SQL query.
        
        Returns:
            sqlite3.Cursor: The cursor object for fetching results.
        """
        try:
            cursor = self.conn.execute(query, params)
            return cursor
        except sqlite3.Error as e:
            raise Exception(f"Error executing query: {e}")
    
    def get_schema(self) -> str:
        """
        Retrieve the database schema formatted for Gemini API.

        Returns:
            str: A formatted string describing all tables and their columns.
        """
        try:
            # Get all tables
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = "Database Schema:\n"
            
            for table in tables:
                table_name = table['name']
                schema_info += f"\nTable: {table_name}\n"
                
                # Get columns for each table
                column_cursor = self.conn.execute(f"PRAGMA table_info({table_name});")
                columns = column_cursor.fetchall()
                
                schema_info += "Columns:\n"
                for column in columns:
                    col_name = column['name']
                    col_type = column['type']
                    schema_info += f"  - {col_name} ({col_type})\n"
            
            return schema_info
            
        except sqlite3.Error as e:
            raise Exception(f"Error retrieving schema: {e}")

    def save(self) -> None:
        """
        Commit the current transaction.
        """
        if self.conn:
            self.conn.commit()