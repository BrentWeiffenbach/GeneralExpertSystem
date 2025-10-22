from gemini import GeminiAPI
from expert_system import KnowledgeBase
import json
from typing import List
import sqlite3

EXPERT_SQL_SYSTEM_PROMPT =  """
                            Only generate responses in valid JSON format with the following structure:
                            {
                                "sql_statements": [
                                    "SQL statement 1 as a string",
                                    "SQL statement 2 as a string",
                                    ...
                                ]
                            }
                            The SQL statements will be run on SQLite and should be designed to update or modify the database schema or data based on the expert input provided.
                            Attempt to capture the essence of the expert knowledge explicitly.
                            Your goal is to make an Expert System that is scalable and can handle a wide range of expert knowledge inputs, while being efficient to query for any number of user questions.
                            """
USER_SQL_SYSTEM_PROMPT =    """
                            Only generate responses in valid JSON format with the following structure:
                            {
                                "sql_statements": [
                                    "SQL statement 1 as a string",
                                    "SQL statement 2 as a string",
                                    ...
                                ]
                            }
                            The SQL statements will be run on SQLite and should be designed to obtain as much relevant information as possible to answer the user's question based on the current database schema and data.
                            """
USER_ANSWER_SYSTEM_PROMPT = """
                            Based on the user's question and the SQL answer data provided, generate a clear and concise final answer.
                            If you cannot answer becuase there was no data found, try to suggest wording changes to the user's question that might yield better results basedon the current database schema.
                            Ensure that the answer directly addresses the user's question and is easy to understand.
                            """


def parse_sql_from_response(response_text: str) -> List[str]:
    """
    Parse the SQL statement from the Gemini response text.

    Args:
        response_text (str): The response text from Gemini.

    Returns:
        List[str]: The extracted SQL statements.
    """
    try:
        # Handle markdown-wrapped JSON responses
        text = response_text.strip()
        if text.startswith("```json"):
            # Extract JSON from markdown code block
            start_idx = text.find("```json") + 7
            end_idx = text.find("```", start_idx)
            if end_idx != -1:
                text = text[start_idx:end_idx].strip()
            else:
                # If no closing ```, take everything after ```json
                text = text[start_idx:].strip()
        elif text.startswith("```"):
            # Handle generic code block
            start_idx = text.find("```") + 3
            end_idx = text.find("```", start_idx)
            if end_idx != -1:
                text = text[start_idx:end_idx].strip()
            else:
                text = text[start_idx:].strip()
        
        response_json = json.loads(text)
        assert isinstance(response_json, dict)
        if "sql_statements" not in response_json:
            print("No 'sql_statements' key in Expert SQL Agent response.")
            return []
        sql_statements: List[str] = response_json.get("sql_statements", [])
        # formatted_json = json.dumps(response_json, indent=2)
        # print(formatted_json)
        return sql_statements
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Problematic text: {response_text}")
        raise ValueError("Invalid JSON response from Gemini")


if __name__ == "__main__":
    expert_sql_agent = GeminiAPI(system_prompt=EXPERT_SQL_SYSTEM_PROMPT)
    user_sql_agent = GeminiAPI(system_prompt=USER_SQL_SYSTEM_PROMPT)
    user_answer_agent = GeminiAPI(system_prompt=USER_ANSWER_SYSTEM_PROMPT)
    kb = KnowledgeBase(db_path="expert_system.db")

    expert_mode = True

    while expert_mode:
        # get expert input
        expert_input = input(
            "Enter your expert knowledge to be added to the Expert System (or type 'exit' to quit): "
        )
        if expert_input.lower() == "exit":
            expert_mode = False
            kb.save()
            continue

        # send schema and expert input to gemini
        cur_schema = kb.get_schema()

        combined_prompt = (
            f"""Current database schema: {cur_schema}, Expert input: {expert_input}"""
        )

        print(f"Sending full prompt: \n {combined_prompt}")

        # expert response is json with unknown number of sql statements
        expert_response = expert_sql_agent.call(combined_prompt)
        # print(f"Expert SQL Agent Response: {expert_response}")
        if expert_response is None:
            print("No response from Expert SQL Agent.")
            continue
        if expert_response.text is None:
            print("No text in response from Expert SQL Agent.")
            continue

        sql_statements = parse_sql_from_response(expert_response.text)

        # run the sql
        for sql_statement in sql_statements:
            try:
                kb.execute(sql_statement)
                # print(f"Executed SQL: {sql_statement}")
            except Exception as e:
                print(f"Error executing SQL '{sql_statement}': {e}")

        print(f"New database schema after expert input: {kb.get_schema()}")

    while True:
        # get user question
        user_question = input(
            "Enter your question for the Expert System (or type 'exit' to quit): "
        )
        if user_question.lower() == "exit":
            break

        # send schema and user question to gemini
        cur_schema = kb.get_schema()

        combined_prompt = (
            f"""Current database schema: {cur_schema}, User question: {user_question}"""
        )

        print(f"Sending full prompt: \n {combined_prompt}")

        # user response is json with unknown number of sql statements
        user_response = user_sql_agent.call(combined_prompt)
        # print(f"User SQL Agent Response: {user_response}")
        if user_response is None:
            print("No response from User SQL Agent.")
            continue
        if user_response.text is None:
            print("No text in response from User SQL Agent.")
            continue

        sql_statements = parse_sql_from_response(user_response.text)

        answer_data: List[List[dict]] = []

        # run the sql
        for sql_statement in sql_statements:
            try:
                response_cursor: sqlite3.Cursor = kb.execute(sql_statement)
                rows = response_cursor.fetchall()
                # Convert sqlite3.Row objects to dicts
                result = [dict(row) for row in rows]
                answer_data.append(result)
                # print(f"Executed SQL: {sql_statement}")
            except Exception as e:
                print(f"Error executing SQL '{sql_statement}': {e}")

        # send answer data to gemini for final answer
        answer_prompt: str = f"""User question: {user_question}, SQL Answer Data: {json.dumps(answer_data)}"""

        print(f"Sending final answer prompt: \n {answer_prompt}")

        final_answer_response = user_answer_agent.call(answer_prompt)
        if final_answer_response is None:
            print("No response from User Answer Agent.")
            continue
        if final_answer_response.text is None:
            print("No text in response from User Answer Agent.")
            continue

        print("Final Answer from Expert System:")
        print(final_answer_response.text)
