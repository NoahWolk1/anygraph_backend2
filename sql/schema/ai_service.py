import os
import json
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
  
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)


def process_query(
    query: str,
    columns: List[Dict[str, Any]],
    dataset_url: str,
    conversation_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    column_info = "\n".join(
        [
            f"- {col['name']}: {col['datatype']} (example: {col.get('example_value', 'N/A')})"
            for col in columns
        ]
    )

    context_messages = []
    for msg in conversation_history:
        role = "User" if msg["sender"] == "user" else "Assistant"
        context_messages.append(f"{role}: {msg['message_txt']}")

    conversation_context = "\n".join(context_messages) if context_messages else ""

    # Build context section only if there's conversation history
    context_section = f"\nPrevious Conversation:\n{conversation_context}\n" if conversation_context else ""

    # Single prompt that handles both decision and code generation
    prompt = f"""You are a data analysis assistant. Analyze the user's query and respond appropriately.

Dataset URL: {dataset_url}
Dataset Schema:
{column_info}{context_section}
User Query: {query}

RESPONSE RULES:
- If the answer exists in conversation history, respond with TEXT only (no code)
- If asking about schema/columns, respond with TEXT only
- If asking for clarification of previous results, respond with TEXT only
- If needing NEW data analysis (calculations, filtering, aggregations, visualizations), respond with PYTHON CODE

OUTPUT FORMAT:
1. For TEXT responses, output JSON:
{{"type": "text", "response": "your answer here"}}

2. For CODE responses, output JSON:
{{"type": "code", "code": "complete Python code here"}}

CODE REQUIREMENTS (when type is "code"):
- Import pandas and needed libraries
- Load data: df = pd.read_csv("{dataset_url}")
- Include df_to_markdown helper for tables
- Print results in markdown format
- Use exact URL provided above

HELPER FUNCTION for tables:
def df_to_markdown(df, max_rows=None):
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        note = f"\\n*Showing first {{max_rows}} rows*"
    else:
        note = ""
    return df.to_markdown(index=False) + note

PANDAS BEST PRACTICES:
- NEVER use .fillna() on Index objects with non-scalar values
- When mapping Index values, use: df.index.map(dict).astype(str) or handle unmapped with .get()
- For renaming index labels: pd.Series(data, index=[...]) or df.rename()
- Convert Index to strings BEFORE operations: df.index.astype(str)
- Use .get() for safe dictionary access: mapping.get(val, str(val))
- Avoid chaining Index operations with fillna - use Series instead

Output ONLY the JSON object, nothing else."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )

        response_text = response.text.strip()
        
        # Clean markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:].strip()
        if response_text.startswith("```"):
            response_text = response_text[3:].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()

        result = json.loads(response_text)

        if result.get("type") == "text":
            return {
                "needs_code": False,
                "response": result.get("response", "I can help with that.")
            }
        elif result.get("type") == "code":
            code = result.get("code", "")
            # Clean code if it has markdown blocks
            code = clean_generated_code(code)
            return {
                "needs_code": True,
                "code": code
            }
        else:
            # Fallback: treat as text response
            return {
                "needs_code": False,
                "response": result.get("response", str(result))
            }

    except json.JSONDecodeError as e:
        # If JSON parsing fails, treat response as code
        return {
            "needs_code": False,
            "response": f"I encountered an error processing your request. Please try rephrasing your question."
        }
    except Exception as e:
        raise Exception(f"Failed to process query: {str(e)}")


def generate_analysis_code(
    query: str,
    columns: List[Dict[str, Any]],
    dataset_url: str,
    conversation_context: Optional[str] = None
) -> str:
    column_info = "\n".join(
        [
            f"- {col['name']}: {col['datatype']} (example: {col.get('example_value', 'N/A')})"
            for col in columns
        ]
    )

    context_section = ""
    if conversation_context and conversation_context != "No previous conversation.":
        context_section = f"""
Previous Conversation (for context on what the user might be referring to):
{conversation_context}

"""

    prompt = f"""You are a Python data analysis code generator. Generate Python code to analyze a dataset.

Dataset URL: {dataset_url}
Available Columns:
{column_info}
{context_section}
User Query: {query}

Generate Python code that:
1. Imports pandas as pd and any other needed libraries
2. Loads data with: df = pd.read_csv("{dataset_url}")
3. Performs the requested analysis
4. Prints results in MARKDOWN FORMAT

OUTPUT FORMAT RULES:
- When showing tabular data (multiple rows/columns), ALWAYS use markdown tables:
  | Column1 | Column2 | Column3 |
  |---------|---------|---------|
  | value1  | value2  | value3  |
- For single values or simple results, use plain text
- Use headers (## or ###) to organize sections if needed
- The AI decides: if data is better shown as a table, output a table

HELPER FUNCTION (include this for table output):
def df_to_markdown(df, max_rows=None):
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        note = f"\\n*Showing first {{max_rows}} rows*"
    else:
        note = ""
    return df.to_markdown(index=False) + note

NOTE: If the user asks for "all", "every", or "each" item, show ALL rows (pass max_rows=None).
Only limit rows if the dataset is very large (100+ rows) and user didn't explicitly ask for all.

PANDAS BEST PRACTICES - AVOID COMMON ERRORS:
1. Index Operations:
   - NEVER: index.map(dict).fillna(index)  ❌
   - CORRECT: index.map(dict).astype(str) ✓
   - CORRECT: pd.Series(values).rename(mapping) ✓
   - CORRECT: [mapping.get(x, str(x)) for x in index] ✓

2. Value Mapping:
   - For Series: use .map() or .replace()
   - For Index: convert to list/Series first or use list comprehension
   - Always handle unmapped values explicitly

3. Type Conversions:
   - Use .astype(str) for safe string conversion
   - Check for NaN values before string operations
   - Use .fillna() only with scalar values

CRITICAL: Use the EXACT URL provided above. Do NOT create variables for the URL.
CRITICAL: The code must be complete and runnable as-is.
CRITICAL: Print output in markdown format for better display.

Example structure:
import pandas as pd

def df_to_markdown(df, max_rows=None):
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        note = f"\\n*Showing first {{max_rows}} rows*"
    else:
        note = ""
    return df.to_markdown(index=False) + note

try:
    df = pd.read_csv("{dataset_url}")
    # Your analysis here
    result_df = df[['col1', 'col2']]  # example
    print(df_to_markdown(result_df))  # shows all rows
    # Or: print(df_to_markdown(result_df, max_rows=50)) to limit
except Exception as e:
    print(f"Error: {{e}}")

Output ONLY executable Python code. No markdown code blocks, no explanations."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )

        code = response.text.strip()

        if code.startswith("```python"):
            code = code[len("```python") :].strip()
        if code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()

        return code

    except Exception as e:
        raise Exception(f"Failed to generate code with Gemini: {str(e)}")


def stream_analysis_code(
    query: str,
    columns: List[Dict[str, Any]],
    dataset_url: str,
    conversation_context: Optional[str] = None
):
    column_info = "\n".join(
        [
            f"- {col['name']}: {col['datatype']} (example: {col.get('example_value', 'N/A')})"
            for col in columns
        ]
    )

    context_section = ""
    if conversation_context and conversation_context != "No previous conversation.":
        context_section = f"""
Previous Conversation (for context on what the user might be referring to):
{conversation_context}

"""

    prompt = f"""You are a Python data analysis code generator. Generate Python code to analyze a dataset.

Dataset URL: {dataset_url}
Available Columns:
{column_info}
{context_section}
User Query: {query}

Generate Python code that:
1. Imports pandas as pd and any other needed libraries
2. Loads data with: df = pd.read_csv("{dataset_url}")
3. Performs the requested analysis
4. Prints results in MARKDOWN FORMAT

OUTPUT FORMAT RULES:
- When showing tabular data (multiple rows/columns), ALWAYS use markdown tables:
  | Column1 | Column2 | Column3 |
  |---------|---------|---------|
  | value1  | value2  | value3  |
- For single values or simple results, use plain text
- Use headers (## or ###) to organize sections if needed
- The AI decides: if data is better shown as a table, output a table

HELPER FUNCTION (include this for table output):
def df_to_markdown(df, max_rows=None):
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        note = f"\\n*Showing first {{max_rows}} rows*"
    else:
        note = ""
    return df.to_markdown(index=False) + note

NOTE: If the user asks for "all", "every", or "each" item, show ALL rows (pass max_rows=None).
Only limit rows if the dataset is very large (100+ rows) and user didn't explicitly ask for all.

PANDAS BEST PRACTICES - AVOID COMMON ERRORS:
1. Index Operations:
   - NEVER: index.map(dict).fillna(index)  ❌
   - CORRECT: index.map(dict).astype(str) ✓
   - CORRECT: pd.Series(values).rename(mapping) ✓
   - CORRECT: [mapping.get(x, str(x)) for x in index] ✓

2. Value Mapping:
   - For Series: use .map() or .replace()
   - For Index: convert to list/Series first or use list comprehension
   - Always handle unmapped values explicitly

3. Type Conversions:
   - Use .astype(str) for safe string conversion
   - Check for NaN values before string operations
   - Use .fillna() only with scalar values

CRITICAL: Use the EXACT URL provided above. Do NOT create variables for the URL.
CRITICAL: The code must be complete and runnable as-is.
CRITICAL: Print output in markdown format for better display.

Example structure:
import pandas as pd

def df_to_markdown(df, max_rows=None):
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        note = f"\\n*Showing first {{max_rows}} rows*"
    else:
        note = ""
    return df.to_markdown(index=False) + note

try:
    df = pd.read_csv("{dataset_url}")
    # Your analysis here
    result_df = df[['col1', 'col2']]  # example
    print(df_to_markdown(result_df))  # shows all rows
    # Or: print(df_to_markdown(result_df, max_rows=50)) to limit
except Exception as e:
    print(f"Error: {{e}}")

Output ONLY executable Python code. No markdown code blocks, no explanations."""

    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
            ),
        ):
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"# Error: {str(e)}"


def clean_generated_code(code: str) -> str:
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def generate_chat_response(query: str, context: str = None) -> str:
    prompt = query
    if context:
        prompt = f"Context: {context}\n\nUser: {query}\n\nAssistant:"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
            ),
        )
        return response.text.strip()
    except Exception as e:
        raise Exception(f"Failed to generate response with Gemini: {str(e)}")


def stream_direct_response(
    query: str,
    columns: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]]
):
    column_info = "\n".join(
        [
            f"- {col['name']}: {col['datatype']} (example: {col.get('example_value', 'N/A')})"
            for col in columns
        ]
    )

    context_messages = []
    for msg in conversation_history:
        role = "User" if msg["sender"] == "user" else "Assistant"
        context_messages.append(f"{role}: {msg['message_txt']}")

    conversation_context = "\n".join(context_messages) if context_messages else "No previous conversation."

    prompt = f"""You are a helpful data analysis assistant. Answer the user's question based on the dataset schema and conversation history.

Dataset Schema:
{column_info}

Previous Conversation:
{conversation_context}

User Query: {query}

Provide a helpful, concise response. If referring to data from the conversation, be specific."""

    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
            ),
        ):
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error: {str(e)}"
