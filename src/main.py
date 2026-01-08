from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import re

from . import user_service
from . import chat_service
from . import dataset_service
from . import ai_service
from . import stats_service
from .code_executor import get_executor
from .database import test_connection

app = FastAPI(
    title="AnyGraph API",
    description="Backend API for AnyGraph - Easy Data Analysis Platform",
    version="1.0.0"
)

# Define allowed origins for CORS
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://anygraphfrontend2.vercel.app",
    "https://anygraphfrontend2-m12lkp87a-noahs-projects-c9f6b98e.vercel.app",
    "https://anygraph-backend-production.up.railway.app",
]

# Custom CORS middleware to handle Vercel preview deployments
@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin")
    
    # Check if origin is allowed
    is_allowed = False
    if origin:
        if origin in allowed_origins:
            is_allowed = True
        elif origin.endswith(".vercel.app"):
            is_allowed = True
    
    # Handle preflight OPTIONS requests
    if request.method == "OPTIONS":
        if is_allowed and origin:
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                }
            )
        else:
            return Response(status_code=400)
    
    response = await call_next(request)
    
    if is_allowed and origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response

# Standard CORS middleware as fallback
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserLogin(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None


class UserUpdate(BaseModel):
    full_name: Optional[str] = None


class ChatSessionCreate(BaseModel):
    email: EmailStr
    title: str = "New Chat"


class ChatSessionUpdate(BaseModel):
    title: str


class MessageCreate(BaseModel):
    chat_session_id: str
    sender: str
    message_txt: str


class DatasetAnalyze(BaseModel):
    dataset_url: str
    email: EmailStr
    chat_session_id: str
    name: str
    file_type: Optional[str] = None


class QueryExecute(BaseModel):
    query: str
    dataset_url: str
    chat_session_id: str


class ChatMessage(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {
        "message": "AnyGraph API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health")
def health_check():
    db_status = test_connection()
    executor_status = get_executor().test_docker()  # Method name kept for backwards compatibility

    return {
        "status": "healthy" if (db_status and executor_status) else "unhealthy",
        "database": "connected" if db_status else "disconnected",
        "executor": "available" if executor_status else "unavailable",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/users/login", status_code=status.HTTP_200_OK)
def login_user(user: UserLogin):
    try:
        user_data = user_service.add_or_login_user(user.email, user.full_name)
        return {
            "message": "Login successful",
            "user": user_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to login user: {str(e)}"
        )


@app.get("/users/{email}")
def get_user(email: str):
    try:
        user_data = user_service.get_user(email)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user: {str(e)}"
        )


@app.get("/users/{email}/data")
def get_user_data(email: str):
    try:
        user_data = user_service.get_user_with_chat_sessions(email)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user data: {str(e)}"
        )


@app.get("/users/{email}/stats")
def get_user_stats(email: str):
    try:
        if not user_service.user_exists(email):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        stats = stats_service.get_user_stats(email)
        return {
            "email": email,
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user stats: {str(e)}"
        )


@app.put("/users/{email}")
def update_user(email: str, user_update: UserUpdate):
    try:
        user_data = user_service.update_user(email, user_update.full_name)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return {
            "message": "User updated successfully",
            "user": user_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}"
        )


@app.post("/chat-sessions", status_code=status.HTTP_201_CREATED)
def create_chat_session(session: ChatSessionCreate):
    try:
        if not user_service.user_exists(session.email):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        session_data = chat_service.create_chat_session(session.email, session.title)
        return {
            "message": "Chat session created successfully",
            "session": session_data
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error creating chat session: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chat session: {str(e)}"
        )


@app.get("/chat-sessions/{session_id}")
def get_chat_session(session_id: str):
    try:
        session_data = chat_service.get_chat_session(session_id)
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        return session_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat session: {str(e)}"
        )


@app.get("/chat-sessions/{session_id}/full")
def get_chat_session_full(session_id: str, email: Optional[str] = None):
    try:
        session_data = chat_service.get_chat_session_with_messages(session_id)
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )

        if email and not chat_service.verify_session_owner(session_id, email):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this session"
            )

        return session_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat session: {str(e)}"
        )


@app.get("/users/{email}/chat-sessions")
def get_user_sessions(email: str):
    try:
        sessions = chat_service.get_user_chat_sessions(email)
        return {
            "email": email,
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat sessions: {str(e)}"
        )


@app.put("/chat-sessions/{session_id}")
def update_chat_session(session_id: str, update: ChatSessionUpdate):
    try:
        session_data = chat_service.update_chat_session_title(session_id, update.title)
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        return {
            "message": "Chat session updated successfully",
            "session": session_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update chat session: {str(e)}"
        )


@app.delete("/chat-sessions/{session_id}")
def delete_chat_session(session_id: str):
    try:
        deleted = chat_service.delete_chat_session(session_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        return {
            "message": "Chat session deleted successfully",
            "session_id": session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chat session: {str(e)}"
        )


@app.post("/messages", status_code=status.HTTP_201_CREATED)
def add_message(message: MessageCreate):
    try:
        session = chat_service.get_chat_session(message.chat_session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )

        message_data = chat_service.add_message(
            message.chat_session_id,
            message.sender,
            message.message_txt
        )
        return {
            "message": "Message added successfully",
            "data": message_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add message: {str(e)}"
        )


@app.get("/chat-sessions/{session_id}/messages")
def get_messages(session_id: str):
    try:
        messages = chat_service.get_messages(session_id)
        return {
            "session_id": session_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get messages: {str(e)}"
        )


@app.post("/datasets/analyze", status_code=status.HTTP_201_CREATED)
def analyze_dataset(dataset: DatasetAnalyze):
    try:
        if not user_service.user_exists(dataset.email):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        session = chat_service.get_chat_session(dataset.chat_session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )

        result = dataset_service.analyze_dataset(
            dataset.dataset_url,
            dataset.email,
            dataset.chat_session_id,
            dataset.name,
            dataset.file_type
        )

        return {
            "message": "Dataset analyzed successfully",
            **result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze dataset: {str(e)}"
        )


@app.get("/datasets")
def get_dataset(dataset_url: str):
    try:
        dataset = dataset_service.get_dataset(dataset_url)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        return dataset
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset: {str(e)}"
        )


@app.get("/chat-sessions/{session_id}/datasets")
def get_session_datasets(session_id: str):
    try:
        datasets = dataset_service.get_session_datasets(session_id)
        return {
            "session_id": session_id,
            "datasets": datasets,
            "count": len(datasets)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get datasets: {str(e)}"
        )


@app.get("/datasets/columns")
def get_dataset_columns(dataset_url: str):
    try:
        columns = dataset_service.get_dataset_columns(dataset_url)
        if not columns:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found or has no columns"
            )
        return {
            "dataset_url": dataset_url,
            "columns": columns,
            "count": len(columns)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get columns: {str(e)}"
        )


@app.delete("/datasets")
def delete_dataset(dataset_url: str):
    try:
        deleted = dataset_service.delete_dataset(dataset_url)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        return {
            "message": "Dataset deleted successfully",
            "dataset_url": dataset_url
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete dataset: {str(e)}"
        )


@app.get("/datasets/observations")
def get_dataset_observations(dataset_url: str, limit: int = 100, offset: int = 0):
    try:
        observations_data = dataset_service.get_dataset_observations(
            dataset_url, limit=limit, offset=offset
        )
        return observations_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset observations: {str(e)}"
        )


@app.post("/query/execute")
def execute_query(query_request: QueryExecute):
    try:
        columns = dataset_service.get_dataset_columns(query_request.dataset_url)
        if not columns:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found. Please analyze the dataset first."
            )

        conversation_history = chat_service.get_messages(query_request.chat_session_id)

        chat_service.add_message(
            query_request.chat_session_id,
            "user",
            query_request.query
        )

        try:
            result = ai_service.process_query(
                query_request.query,
                columns,
                query_request.dataset_url,
                conversation_history
            )

            if result["needs_code"]:
                code = result["code"]
                executor = get_executor()
                execution_result = executor.execute_code(code)

                if execution_result["success"]:
                    response_text = execution_result['output']
                    
                    # Extract base64 images from markdown
                    image_pattern = r'!\[Chart\]\(data:image/png;base64,([A-Za-z0-9+/=]+)\)'
                    images = re.findall(image_pattern, response_text)
                    
                    # Remove image markdown from text (optional - keep if you want both)
                    # response_text_clean = re.sub(image_pattern, '', response_text).strip()
                else:
                    response_text = f"Error: {execution_result['error']}"
                    images = []

                chat_service.add_message(
                    query_request.chat_session_id,
                    "assistant",
                    response_text
                )

                return {
                    "query": query_request.query,
                    "code": code,
                    "execution": execution_result,
                    "response": response_text,
                    "images": images  # Array of base64 image strings
                }
            else:
                response_text = result["response"]
                chat_service.add_message(
                    query_request.chat_session_id,
                    "assistant",
                    response_text
                )

                return {
                    "query": query_request.query,
                    "code": None,
                    "execution": None,
                    "response": response_text,
                    "images": []  # No images for text responses
                }

        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            chat_service.add_message(
                query_request.chat_session_id,
                "system",
                error_msg
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute query: {str(e)}"
        )


@app.post("/query/execute/stream")
def execute_query_stream(query_request: QueryExecute):
    columns = dataset_service.get_dataset_columns(query_request.dataset_url)
    if not columns:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found. Please analyze the dataset first."
        )

    conversation_history = chat_service.get_messages(query_request.chat_session_id)

    context_messages = []
    for msg in conversation_history:
        role = "User" if msg["sender"] == "user" else "Assistant"
        context_messages.append(f"{role}: {msg['message_txt']}")
    conversation_context = "\n".join(context_messages) if context_messages else "No previous conversation."

    chat_service.add_message(
        query_request.chat_session_id,
        "user",
        query_request.query
    )

    try:
        result = ai_service.process_query(
            query_request.query,
            columns,
            query_request.dataset_url,
            conversation_history
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )

    session_id = query_request.chat_session_id

    if result["needs_code"]:
        def generate_with_code():
            try:
                # Code already generated by process_query - no additional API call needed
                clean_code = result["code"]
                
                # Stream the code to frontend
                yield f"data: {json.dumps({'type': 'code_complete', 'code': clean_code})}\n\n"
                yield f"data: {json.dumps({'type': 'executing'})}\n\n"

                executor = get_executor()
                execution_result = executor.execute_code(clean_code)

                if execution_result["success"]:
                    response_text = execution_result['output']
                else:
                    response_text = f"Error: {execution_result['error']}"

                chat_service.add_message(session_id, "assistant", response_text, clean_code)

                yield f"data: {json.dumps({'type': 'result', 'content': response_text})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'full_response': response_text, 'generated_code': clean_code})}\n\n"

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"

        return StreamingResponse(
            generate_with_code(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        def generate_stream():
            try:
                # Response already generated by process_query - no additional API call needed
                complete_response = result["response"]
                
                # Stream the response to frontend
                yield f"data: {json.dumps({'type': 'chunk', 'content': complete_response})}\n\n"
                
                chat_service.add_message(session_id, "assistant", complete_response)
                yield f"data: {json.dumps({'type': 'done', 'full_response': complete_response})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )


@app.post("/chat")
def chat(message: ChatMessage):
    try:
        response = ai_service.generate_chat_response(message.message)
        return {
            "query": message.message,
            "response": response
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {
        "error": "Internal server error",
        "detail": str(exc)
    }
