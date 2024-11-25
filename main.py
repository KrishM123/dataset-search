from search import search_datasets_for_ml_task
import json
from vectordb import query_index

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from vectordb import query_index
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Dataset Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
async def search_datasets(request: QueryRequest):
    """
    Search for datasets based on the query in request body
    """
    try:
        results = query_index(
            query_text=request.query
        )
        
        # Return structured response with count
        return {
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



