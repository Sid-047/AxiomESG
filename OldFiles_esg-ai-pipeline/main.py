"""FastAPI backend for ESG document processing."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import traceback
from datetime import datetime
import os

from config import Config
from document_extractor import extract_text_from_file
from ocr_module import OCRModule, get_content_type
from esg_filter import filter_esg_content, aggregate_esg_content
from awfa import AWFA
from llm_integration import LLMIntegration
from schema_validator import validate_esg_json

app = FastAPI(title="ESG AI Pipeline")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize modules
try:
    Config.validate()
    ocr_module = OCRModule()
    awfa = AWFA()
    llm_integration = LLMIntegration()
except ValueError as e:
    print(f"Warning: Configuration validation failed: {e}")
    ocr_module = None
    awfa = None
    llm_integration = None

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "ESG AI Pipeline API"}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Process uploaded ESG documents and return aggregated JSON.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        # Step 1: Extract content from all files
        extracted_contents = []
        source_files = []
        
        for file in files:
            file_content = await file.read()
            filename = file.filename
            
            source_files.append(filename)
            
            # Extract text (deterministic extraction)
            extracted_text, needs_ocr = extract_text_from_file(file_content, filename)
            
            # Step 2: Use OCR if needed
            if needs_ocr:
                if ocr_module is None:
                    raise HTTPException(
                        status_code=500,
                        detail="OCR module not initialized. Check Azure credentials."
                    )
                
                content_type = get_content_type(filename)
                extracted_text = ocr_module.extract_text(file_content, content_type)
            
            if not extracted_text or not extracted_text.strip():
                # Skip empty files but continue processing others
                continue
            
            extracted_contents.append({
                'filename': filename,
                'text': extracted_text
            })
        
        if not extracted_contents:
            raise HTTPException(
                status_code=400,
                detail="No extractable content found in uploaded files"
            )
        
        # Step 3: Filter for ESG content
        filtered_results = []
        for content in extracted_contents:
            filtered = filter_esg_content(content['text'])
            filtered_results.append(filtered)
        
        # Step 4: Aggregate ESG content
        aggregated_content = aggregate_esg_content(filtered_results)
        
        # Check if we have any ESG content
        total_esg_sentences = (
            len(aggregated_content['environmental']) +
            len(aggregated_content['social']) +
            len(aggregated_content['governance'])
        )
        
        if total_esg_sentences == 0:
            raise HTTPException(
                status_code=400,
                detail="No ESG-relevant content found in uploaded documents"
            )
        
        # Step 5: Apply AWFA
        if awfa is None:
            raise HTTPException(
                status_code=500,
                detail="AWFA module not initialized."
            )
        weighted_content = awfa.apply_awfa(aggregated_content)
        
        # Count weighted blocks
        weighted_blocks_count = (
            len(weighted_content['environmental']) +
            len(weighted_content['social']) +
            len(weighted_content['governance'])
        )
        
        # Format for LLM
        formatted_content = awfa.format_for_llm(weighted_content)
        
        # Step 6: Single LLM call to convert to JSON
        metadata = {
            'source_files': source_files,
            'extraction_date': datetime.utcnow().isoformat() + 'Z',
            'total_documents': len(extracted_contents),
            'weighted_content_blocks': weighted_blocks_count
        }
        
        if llm_integration is None:
            raise HTTPException(
                status_code=500,
                detail="LLM integration not initialized. Check OpenRouter API key."
            )
        
        result_json = llm_integration.convert_to_json(formatted_content, metadata)
        
        # Step 7: Final schema validation
        try:
            result_json = validate_esg_json(result_json)
        except ValueError as e:
            # Log warning but continue with LLM output
            print(f"Warning: Final schema validation failed: {e}")
        
        # Step 8: Return JSON response
        return JSONResponse(content=result_json)
        
    except HTTPException:
        raise
    except Exception as e:
        error_detail = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error processing files: {error_detail}\n{traceback_str}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing files: {error_detail}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
