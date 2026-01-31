"""OCR module using Azure Document Intelligence."""
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from config import Config
from typing import Optional

class OCRModule:
    """OCR module for scanned documents and images."""
    
    def __init__(self):
        """Initialize Azure Document Intelligence client."""
        Config.validate()
        self.endpoint = Config.AZURE_DOCINTEL_ENDPOINT
        self.key = Config.AZURE_DOCINTEL_KEY
        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
    
    def extract_text(self, file_content: bytes, content_type: str) -> str:
        """
        Extract text using Azure Document Intelligence OCR.
        Returns raw text only (no layout/table information).
        """
        try:
            # Use analyze_document with prebuilt-read model
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-read",
                analyze_request=file_content,
                content_type=content_type
            )
            result = poller.result()
            
            # Extract only text content
            text_parts = []
            if result.content:
                text_parts.append(result.content)
            
            # Also extract text from pages if available
            if hasattr(result, 'pages') and result.pages:
                for page in result.pages:
                    if hasattr(page, 'lines') and page.lines:
                        for line in page.lines:
                            if hasattr(line, 'content'):
                                text_parts.append(line.content)
            
            extracted_text = "\n".join(text_parts)
            return extracted_text.strip()
            
        except Exception as e:
            raise ValueError(f"OCR extraction failed: {str(e)}")

def get_content_type(filename: str) -> str:
    """Get content type for Azure Document Intelligence."""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return 'application/pdf'
    elif filename_lower.endswith(('.png', '.jpg', '.jpeg')):
        return 'image/png' if filename_lower.endswith('.png') else 'image/jpeg'
    else:
        return 'application/octet-stream'
