"""LLM integration using OpenRouter API."""
from openai import OpenAI
from config import Config
from typing import Dict
import json
from schema_validator import validate_esg_json

class LLMIntegration:
    """LLM integration for converting ESG content to JSON schema."""
    
    def __init__(self):
        """Initialize OpenRouter client."""
        Config.validate()
        self.api_key = Config.OPENROUTER_API_KEY
        # OpenRouter uses OpenAI-compatible API
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
    
    def convert_to_json(self, weighted_content: str, metadata: Dict) -> Dict:
        """
        Convert weighted ESG content to fixed JSON schema using single LLM call.
        """
        schema_prompt = """You are an ESG data extraction assistant. Convert the provided weighted ESG content into a structured JSON format following this EXACT schema:

{{
  "metadata": {{
    "source_files": ["list of filenames"],
    "extraction_date": "ISO date string",
    "awfa_weights_preserved": true
  }},
  "environmental": {{
    "narrative": "string - consolidated narrative from all environmental content",
    "metrics": [
      {{
        "name": "metric name",
        "value": "metric value (as string)",
        "unit": "unit of measurement",
        "year": "year (as string or null)",
        "source_text": "original text where metric was found (optional)"
      }}
    ],
    "confidence_score": 0.0-1.0
  }},
  "social": {{
    "narrative": "string - consolidated narrative from all social content",
    "metrics": [
      {{
        "name": "metric name",
        "value": "metric value (as string)",
        "unit": "unit of measurement",
        "year": "year (as string or null)",
        "source_text": "original text where metric was found (optional)"
      }}
    ],
    "confidence_score": 0.0-1.0
  }},
  "governance": {{
    "narrative": "string - consolidated narrative from all governance content",
    "metrics": [
      {{
        "name": "metric name",
        "value": "metric value (as string)",
        "unit": "unit of measurement",
        "year": "year (as string or null)",
        "source_text": "original text where metric was found (optional)"
      }}
    ],
    "confidence_score": 0.0-1.0
  }},
  "aggregation_info": {{
    "total_documents": 0,
    "awfa_applied": true,
    "weighted_content_blocks": 0
  }}
}}

CRITICAL RULES:
1. DO NOT invent or make up data - only extract what is actually present
2. DO NOT summarize metrics - extract them as-is
3. DO NOT normalize or convert units unless explicitly stated
4. Extract metrics from the content (numbers with units, percentages, etc.)
5. Narrative should be a coherent summary of the weighted content
6. Confidence scores should reflect the quality and completeness of extracted data
7. Return ONLY valid JSON, no markdown formatting, no code blocks

Weighted ESG Content:
{content}

Source Files: {files}
"""

        try:
            # Try multiple free models in order of preference (using verified available models)
            free_models = [
                "meta-llama/llama-3.2-3b-instruct:free",
                "google/gemini-2.0-flash-exp:free",
                "qwen/qwen3-4b:free",
                "mistralai/mistral-small-3.1-24b-instruct:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemma-3-12b-it:free",
                "qwen/qwen-2.5-vl-7b-instruct:free"
            ]
            
            last_error = None
            response = None
            used_model = None
            
            for model in free_models:
                try:
                    print(f"Trying model: {model}")
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a precise ESG data extraction assistant. You extract data exactly as provided without invention or summarization."
                            },
                            {
                                "role": "user",
                                "content": schema_prompt.format(
                                    content=weighted_content,
                                    files=", ".join(metadata.get('source_files', []))
                                )
                            }
                        ],
                        temperature=0.1,  # Low temperature for deterministic output
                        max_tokens=4000
                    )
                    used_model = model
                    print(f"Successfully used model: {model}")
                    
                    # Validate response before proceeding
                    if not hasattr(response, 'choices') or not response.choices:
                        print(f"Warning: Model {model} returned invalid response structure (no choices)")
                        last_error = ValueError(f"Invalid response structure from {model}: no choices")
                        continue
                    
                    if not hasattr(response.choices[0], 'message'):
                        print(f"Warning: Model {model} returned invalid response structure (no message)")
                        last_error = ValueError(f"Invalid response structure from {model}: no message")
                        continue
                    
                    if not hasattr(response.choices[0].message, 'content') or response.choices[0].message.content is None:
                        print(f"Warning: Model {model} returned empty content")
                        last_error = ValueError(f"Empty content from {model}")
                        continue
                    
                    # If we get here, response is valid
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    # Check if it's a rate limit error (429) - we'll still try next model
                    if "429" in error_str or "rate" in error_str.lower() or "rate-limited" in error_str.lower():
                        print(f"Model {model} is rate-limited, trying next model...")
                    else:
                        print(f"Model {model} failed: {error_str}")
                    # Try next model
                    continue
            
            if response is None:
                # If all models failed, raise the last error with helpful message
                error_msg = f"All free models failed. Last error: {str(last_error)}"
                if last_error:
                    error_msg += f"\nTried models: {', '.join(free_models)}"
                raise ValueError(error_msg)
            
            # Response should be validated by now, but add safety check
            try:
                response_text = response.choices[0].message.content.strip()
            except (AttributeError, IndexError, TypeError) as e:
                raise ValueError(f"Invalid response structure from model {used_model}: {str(e)}")
            
            if not response_text:
                raise ValueError(f"Empty response from model {used_model}: No text content returned")
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate and ensure all required fields are present
            result = self._validate_schema(result, metadata)
            
            # Use schema validator for final validation
            try:
                result = validate_esg_json(result)
            except ValueError as e:
                # If schema validation fails, fall back to basic validation
                print(f"Warning: Schema validation failed, using basic validation: {e}")
                result = self._validate_schema(result, metadata)
            
            return result
            
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {str(e)}")
        except Exception as e:
            raise ValueError(f"LLM conversion failed: {str(e)}")
    
    def _validate_schema(self, data: Dict, metadata: Dict) -> Dict:
        """Validate and ensure schema completeness."""
        # Ensure all top-level keys exist
        required_keys = ['metadata', 'environmental', 'social', 'governance', 'aggregation_info']
        for key in required_keys:
            if key not in data:
                data[key] = {}
        
        # Ensure metadata
        if 'metadata' not in data or not isinstance(data['metadata'], dict):
            data['metadata'] = {}
        data['metadata'].update({
            'source_files': metadata.get('source_files', []),
            'extraction_date': metadata.get('extraction_date', ''),
            'awfa_weights_preserved': True
        })
        
        # Ensure ESG sections have required fields
        for section in ['environmental', 'social', 'governance']:
            if section not in data or not isinstance(data[section], dict):
                data[section] = {}
            
            if 'narrative' not in data[section]:
                data[section]['narrative'] = ""
            if 'metrics' not in data[section] or not isinstance(data[section]['metrics'], list):
                data[section]['metrics'] = []
            if 'confidence_score' not in data[section]:
                data[section]['confidence_score'] = 0.0
        
        # Ensure aggregation_info
        if 'aggregation_info' not in data or not isinstance(data['aggregation_info'], dict):
            data['aggregation_info'] = {}
        
        data['aggregation_info'].update({
            'total_documents': metadata.get('total_documents', 0),
            'awfa_applied': True,
            'weighted_content_blocks': metadata.get('weighted_content_blocks', 0)
        })
        
        return data
