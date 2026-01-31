"""ESG content filtering using keyword-based Python logic."""
import re
from typing import List, Dict

# ESG keywords organized by category
ENVIRONMENTAL_KEYWORDS = [
    'carbon', 'emission', 'greenhouse', 'ghg', 'climate', 'renewable', 'energy',
    'sustainability', 'waste', 'water', 'pollution', 'biodiversity', 'ecosystem',
    'environmental', 'environment', 'recycling', 'solar', 'wind', 'hydroelectric',
    'fossil fuel', 'oil', 'gas', 'coal', 'electricity', 'consumption', 'footprint',
    'co2', 'co₂', 'methane', 'nitrogen', 'air quality', 'toxic', 'hazardous',
    'green', 'clean energy', 'efficiency', 'conservation', 'natural resources'
]

SOCIAL_KEYWORDS = [
    'employee', 'workforce', 'labor', 'worker', 'human rights', 'diversity',
    'inclusion', 'equity', 'gender', 'minority', 'discrimination', 'harassment',
    'safety', 'health', 'wellbeing', 'well-being', 'training', 'development',
    'community', 'stakeholder', 'customer', 'consumer', 'privacy', 'data protection',
    'accessibility', 'disability', 'wage', 'salary', 'compensation', 'benefit',
    'union', 'collective bargaining', 'child labor', 'forced labor', 'supply chain',
    'sourcing', 'ethical', 'fair trade', 'philanthropy', 'donation', 'charity',
    'volunteer', 'education', 'healthcare', 'social impact', 'human capital'
]

GOVERNANCE_KEYWORDS = [
    'governance', 'board', 'director', 'executive', 'management', 'leadership',
    'ethics', 'compliance', 'regulation', 'legal', 'audit', 'risk', 'oversight',
    'transparency', 'disclosure', 'accountability', 'stakeholder', 'shareholder',
    'corporate', 'policy', 'procedure', 'code of conduct', 'whistleblower',
    'anti-corruption', 'bribery', 'fraud', 'conflict of interest', 'independence',
    'committee', 'nomination', 'compensation committee', 'audit committee',
    'internal control', 'financial reporting', 'esg', 'sustainability report',
    'annual report', 'proxy', 'voting', 'engagement'
]

def filter_esg_content(text: str) -> Dict[str, List[str]]:
    """
    Filter text for ESG-relevant content using keyword matching.
    Returns dictionary with 'environmental', 'social', 'governance' lists of sentences.
    """
    if not text or not text.strip():
        return {
            'environmental': [],
            'social': [],
            'governance': []
        }
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    environmental_sentences = []
    social_sentences = []
    governance_sentences = []
    
    # Normalize keywords for case-insensitive matching
    env_keywords = [kw.lower() for kw in ENVIRONMENTAL_KEYWORDS]
    soc_keywords = [kw.lower() for kw in SOCIAL_KEYWORDS]
    gov_keywords = [kw.lower() for kw in GOVERNANCE_KEYWORDS]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Check for environmental keywords
        env_matches = sum(1 for kw in env_keywords if kw in sentence_lower)
        if env_matches > 0:
            environmental_sentences.append(sentence)
        
        # Check for social keywords
        soc_matches = sum(1 for kw in soc_keywords if kw in sentence_lower)
        if soc_matches > 0:
            social_sentences.append(sentence)
        
        # Check for governance keywords
        gov_matches = sum(1 for kw in gov_keywords if kw in sentence_lower)
        if gov_matches > 0:
            governance_sentences.append(sentence)
    
    return {
        'environmental': environmental_sentences,
        'social': social_sentences,
        'governance': governance_sentences
    }

def aggregate_esg_content(filtered_results: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """
    Aggregate ESG content from multiple documents.
    """
    aggregated = {
        'environmental': [],
        'social': [],
        'governance': []
    }
    
    for result in filtered_results:
        for category in ['environmental', 'social', 'governance']:
            aggregated[category].extend(result[category])
    
    return aggregated
