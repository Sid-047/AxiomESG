"""Adaptive Weighted Fusion Algorithm (AWFA) for ESG content."""
from typing import List, Dict
import math

class AWFA:
    """Adaptive Weighted Fusion Algorithm for ESG content weighting."""
    
    def __init__(self):
        """Initialize AWFA."""
        pass
    
    def calculate_tf_idf_weight(self, sentence: str, all_sentences: List[str]) -> float:
        """
        Calculate TF-IDF-like weight for a sentence.
        Higher weight for sentences with unique ESG terms.
        """
        if not sentence or not all_sentences:
            return 0.0
        
        words = sentence.lower().split()
        if not words:
            return 0.0
        
        # Term frequency in this sentence
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Inverse document frequency (how rare is this word across all sentences)
        total_sentences = len(all_sentences)
        idf_scores = {}
        
        for word in word_counts.keys():
            # Count how many sentences contain this word
            doc_freq = sum(1 for s in all_sentences if word in s.lower())
            if doc_freq > 0:
                idf_scores[word] = math.log(total_sentences / doc_freq)
            else:
                idf_scores[word] = 0.0
        
        # Calculate TF-IDF for sentence
        tf_idf_sum = 0.0
        for word, tf in word_counts.items():
            idf = idf_scores.get(word, 0.0)
            tf_idf_sum += tf * idf
        
        # Normalize by sentence length
        normalized_weight = tf_idf_sum / len(words) if words else 0.0
        
        return normalized_weight
    
    def calculate_confidence_score(self, sentence: str, category: str) -> float:
        """
        Calculate confidence score based on ESG keyword density.
        Returns value between 0 and 1.
        """
        from esg_filter import ENVIRONMENTAL_KEYWORDS, SOCIAL_KEYWORDS, GOVERNANCE_KEYWORDS
        
        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        
        if not words:
            return 0.0
        
        # Get relevant keywords for category
        if category == 'environmental':
            keywords = [kw.lower() for kw in ENVIRONMENTAL_KEYWORDS]
        elif category == 'social':
            keywords = [kw.lower() for kw in SOCIAL_KEYWORDS]
        elif category == 'governance':
            keywords = [kw.lower() for kw in GOVERNANCE_KEYWORDS]
        else:
            return 0.0
        
        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in sentence_lower)
        
        # Calculate confidence (capped at 1.0)
        confidence = min(1.0, matches / max(1, len(words) / 10))
        
        return confidence
    
    def apply_awfa(self, aggregated_content: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
        """
        Apply Adaptive Weighted Fusion Algorithm to ESG content.
        Returns weighted content blocks with confidence scores.
        """
        weighted_content = {
            'environmental': [],
            'social': [],
            'governance': []
        }
        
        for category in ['environmental', 'social', 'governance']:
            sentences = aggregated_content.get(category, [])
            
            if not sentences:
                continue
            
            # Calculate weights for each sentence
            weighted_sentences = []
            for sentence in sentences:
                # TF-IDF weight
                tf_idf_weight = self.calculate_tf_idf_weight(sentence, sentences)
                
                # Confidence score
                confidence = self.calculate_confidence_score(sentence, category)
                
                # Combined weight (normalized to 0-1 range)
                combined_weight = (tf_idf_weight * 0.5 + confidence * 0.5)
                # Normalize to 0-1 range
                normalized_weight = min(1.0, max(0.0, combined_weight / 2.0))
                
                weighted_sentences.append({
                    'text': sentence,
                    'weight': normalized_weight,
                    'confidence': confidence
                })
            
            # Sort by weight (descending) and deduplicate similar sentences
            weighted_sentences.sort(key=lambda x: x['weight'], reverse=True)
            
            # Simple deduplication: remove very similar sentences
            deduplicated = []
            seen_texts = set()
            for item in weighted_sentences:
                text_lower = item['text'].lower()
                # Check if we've seen a very similar sentence
                is_duplicate = False
                for seen in seen_texts:
                    # Simple similarity check: if 80% of words overlap, consider duplicate
                    words1 = set(text_lower.split())
                    words2 = set(seen.split())
                    if words1 and words2:
                        overlap = len(words1 & words2) / len(words1 | words2)
                        if overlap > 0.8:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    deduplicated.append(item)
                    seen_texts.add(text_lower)
            
            weighted_content[category] = deduplicated
        
        return weighted_content
    
    def format_for_llm(self, weighted_content: Dict[str, List[Dict]]) -> str:
        """
        Format weighted content for LLM processing.
        Returns a structured text representation.
        """
        formatted_parts = []
        
        for category in ['environmental', 'social', 'governance']:
            items = weighted_content.get(category, [])
            if not items:
                continue
            
            formatted_parts.append(f"=== {category.upper()} ===")
            for item in items:
                formatted_parts.append(
                    f"[Weight: {item['weight']:.3f}, Confidence: {item['confidence']:.3f}] "
                    f"{item['text']}"
                )
            formatted_parts.append("")
        
        return "\n".join(formatted_parts)
