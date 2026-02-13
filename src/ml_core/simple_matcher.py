import re
import logging
from typing import List, Dict

class SimpleMatcher:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def find_exact_duplicates(self, questions: List[Dict]) -> List[List[Dict]]:
        """Find questions with identical text (exact matches)"""
        groups = {}
        
        for question in questions:
            # Normalize text for comparison
            normalized = self._normalize_text(question['text'])
            
            if normalized not in groups:
                groups[normalized] = []
            groups[normalized].append(question)
        
        # Return only groups with more than one question
        duplicates = [group for group in groups.values() if len(group) > 1]
        
        self.logger.info(f"Found {len(duplicates)} duplicate groups")
        return duplicates
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove question numbers from start
        text = re.sub(r'^\d+[\.\)]\s*', '', text)
        
        # Remove marks notation
        text = re.sub(r'\[\d+\s*marks?\]', '', text)
        text = re.sub(r'\(\d+\s*marks?\)', '', text)
        text = re.sub(r'\d+\s*marks?', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation for better matching
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def generate_frequency_stats(self, duplicate_groups: List[List[Dict]]) -> Dict:
        """Generate basic statistics about duplicates"""
        total_duplicates = sum(len(group) for group in duplicate_groups)
        
        stats = {
            'total_groups': len(duplicate_groups),
            'total_duplicate_questions': total_duplicates,
            'largest_group_size': max(len(group) for group in duplicate_groups) if duplicate_groups else 0,
            'average_group_size': total_duplicates / len(duplicate_groups) if duplicate_groups else 0
        }
        
        return stats