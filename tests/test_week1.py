"""
Test cases for Week 1 MVP
Basic validation of PDF processing and question extraction
"""

import pytest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.pdf_processor import PDFProcessor
from src.preprocessing.question_extractor import QuestionExtractor
from src.ml_core.simple_matcher import SimpleMatcher

class TestWeek1MVP:
    def setup_method(self):
        """Setup test fixtures"""
        self.pdf_processor = PDFProcessor()
        self.question_extractor = QuestionExtractor()
        self.simple_matcher = SimpleMatcher()
    
    def test_pdf_extraction(self):
        """Test basic PDF text extraction"""
        # TODO: Add test PDF files to test_data/
        # TODO: Implement text extraction validation
        pass
    
    def test_question_detection(self):
        """Test question boundary detection"""
        sample_text = """
        1. What is the capital of France? [5 marks]
        
        2. Explain the concept of integration. [10 marks]
        
        3. Define photosynthesis and describe its importance. [8 marks]
        """
        
        questions = self.question_extractor.extract_simple_questions(sample_text, "test_paper")
        
        assert len(questions) == 3
        assert questions[0]['marks'] == 5
        assert questions[1]['marks'] == 10
        assert questions[2]['marks'] == 8
        assert "France" in questions[0]['text']
        assert "integration" in questions[1]['text']
    
    def test_exact_duplicate_detection(self):
        """Test exact text matching"""
        questions = [
            {'id': 'p1_q1', 'text': 'What is photosynthesis?', 'paper': 'P1'},
            {'id': 'p2_q1', 'text': 'What is photosynthesis?', 'paper': 'P2'},
            {'id': 'p1_q2', 'text': 'Explain integration.', 'paper': 'P1'},
            {'id': 'p3_q1', 'text': 'What is photosynthesis?', 'paper': 'P3'},
        ]
        
        duplicates = self.simple_matcher.find_exact_duplicates(questions)
        
        assert len(duplicates) == 1  # One duplicate group
        assert len(duplicates[0]) == 3  # Three questions about photosynthesis
    
    def test_marks_extraction(self):
        """Test marks extraction patterns"""
        test_cases = [
            ("Question text [5 marks]", 5),
            ("Question text (10 marks)", 10),
            ("Question text 15 marks", 15),
            ("Question text", None),
        ]
        
        for text, expected_marks in test_cases:
            marks = self.question_extractor.extract_marks(text)
            assert marks == expected_marks

if __name__ == "__main__":
    pytest.main([__file__])