# Configuration file for Exam Intelligence Engine

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ML Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.80  # Will tune between 0.75-0.85
CLUSTERING_METHOD = "agglomerative"  # Options: "agglomerative", "dbscan"

# Question extraction patterns
QUESTION_PATTERNS = [
    r'^(\d+)[\.\)]\s*(.*?)(?=^\d+[\.\)]|\Z)',
    r'^Q(\d+)[\.\)]\s*(.*?)(?=^Q\d+[\.\)]|\Z)',
    r'Question\s+(\d+)[\:\.\)]\s*(.*?)(?=Question\s+\d+|\Z)'
]

# Marks extraction patterns
MARKS_PATTERNS = [
    r'\[(\d+)\s*marks?\]',
    r'\((\d+)\s*marks?\)',
    r'(\d+)\s*marks?'
]

# File settings
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_FORMATS = ['.pdf', '.txt']

# Output settings
OUTPUT_FORMAT = "pdf"  # Options: "pdf", "json", "csv"
CONSOLIDATION_LEVELS = [5, 10, 15]  # Mark ranges for grouping