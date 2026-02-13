#!/usr/bin/env python3
"""
Exam Paper Intelligence & Trend Modeling Engine
Entry point for the application
"""

import argparse
import logging
from pathlib import Path
from src.preprocessing.pdf_processor import PDFProcessor
from src.preprocessing.question_extractor import QuestionExtractor
from config import *

# Import ML modules conditionally to avoid torch DLL errors in Week 1 mode
def import_ml_modules():
    """Import ML modules only when needed (Week 2+ modes)"""
    from src.ml_core.embedding_engine import EmbeddingEngine
    from src.ml_core.clustering_engine import ClusteringEngine
    from src.output.report_generator import ReportGenerator
    return EmbeddingEngine, ClusteringEngine, ReportGenerator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('exam_intelligence.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='Exam Paper Intelligence Engine')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Directory containing PDF files')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--mode', type=str, default='extract',
                       choices=['extract', 'analyze', 'full'],
                       help='Processing mode')
    
    args = parser.parse_args()
    
    logger.info("Starting Exam Paper Intelligence Engine")
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return
    
    # Initialize processors for Week 1 MVP
    pdf_processor = PDFProcessor()
    question_extractor = QuestionExtractor()
    
    if args.mode == 'extract':
        # Week 1 MVP: Extract questions and find exact duplicates
        run_week1_mvp(input_path, pdf_processor, question_extractor, logger)
    else:
        # Week 2+ modes: Use ML modules
        logger.info("Loading ML modules for advanced processing...")
        try:
            EmbeddingEngine, ClusteringEngine, ReportGenerator = import_ml_modules()
            run_advanced_processing(input_path, EmbeddingEngine, ClusteringEngine, ReportGenerator, logger)
        except ImportError as e:
            logger.error(f"Failed to load ML modules: {e}")
            logger.error("Install sentence-transformers and torch for advanced modes")

def run_week1_mvp(input_path, pdf_processor, question_extractor, logger):
    """Week 1 MVP: Basic extraction and exact duplicate detection"""
    all_questions = []
    
    # Process each PDF
    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_path}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")
        
        try:
            # Extract text
            text = pdf_processor.extract_text(pdf_file)
            
            if not text.strip():
                logger.warning(f"No text extracted from {pdf_file.name} (may be scanned PDF)")
                continue
            
            # Extract questions
            paper_id = pdf_file.stem
            questions = question_extractor.extract_questions(text, paper_id)
            all_questions.extend(questions)
            
            logger.info(f"Extracted {len(questions)} questions from {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
    
    if not all_questions:
        logger.warning("No questions extracted from any PDF files")
        return
    
    # Find exact duplicates
    from src.ml_core.simple_matcher import SimpleMatcher
    from src.output.console_output import ConsoleOutput
    
    matcher = SimpleMatcher()
    console_output = ConsoleOutput()
    
    duplicates = matcher.find_exact_duplicates(all_questions)
    console_output.display_duplicates(duplicates)
    console_output.display_summary_stats(all_questions, duplicates)

def run_advanced_processing(input_path, EmbeddingEngine, ClusteringEngine, ReportGenerator, logger):
    """Week 2+ processing with ML similarity"""
    logger.info("Advanced ML processing not yet implemented")
    logger.info("Use --mode extract for Week 1 MVP functionality")

if __name__ == "__main__":
    main()