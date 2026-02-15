#!/usr/bin/env python3
"""
Exam Paper Intelligence & Trend Modeling Engine
Entry point for the application
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from config import OUTPUTS_DIR
from src.preprocessing.pdf_processor import PDFProcessor
from src.preprocessing.question_extractor import QuestionExtractor


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("exam_intelligence.log"),
            logging.StreamHandler(),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Exam Paper Intelligence Engine")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Output directory for results")
    parser.add_argument(
        "--mode",
        type=str,
        default="ml",
        choices=["extract", "ml", "full"],
        help="extract: Week 1 exact matching, ml/full: enhanced ML trend analysis",
    )
    parser.add_argument("--tune_eps", action="store_true", help="Tune DBSCAN eps parameter")
    return parser.parse_args()


def process_all_pdfs(
    input_path: Path,
    pdf_processor: PDFProcessor,
    question_extractor: QuestionExtractor,
    logger: logging.Logger,
) -> Tuple[List[Dict], List[str]]:
    """Process all PDFs and extract questions."""
    all_questions: List[Dict] = []
    paper_ids: List[str] = []

    pdf_files = sorted(input_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_path}")
        return all_questions, paper_ids

    logger.info(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")
        try:
            text = pdf_processor.extract_text(pdf_file)
            if not text.strip():
                logger.warning(f"No text extracted from {pdf_file.name}")
                continue

            paper_id = pdf_file.stem
            questions = question_extractor.extract_questions(text, paper_id)
            all_questions.extend(questions)
            if questions:
                paper_ids.append(paper_id)

            logger.info(f"Extracted {len(questions)} questions from {pdf_file.name}")
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")

    return all_questions, paper_ids


def run_week1_mvp(
    input_path: Path,
    pdf_processor: PDFProcessor,
    question_extractor: QuestionExtractor,
    logger: logging.Logger,
):
    """Week 1 MVP: basic extraction and exact duplicate detection."""
    all_questions, _paper_ids = process_all_pdfs(input_path, pdf_processor, question_extractor, logger)
    if not all_questions:
        logger.warning("No questions extracted from any PDF files")
        return

    from src.ml_core.simple_matcher import SimpleMatcher
    from src.output.console_output import ConsoleOutput

    matcher = SimpleMatcher()
    console_output = ConsoleOutput()

    duplicates = matcher.find_exact_duplicates(all_questions)
    console_output.display_duplicates(duplicates)
    console_output.display_summary_stats(all_questions, duplicates)


def run_advanced_processing(
    input_path: Path,
    pdf_processor: PDFProcessor,
    question_extractor: QuestionExtractor,
    logger: logging.Logger,
    args,
):
    """Enhanced ML processing with DBSCAN + TON trend reporting."""
    all_questions, _paper_ids = process_all_pdfs(input_path, pdf_processor, question_extractor, logger)
    if not all_questions:
        logger.warning("No questions extracted")
        return

    from src.ml_core.embedding_engine import EmbeddingEngine
    from src.ml_core.clustering_engine import ClusteringEngine
    from src.output.console_output import ConsoleOutput

    embedding_engine = EmbeddingEngine()
    question_texts = [q["text"] for q in all_questions]
    embeddings = embedding_engine.generate_batch_embeddings(question_texts)

    clustering_engine = ClusteringEngine()
    if getattr(args, "tune_eps", False):
        optimal_eps = clustering_engine.tune_eps_parameter(embeddings, all_questions)
        logger.info(f"Optimal eps: {optimal_eps}")

    cluster_labels = clustering_engine.apply_clustering(embeddings)
    clusters = clustering_engine.generate_cluster_stats(cluster_labels, all_questions)

    console_output = ConsoleOutput()
    console_output.display_ton_results(clusters, all_questions)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    logger.info("Starting Exam Paper Intelligence Engine")

    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return

    pdf_processor = PDFProcessor()
    question_extractor = QuestionExtractor()

    if args.mode == "extract":
        run_week1_mvp(input_path, pdf_processor, question_extractor, logger)
    else:
        run_advanced_processing(input_path, pdf_processor, question_extractor, logger, args)


if __name__ == "__main__":
    main()
