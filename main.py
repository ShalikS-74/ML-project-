#!/usr/bin/env python3
"""
Exam Paper Intelligence & Trend Modeling Engine
Entry point for the application
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import OUTPUTS_DIR
from src.preprocessing.pdf_processor import PDFProcessor
from src.preprocessing.question_extractor import QuestionExtractor


def import_ml_modules():
    """Import ML modules only when needed (Week 2+ modes)."""
    from src.ml_core.embedding_engine import EmbeddingEngine
    from src.ml_core.clustering_engine import ClusteringEngine
    from src.output.report_generator import ReportGenerator

    return EmbeddingEngine, ClusteringEngine, ReportGenerator


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
        default="extract",
        choices=["extract", "analyze", "full"],
        help="extract: Week 1 exact matching, analyze/full: Week 2 ML similarity",
    )
    parser.add_argument(
        "--tune_threshold",
        action="store_true",
        help="Tune clustering similarity threshold (Week 2 analyze/full mode)",
    )
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
                logger.warning(f"No text extracted from {pdf_file.name} (may be scanned PDF)")
                continue

            paper_id = pdf_file.stem
            questions = question_extractor.extract_questions(text, paper_id)
            logger.info(f"Extracted {len(questions)} questions from {pdf_file.name}")

            if questions:
                paper_ids.append(paper_id)
                all_questions.extend(questions)
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")

    return all_questions, paper_ids


def run_week1_mvp(
    input_path: Path,
    pdf_processor: PDFProcessor,
    question_extractor: QuestionExtractor,
    logger: logging.Logger,
):
    """Week 1 MVP: Basic extraction and exact duplicate detection."""
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


def display_ml_results(clusters: List[Dict], questions: List[Dict], logger: logging.Logger):
    """Display concise Week 2 clustering summary in console."""
    print("\nML SIMILARITY ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Total questions: {len(questions)}")
    print(f"Total clusters: {len(clusters)}")

    sorted_clusters = sorted(clusters, key=lambda c: c.get("size", 0), reverse=True)
    for cluster in sorted_clusters[:10]:
        cid = cluster.get("cluster_id")
        size = cluster.get("size", 0)
        avg_marks = cluster.get("avg_marks", 0.0)
        similarity = cluster.get("similarity_score", 0.0)
        keywords = ", ".join(cluster.get("topic_keywords", [])) or "n/a"
        print(f"\nCluster {cid}: size={size}, avg_marks={avg_marks:.2f}, similarity={similarity:.2f}")
        print(f"Keywords: {keywords}")
        for q in cluster.get("questions", [])[:3]:
            preview = q.get("text", "")[:120]
            print(f"- {q.get('paper')} Q{q.get('number')}: {preview}")

    logger.info("ML analysis complete")


def run_advanced_processing(
    input_path: Path,
    EmbeddingEngine,
    ClusteringEngine,
    ReportGenerator,
    logger: logging.Logger,
    tune_threshold: bool,
    mode: str,
):
    """Week 2 processing with embedding + clustering."""
    pdf_processor = PDFProcessor()
    question_extractor = QuestionExtractor()
    all_questions, paper_ids = process_all_pdfs(input_path, pdf_processor, question_extractor, logger)

    if not all_questions:
        logger.warning("No questions extracted from any PDF files")
        return

    question_texts = [q["text"] for q in all_questions]
    embedding_engine = EmbeddingEngine()
    embeddings = embedding_engine.generate_batch_embeddings(question_texts)

    clustering_engine = ClusteringEngine()
    if tune_threshold:
        optimal = clustering_engine.tune_threshold(embeddings, all_questions)
        logger.info(f"Optimal threshold selected: {optimal:.2f}")

    cluster_labels = clustering_engine.apply_clustering(embeddings)
    clusters = clustering_engine.generate_cluster_stats(cluster_labels, all_questions, embeddings=embeddings)
    trends = clustering_engine.calculate_weighted_trends(clusters)

    display_ml_results(clusters, all_questions, logger)
    print("\nTOP WEIGHTED TRENDS")
    print("=" * 50)
    for topic, stats in list(trends.items())[:10]:
        print(
            f"{topic}: weighted={stats['weighted_score']:.3f}, "
            f"freq={stats['frequency']:.3f}, avg_marks={stats['avg_marks']:.2f}"
        )

    if mode == "full":
        # Week 3 reporting can plug in here; keep Week 2 full mode non-breaking.
        logger.info("Full mode selected; PDF reporting remains optional until Week 3 implementation.")
        _ = ReportGenerator  # Preserve imported dependency for future usage.
        _ = paper_ids


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    logger.info("Starting Exam Paper Intelligence Engine")
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return

    if args.mode == "extract":
        pdf_processor = PDFProcessor()
        question_extractor = QuestionExtractor()
        run_week1_mvp(input_path, pdf_processor, question_extractor, logger)
        return

    logger.info("Loading ML modules for advanced processing...")
    try:
        EmbeddingEngine, ClusteringEngine, ReportGenerator = import_ml_modules()
        run_advanced_processing(
            input_path,
            EmbeddingEngine,
            ClusteringEngine,
            ReportGenerator,
            logger,
            tune_threshold=args.tune_threshold,
            mode=args.mode,
        )
    except Exception as e:
        logger.error(f"Failed to run advanced processing: {e}")
        logger.error("If this is a torch/sentence-transformers issue, use --mode extract and verify ML deps.")


if __name__ == "__main__":
    main()
