import logging
from typing import List, Dict

from config import HIGH_PRIORITY_THRESHOLD, MEDIUM_PRIORITY_THRESHOLD


class ConsoleOutput:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def display_duplicates(self, duplicate_groups: List[List[Dict]]) -> None:
        """Display duplicate groups in console format"""
        print("\nDUPLICATE ANALYSIS RESULTS")
        print("=" * 50)

        if not duplicate_groups:
            print("No exact duplicates found")
            return

        total_duplicates = sum(len(group) for group in duplicate_groups)
        print(f"Found {len(duplicate_groups)} duplicate groups")
        print(f"Total duplicate questions: {total_duplicates}")

        for i, group in enumerate(duplicate_groups, 1):
            print(f"\nGROUP {i} ({len(group)} questions):")
            print("-" * 30)

            for question in group:
                marks_info = f" [{question['marks']} marks]" if question['marks'] else ""
                print(f"  - {question['paper']} Q{question['number']}{marks_info}")

            sample_text = group[0]['text']
            display_text = sample_text[:100] + "..." if len(sample_text) > 100 else sample_text
            print(f"  Text: {display_text}")

    def display_summary_stats(self, all_questions: List[Dict], duplicate_groups: List[List[Dict]]) -> None:
        """Display summary statistics"""
        total_questions = len(all_questions)
        duplicate_count = sum(len(group) - 1 for group in duplicate_groups)
        unique_questions = total_questions - duplicate_count

        print("\nSUMMARY:")
        print(f"Total questions processed: {total_questions}")
        print(f"Unique questions: {unique_questions}")
        print(f"Duplicate questions: {duplicate_count}")

        if total_questions > 0:
            duplicate_rate = (duplicate_count / total_questions) * 100
            print(f"Duplicate rate: {duplicate_rate:.1f}%")

        papers = list(set(q['paper'] for q in all_questions))
        print(f"Papers processed: {len(papers)}")
        for paper in sorted(papers):
            count = len([q for q in all_questions if q['paper'] == paper])
            print(f"  - {paper}: {count} questions")

    def display_processing_progress(self, filename: str, question_count: int) -> None:
        """Show progress during processing"""
        print(f"  OK {filename}: {question_count} questions extracted")

    def display_ton_results(self, clusters: List[Dict], all_questions: List[Dict]) -> None:
        """Display TON (Topics of Notice) style trend analysis."""
        print("\n" + "=" * 60)
        print("EXAM TREND INTELLIGENCE - TON ANALYSIS")
        print("=" * 60)

        total_papers = len(set(q["paper"] for q in all_questions)) if all_questions else 0
        high_priority = [c for c in clusters if c.get("trend_score", 0.0) >= HIGH_PRIORITY_THRESHOLD]
        medium_priority = [
            c
            for c in clusters
            if MEDIUM_PRIORITY_THRESHOLD <= c.get("trend_score", 0.0) < HIGH_PRIORITY_THRESHOLD
        ]
        low_frequency = [c for c in clusters if c.get("trend_score", 0.0) < MEDIUM_PRIORITY_THRESHOLD]

        if high_priority:
            print("\nHIGH PRIORITY (Strong Historical Signal)")
            print("-" * 45)
            for cluster in high_priority:
                topic_name = (
                    cluster["topic_keywords"][0]
                    if cluster.get("topic_keywords")
                    else f"topic_{cluster.get('cluster_id')}"
                )
                print(f"\n{topic_name.upper()}")
                print(f"  Appeared in {len(cluster.get('papers_covered', []))}/{total_papers} papers")
                print(f"  Frequency: {cluster.get('frequency', 0)} questions")
                print(f"  Weighted Score: {cluster.get('total_marks', 0)}")
                print(f"  Trend Score: {cluster.get('trend_score', 0.0):.2f}")

                paper_dist = {}
                for question in cluster.get("questions", []):
                    paper = question.get("paper")
                    paper_dist[paper] = paper_dist.get(paper, 0) + 1
                print(f"  Distribution: {paper_dist}")

        if medium_priority:
            print("\nMEDIUM PRIORITY")
            print("-" * 20)
            for cluster in medium_priority:
                topic_name = (
                    cluster["topic_keywords"][0]
                    if cluster.get("topic_keywords")
                    else f"topic_{cluster.get('cluster_id')}"
                )
                print(f"  {topic_name} (Score: {cluster.get('trend_score', 0.0):.2f})")

        if low_frequency:
            print("\nLOW FREQUENCY")
            print("-" * 15)
            for cluster in low_frequency[:5]:
                topic_name = (
                    cluster["topic_keywords"][0]
                    if cluster.get("topic_keywords")
                    else f"topic_{cluster.get('cluster_id')}"
                )
                print(f"  {topic_name}")

        print("\nSUMMARY STATISTICS")
        print("-" * 25)
        print(f"Total questions processed: {len(all_questions)}")
        print(f"Papers analyzed: {total_papers}")
        print(f"Semantic clusters found: {len(clusters)}")
        print(f"High priority topics: {len(high_priority)}")
        cross_paper = len([c for c in clusters if len(c.get('papers_covered', [])) > 1])
        print(f"Cross-paper topics: {cross_paper}")
