import logging
from typing import List, Dict


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
