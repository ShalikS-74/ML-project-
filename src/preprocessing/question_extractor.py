import re
import logging
from typing import List, Dict, Optional

class QuestionExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.question_patterns = [
            r'(?:^|\n)(\d+)[\.\)]\s*(.*?)(?=\n\d+[\.\)]|\Z)',
            r'(?:^|\n)Q(\d+)[\.\)]\s*(.*?)(?=\nQ\d+[\.\)]|\Z)',
            r'(?:^|\n)Question\s+(\d+)[\:\.\)]\s*(.*?)(?=\nQuestion\s+\d+|\Z)'
        ]
        self.marks_patterns = [
            r'\[(\d+)\s*marks?\]',
            r'\((\d+)\s*marks?\)',
            r'(\d+)\s*marks?'
        ]
        
    def extract_questions(self, text: str, paper_id: str) -> List[Dict]:
        """Enhanced extraction for OCR'd papers."""
        questions = self._try_standard_patterns(text, paper_id)

        # OCR fallback if standard parsing produced too few questions.
        if len(questions) < 2:
            questions = self._try_ocr_patterns(text, paper_id)

        if not questions:
            self.logger.warning(f"No questions found in {paper_id} with any pattern")
        return questions

    def extract_simple_questions(self, text: str, paper_id: str) -> List[Dict]:
        """Backward-compatible Week 1 API."""
        return self.extract_questions(text, paper_id)

    def extract_marks(self, question_text: str) -> Optional[int]:
        """Backward-compatible public marks API."""
        return self._extract_marks(question_text)

    def _try_standard_patterns(self, text: str, paper_id: str) -> List[Dict]:
        """Try baseline question numbering patterns."""
        for pattern in self.question_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            if matches:
                questions = self._build_question_objects(matches, paper_id)
                if questions:
                    self.logger.info(f"Using standard pattern: found {len(questions)} questions")
                    return questions
        return []

    def _try_ocr_patterns(self, text: str, paper_id: str) -> List[Dict]:
        """OCR-specific extraction patterns + fallback heuristics."""
        patterns = [
            r'(?:^|\n)\s*(\d+)\s*[\.)\]]\s*(.*?)(?=\n\s*\d+\s*[\.)\]]|\Z)',
            r'(.*?)[\(\[](\d+)\s*[Mm]arks?[\)\]](.*?)(?=.*?[\(\[]\d+\s*[Mm]arks?|\Z)',
            r'(?:[Qq]uestion|[Qq])\s*(\d+)(?:\s*[:\.)]|\s+)(.*?)(?=(?:[Qq]uestion|[Qq])\s*\d+|\Z)',
            r'\n+\s*(\d+)(?:\s*[\.)]|(?=\s+[A-Z]))(.*?)(?=\n+\s*\d+|\Z)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            questions = self._build_question_objects(matches, paper_id)
            if len(questions) >= 2:
                self.logger.info(f"Using OCR pattern: found {len(questions)} questions")
                return questions

        marks_questions = self._extract_questions_by_marks(text, paper_id)
        if len(marks_questions) >= 2:
            self.logger.info(f"Using marks-based fallback... found {len(marks_questions)} questions")
            return marks_questions

        indicator_questions = self._split_by_indicators(text, paper_id)
        if indicator_questions:
            self.logger.info(f"Using indicator-based fallback... found {len(indicator_questions)} questions")
        return indicator_questions

    def _build_question_objects(self, matches: List, paper_id: str) -> List[Dict]:
        """Build normalized question dictionaries from regex matches."""
        questions: List[Dict] = []
        next_auto_number = 1

        for raw_match in matches:
            match = raw_match if isinstance(raw_match, tuple) else (raw_match,)
            question_num: Optional[int] = None
            question_text = ""
            marks: Optional[int] = None

            if len(match) == 2:
                # (question_number, question_text)
                raw_num, raw_text = match
                if str(raw_num).strip().isdigit():
                    question_num = int(str(raw_num).strip())
                question_text = str(raw_text)
            elif len(match) == 3:
                # OCR marks-style match: (pre_text, marks, post_text)
                pre_text, raw_marks, post_text = match
                question_text = f"{pre_text} {post_text}"
                if str(raw_marks).strip().isdigit():
                    marks = int(str(raw_marks).strip())
            else:
                question_text = " ".join(str(part) for part in match)

            if marks is None:
                marks = self._extract_marks(question_text)
            clean_text = self._clean_question_text(question_text)
            if not clean_text:
                continue
            if re.search(r'^(module|or|note|time|max\.?\s*marks|usn)\b', clean_text, re.IGNORECASE):
                continue
            if len(clean_text) < 15:
                continue

            if question_num is None:
                question_num = next_auto_number
                next_auto_number += 1

            questions.append({
                'id': f"{paper_id}_q{question_num}",
                'text': clean_text,
                'marks': marks,
                'paper': paper_id,
                'number': question_num
            })

        return questions
    
    def _clean_question_text(self, text: str) -> str:
        """Clean question text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove marks notation from main text
        text = re.sub(r'\[\d+\s*marks?\]', '', text)
        text = re.sub(r'\(\d+\s*marks?\)', '', text)
        return text.strip()
    
    def _extract_marks(self, question_text: str) -> Optional[int]:
        """Extract marks from question text"""
        for pattern in self.marks_patterns:
            match = re.search(pattern, question_text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _extract_questions_by_marks(self, text: str, paper_id: str) -> List[Dict]:
        """Fallback extraction: segment question blocks ending with '(xx Marks)'."""
        questions: List[Dict] = []
        pattern = re.compile(r'(.{20,}?)\(\s*(\d{1,2})\s*Marks?\s*\)', re.IGNORECASE | re.DOTALL)

        for idx, match in enumerate(pattern.finditer(text), start=1):
            chunk = match.group(1)
            marks = int(match.group(2))

            # Use the last non-empty line before marks as the likely question statement.
            lines = [line.strip(" .;:-\t") for line in chunk.splitlines() if line.strip()]
            if not lines:
                continue
            candidate = lines[-1]

            # Skip obvious non-question headings/noise.
            if re.search(r'^(module|or|note|time|max\.?\s*marks|usn)\b', candidate, re.IGNORECASE):
                continue
            if len(candidate) < 12:
                continue

            questions.append({
                'id': f"{paper_id}_q{idx}",
                'text': self._clean_question_text(candidate),
                'marks': marks,
                'paper': paper_id,
                'number': idx
            })

        return questions

    def _split_by_indicators(self, text: str, paper_id: str) -> List[Dict]:
        """Split by common exam indicators when regex parsing fails."""
        indicators = ['marks', 'define', 'explain', 'write', 'solve', 'find', 'calculate', 'implement', 'derive']
        blocks: List[str] = []
        current_block = ""

        for line in text.split('\n'):
            cleaned_line = line.strip()
            if not cleaned_line:
                continue

            if any(indicator in cleaned_line.lower() for indicator in indicators):
                if current_block.strip():
                    blocks.append(current_block.strip())
                current_block = cleaned_line
            else:
                current_block = (current_block + " " + cleaned_line).strip()

        if current_block.strip():
            blocks.append(current_block.strip())

        questions: List[Dict] = []
        for i, block in enumerate(blocks[:20], start=1):
            if len(block) < 20:
                continue
            if re.search(r'^(module|or|note|time|max\.?\s*marks|usn)\b', block, re.IGNORECASE):
                continue
            questions.append({
                'id': f"{paper_id}_q{i}",
                'text': self._clean_question_text(block),
                'marks': self._extract_marks(block),
                'paper': paper_id,
                'number': i
            })

        return questions
