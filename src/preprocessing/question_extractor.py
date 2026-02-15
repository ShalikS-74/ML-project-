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
        """More aggressive OCR question extraction."""
        patterns = [
            # Standard numbered questions (more flexible)
            r'(?:^|\n)\s*(\d+)\s*[\.)\]]\s*(.*?)(?=\n\s*\d+\s*[\.)\]]|\Z)',
            # Questions with "OR" alternatives
            r'(?:^|\n)\s*(\d+)\s*[\.)\]]\s*(.*?)(?:\n\s*OR\s*|\n\s*\d+\s*[\.)\]]|\Z)',
            # Parts like "a)", "b)", "c)" within questions
            r'(?:^|\n)\s*([a-z])\s*\)\s*(.*?)(?=\n\s*[a-z]\s*\)|\n\s*\d+|\Z)',
            # Questions starting with keywords
            r'(?:^|\n)((?:Define|Explain|Write|Solve|Find|Calculate|Derive|Prove|Show|Compare).*?)(?=\n(?:Define|Explain|Write|Solve|Find|Calculate|Derive|Prove|Show|Compare)|\n\d+|\Z)',
            # Mark-based with more flexibility
            r'(.*?)[\(\[](\d+)\s*[Mm]arks?[\)\]](.*?)(?=.*?[\(\[]\d+\s*[Mm]arks?|\Z)',
        ]
        structured_questions = self._extract_structured_subparts(text, paper_id)
        best_questions: List[Dict] = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            if not matches:
                continue
            self.logger.info(f"OCR pattern found {len(matches)} candidate chunks")
            pattern_questions = self._build_question_objects(matches, paper_id)
            if len(pattern_questions) > len(best_questions):
                best_questions = pattern_questions

        marks_token_count = len(re.findall(r'[\(\[]\s*\d{1,2}\s*marks?\s*[\)\]]', text, flags=re.IGNORECASE))
        if marks_token_count > 0:
            candidate_sets = [structured_questions, best_questions]
            candidate_sets = [c for c in candidate_sets if c]
            if candidate_sets:
                chosen = min(candidate_sets, key=lambda c: (abs(len(c) - marks_token_count), -len(c)))
                self.logger.info(
                    f"Using OCR set closest to marks-token count {marks_token_count}: {len(chosen)} questions"
                )
                return chosen

        if len(structured_questions) > len(best_questions):
            self.logger.info(
                f"Using structured subpart parser: found {len(structured_questions)} questions"
            )
            return structured_questions

        if best_questions:
            self.logger.info(f"Using best OCR pattern: found {len(best_questions)} questions")
        return best_questions

    def _build_question_objects(self, matches: List, paper_id: str) -> List[Dict]:
        """Build question objects with noise filtering."""
        questions: List[Dict] = []

        for i, raw_match in enumerate(matches):
            match = raw_match if isinstance(raw_match, tuple) else (raw_match,)
            question_num = str(i + 1)
            question_text = ""
            marks: Optional[int] = None

            if len(match) == 2:
                raw_num, raw_text = match
                if str(raw_num).strip().isdigit():
                    question_num = str(raw_num).strip()
                question_text = str(raw_text)
            elif len(match) == 3:
                pre_text, raw_marks, post_text = match
                question_text = f"{pre_text} {post_text}"
                if str(raw_marks).strip().isdigit():
                    marks = int(str(raw_marks).strip())
            else:
                question_text = " ".join(str(part) for part in match)

            if marks is None:
                marks = self._extract_marks(question_text)

            clean_text = self._clean_question_text(question_text)
            if not self._is_valid_question(clean_text):
                continue

            questions.append({
                'id': f"{paper_id}_q{question_num}",
                'text': clean_text,
                'marks': marks,
                'paper': paper_id,
                'number': int(question_num) if question_num.isdigit() else i + 1
            })

        return questions

    def _is_valid_question(self, text: str) -> bool:
        """Filter out noise and keep real questions."""
        if len(text) < 5 or len(text) > 800:
            return False

        noise_patterns = [
            r'^\s*page\s+\d+(\s+of\s+\d+)?\s*$',
            r'^\s*max\.?\s*marks?\s*:?\s*\d+\s*$',
            r'^\s*time\s*:?\s*[\d\.:]+\s*(hrs?|hours?)\s*$',
            r'^\s*note\s*:?\s*$',
            r'^\s*module[-\s]*\d+\s*$',
            r'^\s*or\s*$',
            r'^\s*important\s+note\s*$',
            r'^\s*[^\w]*$',
            r'^\s*continued\s*$',
        ]

        text_lower = text.lower()
        for pattern in noise_patterns:
            if re.search(pattern, text_lower):
                return False

        question_indicators = [
            r'define|explain|write|solve|find|calculate',
            r'derive|prove|show|compare|discuss|analyze',
            r'what|why|how|when|where|which',
            r'draw|sketch|plot|design|implement',
            r'\?|marks?|points?',
            r'^[a-z]\)\s+',
            r'^\([a-z]\)\s+',
            r'^[0-9]+\.\s+',
        ]
        has_indicator = any(re.search(pattern, text_lower) for pattern in question_indicators)
        # Keep short symbolic subparts if they carry structure-like tokens.
        has_structural_hint = bool(re.search(r'[\(\[]\d{1,2}\s*marks?[\)\]]|[a-z]\)|\bq\s*\d+\b', text_lower))
        return has_indicator or has_structural_hint

    def _extract_structured_subparts(self, text: str, paper_id: str) -> List[Dict]:
        """
        Parse question units by scanning lines and capturing each marks-tagged subpart.
        This improves OCR papers where (a)/(b)/(c) labels are inconsistent.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        marks_re = re.compile(r'[\(\[]\s*(\d{1,2})\s*marks?\s*[\)\]]', re.IGNORECASE)
        main_q_re = re.compile(r'^\s*(?:q(?:uestion)?\s*)?(\d{1,2})\s*[\.\):-]\s*(.*)$', re.IGNORECASE)
        subpart_re = re.compile(r'^\s*(?:\(?([a-z])\)|([a-z])\))\s*(.*)$', re.IGNORECASE)

        questions: List[Dict] = []
        seen_text = set()
        unmarked_candidates: List[Dict] = []
        trailing_marks: List[int] = []
        current_main = None
        pending_subpart: Optional[Dict] = None

        for idx, line in enumerate(lines):
            main_match = main_q_re.match(line)
            if main_match:
                try:
                    current_main = int(main_match.group(1))
                except ValueError:
                    current_main = None

            subpart_match = subpart_re.match(line)
            if subpart_match:
                sub_label = next((g for g in subpart_match.groups()[:2] if g), "")
                sub_text = subpart_match.group(3).strip()
                if sub_text:
                    pending_subpart = {
                        "main": current_main,
                        "label": sub_label,
                        "text": sub_text,
                        "line_idx": idx,
                    }

            marks_match = marks_re.search(line)
            if not marks_match and not pending_subpart:
                clean_line = self._clean_question_text(line)
                if self._is_valid_question(clean_line):
                    unmarked_candidates.append({
                        "number": current_main if current_main is not None else (len(unmarked_candidates) + 1),
                        "text": clean_line
                    })
                continue

            marks = int(marks_match.group(1)) if marks_match else None
            candidate = marks_re.sub("", line).strip(" .;:-") if marks_match else ""

            if pending_subpart and (marks_match or idx == pending_subpart["line_idx"] + 1):
                candidate = pending_subpart["text"]
                if marks_match and idx == pending_subpart["line_idx"] + 1:
                    # marks are in following line; keep subpart text
                    pass
                pending_subpart = None

            marks_only_line = marks_match is not None and not candidate.strip()
            if marks_only_line:
                trailing_marks.append(marks)
                continue

            if len(candidate) < 12 and idx > 0:
                prev = lines[idx - 1]
                candidate = f"{prev} {candidate}".strip()
                candidate = marks_re.sub("", candidate).strip(" .;:-")

            clean_text = self._clean_question_text(candidate)
            if not clean_text:
                continue
            if re.search(r'^(module|or|note|time|max\.?\s*marks|usn)\b', clean_text, re.IGNORECASE):
                continue
            if re.search(r'(malpractice|revealing of identification|remaining blank pages)', clean_text, re.IGNORECASE):
                continue
            min_len = 3 if marks is not None else 15
            if len(clean_text) < min_len:
                continue

            dedupe_key = re.sub(r'\s+', ' ', clean_text.lower())
            if dedupe_key in seen_text:
                continue
            seen_text.add(dedupe_key)

            q_number = current_main if current_main is not None else (len(questions) + 1)
            questions.append({
                'id': f"{paper_id}_q{len(questions) + 1}",
                'text': clean_text,
                'marks': marks,
                'paper': paper_id,
                'number': q_number
            })

        # OCR often places marks in a right-side column at page end.
        # Pair trailing marks-only lines with preceding unmarked question lines.
        if trailing_marks and unmarked_candidates:
            count = min(len(trailing_marks), len(unmarked_candidates))
            candidate_slice = unmarked_candidates[-count:]
            mark_slice = trailing_marks[-count:]
            for candidate, mark in zip(candidate_slice, mark_slice):
                clean_text = candidate["text"]
                dedupe_key = re.sub(r'\s+', ' ', clean_text.lower())
                if dedupe_key in seen_text:
                    continue
                seen_text.add(dedupe_key)
                questions.append({
                    'id': f"{paper_id}_q{len(questions) + 1}",
                    'text': clean_text,
                    'marks': mark,
                    'paper': paper_id,
                    'number': candidate["number"]
                })

        marks_token_count = len(re.findall(r'[\(\[]\s*\d{1,2}\s*marks?\s*[\)\]]', text, flags=re.IGNORECASE))
        if marks_token_count > 0 and len(questions) > marks_token_count:
            self.logger.info(
                f"Capping structured extraction from {len(questions)} to marks-token count {marks_token_count}"
            )
            questions = questions[:marks_token_count]

        return questions

    
    def _clean_question_text(self, text: str) -> str:
        """Clean question text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove marks notation from main text
        text = re.sub(r'\[\d+\s*marks?\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\(\d+\s*marks?\)', '', text, flags=re.IGNORECASE)
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
