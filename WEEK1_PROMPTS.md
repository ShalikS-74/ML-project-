"""
======================== WEEK 1 MVP IMPLEMENTATION PROMPTS ========================

OBJECTIVE: Create a working MVP that can extract questions and find exact duplicates
DELIVERABLE: Console application that processes 3-4 papers and outputs duplicate groups

=== TASK 1: Basic PDF Text Extraction ===
PROMPT FOR CODEX:

Implement a simple PDF text extractor that:
1. Uses pdfplumber as primary method (more reliable than PyPDF2)
2. Falls back to PyPDF2 if pdfplumber fails
3. Handles basic error cases (file not found, corrupted PDF)
4. Returns clean text without complex OCR

Code this function in pdf_processor.py:

def extract_simple_text(pdf_path: Path) -> str:
    '''Extract text from PDF using pdfplumber and PyPDF2 fallback'''
    # Try pdfplumber first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except:
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except:
            raise Exception(f"Could not extract text from {pdf_path}")

Test with sample papers and ensure text extraction works reliably.

=== TASK 2: Simple Question Detection ===
PROMPT FOR CODEX:

Implement basic question extraction using regex patterns.
Focus on common formats: "1.", "Q1)", "Question 1:"

Code this function in question_extractor.py:

def extract_simple_questions(text: str, paper_id: str) -> List[Dict]:
    '''Extract questions using basic regex patterns'''
    questions = []
    
    # Pattern: "1." or "1)" followed by text until next question
    pattern = r'(?:^|\n)(\d+)[\.\)]\s*(.*?)(?=\n\d+[\.\)]|\Z)'
    matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
    
    for question_num, question_text in matches:
        # Clean the question text
        clean_text = re.sub(r'\s+', ' ', question_text.strip())
        
        # Extract marks if present
        marks_match = re.search(r'\[(\d+)\s*marks?\]', clean_text)
        marks = int(marks_match.group(1)) if marks_match else None
        
        questions.append({
            'id': f"{paper_id}_q{question_num}",
            'text': clean_text,
            'marks': marks,
            'paper': paper_id,
            'number': int(question_num)
        })
    
    return questions

Test with real exam papers to ensure question boundaries are detected correctly.

=== TASK 3: Exact Match Duplicate Detection ===
PROMPT FOR CODEX:

Implement simple exact text matching for duplicate detection.
This handles identical questions before we add ML similarity.

Code this function in a new file: src/ml_core/simple_matcher.py

def find_exact_duplicates(questions: List[Dict]) -> List[List[Dict]]:
    '''Find questions with identical text (exact matches)'''
    groups = {}
    
    for question in questions:
        # Normalize text for comparison
        normalized = question['text'].lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        
        if normalized not in groups:
            groups[normalized] = []
        groups[normalized].append(question)
    
    # Return only groups with more than one question
    duplicates = [group for group in groups.values() if len(group) > 1]
    return duplicates

=== TASK 4: Simple Console Output ===
PROMPT FOR CODEX:

Create a simple console output that shows found duplicates.
This replaces PDF generation for Week 1 MVP.

Code this function in src/output/console_output.py:

def display_duplicates(duplicate_groups: List[List[Dict]]):
    '''Display duplicate groups in console'''
    print(f"\n🔍 DUPLICATE ANALYSIS RESULTS")
    print(f"=" * 50)
    
    if not duplicate_groups:
        print("✅ No exact duplicates found")
        return
    
    total_duplicates = sum(len(group) for group in duplicate_groups)
    print(f"📊 Found {len(duplicate_groups)} duplicate groups")
    print(f"📈 Total duplicate questions: {total_duplicates}")
    
    for i, group in enumerate(duplicate_groups, 1):
        print(f"\n🔗 GROUP {i} ({len(group)} questions):")
        print("-" * 30)
        
        for question in group:
            marks_info = f" [{question['marks']} marks]" if question['marks'] else ""
            print(f"  📝 {question['paper']} Q{question['number']}{marks_info}")
        
        # Show the question text (truncated)
        sample_text = group[0]['text'][:100] + "..." if len(group[0]['text']) > 100 else group[0]['text']
        print(f"  💬 Text: {sample_text}")

=== TASK 5: Week 1 Main Function ===
PROMPT FOR CODEX:

Update main.py to implement Week 1 MVP workflow:

def main():
    # Modified main function for Week 1 MVP
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='Week 1 MVP - Exact Duplicate Detection')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with PDF files')
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return
    
    # Initialize processors
    pdf_processor = PDFProcessor()
    question_extractor = QuestionExtractor()
    
    all_questions = []
    
    # Process each PDF
    for pdf_file in input_path.glob("*.pdf"):
        logger.info(f"Processing: {pdf_file.name}")
        
        try:
            # Extract text
            text = pdf_processor.extract_simple_text(pdf_file)
            
            # Extract questions
            paper_id = pdf_file.stem
            questions = question_extractor.extract_simple_questions(text, paper_id)
            all_questions.extend(questions)
            
            logger.info(f"Extracted {len(questions)} questions from {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
    
    # Find duplicates
    from src.ml_core.simple_matcher import find_exact_duplicates
    from src.output.console_output import display_duplicates
    
    duplicates = find_exact_duplicates(all_questions)
    display_duplicates(duplicates)
    
    # Summary stats
    total_questions = len(all_questions)
    unique_questions = total_questions - sum(len(group)-1 for group in duplicates)
    
    print(f"\n📊 SUMMARY:")
    print(f"Total questions processed: {total_questions}")
    print(f"Unique questions: {unique_questions}")
    print(f"Duplicate questions: {total_questions - unique_questions}")

TESTING INSTRUCTIONS:
1. Create test folder with 3-4 sample PDFs
2. Run: python main.py --input_dir test_papers/
3. Verify question extraction and duplicate detection works
4. Debug any regex issues with question patterns
"""