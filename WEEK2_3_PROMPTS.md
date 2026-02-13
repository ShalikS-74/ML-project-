"""
======================= WEEK 2-3 ENHANCEMENT PROMPTS =======================

OBJECTIVE: Add ML-based similarity detection and polish the system
DELIVERABLE: Full-featured system with embeddings, clustering, and PDF output

=============== WEEK 2: Add Intelligence (ML Core) ===============

=== TASK 1: Implement Embedding Engine ===
PROMPT FOR CODEX:

Complete the embedding engine in embedding_engine.py:

def load_model(self) -> None:
    '''Load sentence transformer model'''
    try:
        self.model = SentenceTransformer(self.model_name)
        self.logger.info(f"Loaded model: {self.model_name}")
    except Exception as e:
        self.logger.error(f"Failed to load model: {e}")
        raise

def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
    '''Generate embeddings for multiple texts efficiently'''
    if self.model is None:
        self.load_model()
    
    # Clean texts
    clean_texts = [self.clean_text_for_embedding(text) for text in texts]
    
    # Generate embeddings
    embeddings = self.model.encode(clean_texts, 
                                 batch_size=32,
                                 show_progress_bar=True,
                                 convert_to_numpy=True)
    
    # Normalize for cosine similarity
    return self.normalize_embeddings(embeddings)

def clean_text_for_embedding(self, text: str) -> str:
    '''Clean text before embedding generation'''
    # Remove question numbers and formatting
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
    # Remove marks notation
    text = re.sub(r'\[\d+\s*marks?\]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    return text

Key features:
- Batch processing for efficiency
- Progress bars for user feedback
- Text cleaning before embedding
- Error handling and logging
- Caching for repeated use

=== TASK 2: Implement Clustering Engine ===
PROMPT FOR CODEX:

Complete the clustering implementation in clustering_engine.py:

def apply_clustering(self, embeddings: np.ndarray) -> np.ndarray:
    '''Apply Agglomerative clustering with distance threshold'''
    
    # Calculate distance matrix (1 - cosine similarity)
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    
    # Apply clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=self.similarity_threshold,
        affinity='precomputed',
        linkage='average'
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    self.logger.info(f"Found {len(set(cluster_labels))} clusters")
    
    return cluster_labels

def tune_threshold(self, embeddings: np.ndarray, questions: List[Dict]) -> float:
    '''Test different thresholds and find optimal'''
    thresholds = np.arange(0.75, 0.86, 0.02)
    best_threshold = self.similarity_threshold
    best_score = 0
    
    for threshold in thresholds:
        self.similarity_threshold = threshold
        labels = self.apply_clustering(embeddings)
        
        # Calculate quality score (balance cluster size and coherence)
        score = self.evaluate_clustering_quality(labels, embeddings)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    self.similarity_threshold = best_threshold
    return best_threshold

def extract_topic_keywords(self, questions: List[Dict]) -> List[str]:
    '''Extract representative keywords using TF-IDF'''
    texts = [q['text'] for q in questions]
    
    vectorizer = TfidfVectorizer(
        max_features=5,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top features
    scores = tfidf_matrix.sum(axis=0).A1
    top_indices = scores.argsort()[-5:][::-1]
    
    return [feature_names[i] for i in top_indices]

=== TASK 3: Upgrade Main Pipeline ===
PROMPT FOR CODEX:

Update main.py to use ML-based similarity:

def main_week2():
    '''Week 2 main function with ML similarity'''
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Process all questions (same as Week 1)
    all_questions = process_all_pdfs(args.input_dir)
    
    # Generate embeddings
    embedding_engine = EmbeddingEngine()
    question_texts = [q['text'] for q in all_questions]
    embeddings = embedding_engine.generate_batch_embeddings(question_texts)
    
    # Apply clustering
    clustering_engine = ClusteringEngine()
    
    # Tune threshold if requested
    if args.tune_threshold:
        optimal_threshold = clustering_engine.tune_threshold(embeddings, all_questions)
        logger.info(f"Optimal threshold: {optimal_threshold}")
    
    # Get clusters
    cluster_labels = clustering_engine.apply_clustering(embeddings)
    
    # Generate cluster statistics
    clusters = clustering_engine.generate_cluster_stats(cluster_labels, all_questions)
    
    # Display results
    display_ml_results(clusters, all_questions)

=============== WEEK 3: Polish & Professional Output ===============

=== TASK 1: Professional PDF Generation ===
PROMPT FOR CODEX:

Complete PDF report generation in report_generator.py:

def export_to_pdf(self, report_data: Dict, filename: str) -> str:
    '''Generate professional PDF report'''
    
    doc = SimpleDocTemplate(
        str(self.output_dir / f"{filename}.pdf"),
        pagesize=letter,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=18
    )
    
    # Build story (content)
    story = []
    
    # Title
    story.append(Paragraph("Exam Paper Intelligence Report", 
                          self.styles['Title']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    summary = report_data['executive_summary']
    story.append(Paragraph("Executive Summary", self.styles['Heading1']))
    story.append(Paragraph(f"Papers Analyzed: {summary['paper_count']}", 
                          self.styles['Normal']))
    story.append(Paragraph(f"Questions Processed: {summary['question_count']}", 
                          self.styles['Normal']))
    story.append(Paragraph(f"Clusters Found: {summary['cluster_count']}", 
                          self.styles['Normal']))
    
    # Question Groups
    for mark_range in [5, 10, 15]:
        story.append(PageBreak())
        story.append(Paragraph(f"{mark_range}+ Mark Questions", 
                              self.styles['Heading1']))
        
        groups = report_data['question_groups'][f"{mark_range}M"]
        self.add_question_groups_to_story(story, groups)
    
    # Build PDF
    doc.build(story)
    return str(self.output_dir / f"{filename}.pdf")

=== TASK 2: Add Weighted Trend Analysis ===
PROMPT FOR CODEX:

Implement trend scoring in clustering_engine.py:

def calculate_weighted_trends(self, clusters: List[Dict]) -> Dict:
    '''Calculate frequency and marks-weighted trends'''
    
    total_questions = sum(len(cluster['questions']) for cluster in clusters)
    trends = {}
    
    for cluster in clusters:
        topic_name = cluster['topic_keywords'][0] if cluster['topic_keywords'] else f"Topic_{cluster['cluster_id']}"
        
        # Basic frequency
        frequency = len(cluster['questions']) / total_questions
        
        # Marks-weighted score
        total_marks = sum(q.get('marks', 0) for q in cluster['questions'])
        avg_marks = total_marks / len(cluster['questions']) if cluster['questions'] else 0
        weighted_score = frequency * avg_marks
        
        trends[topic_name] = {
            'frequency': frequency,
            'avg_marks': avg_marks,
            'weighted_score': weighted_score,
            'question_count': len(cluster['questions']),
            'total_marks': total_marks
        }
    
    # Sort by weighted score
    sorted_trends = dict(sorted(trends.items(), 
                               key=lambda x: x[1]['weighted_score'], 
                               reverse=True))
    
    return sorted_trends

=== TASK 3: Add Diagram Question Detection ===
PROMPT FOR CODEX:

Add diagram detection in question_extractor.py:

def detect_diagram_questions(self, question_text: str) -> bool:
    '''Detect if question involves diagrams/figures'''
    
    diagram_keywords = [
        'diagram', 'figure', 'graph', 'chart', 'plot',
        'draw', 'sketch', 'illustrate', 'show',
        'above figure', 'given diagram', 'refer to'
    ]
    
    text_lower = question_text.lower()
    return any(keyword in text_lower for keyword in diagram_keywords)

def extract_enhanced_questions(self, text: str, paper_id: str) -> List[Dict]:
    '''Enhanced question extraction with metadata'''
    
    basic_questions = self.extract_simple_questions(text, paper_id)
    
    for question in basic_questions:
        # Add diagram detection
        question['has_diagram'] = self.detect_diagram_questions(question['text'])
        
        # Add difficulty estimation (based on marks and keywords)
        question['difficulty'] = self.estimate_difficulty(question)
        
        # Add topic hints
        question['topic_hints'] = self.extract_topic_hints(question['text'])
    
    return basic_questions

=== TASK 4: Simple UI (Optional) ===
PROMPT FOR CODEX:

Create a simple Streamlit UI in ui/app.py:

import streamlit as st
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from main import process_papers_pipeline

def main():
    st.title("📚 Exam Paper Intelligence Engine")
    st.markdown("Upload exam papers and discover question patterns!")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Process files
        with st.spinner("Processing papers..."):
            results = process_papers_pipeline(uploaded_files)
        
        # Display results
        st.success(f"Processed {results['question_count']} questions!")
        
        # Show clusters
        st.subheader("🔍 Question Clusters")
        for cluster in results['clusters']:
            with st.expander(f"Cluster {cluster['cluster_id']} ({len(cluster['questions'])} questions)"):
                st.write("**Keywords:**", ", ".join(cluster['topic_keywords']))
                st.write("**Average Marks:**", cluster['avg_marks'])
                
                for q in cluster['questions'][:3]:  # Show first 3
                    st.write(f"- {q['paper']} Q{q['number']}: {q['text'][:100]}...")

if __name__ == "__main__":
    main()

=============== TESTING & VALIDATION ===============

=== Week 2 Testing ===
1. Test with 3-4 real exam papers
2. Verify embedding generation works
3. Tune clustering threshold for your domain
4. Compare ML results vs exact matching
5. Validate topic keyword extraction

=== Week 3 Testing ===
1. Test PDF generation with all features
2. Verify trend calculations are accurate
3. Test diagram detection on varied questions
4. Performance test with larger question sets
5. User acceptance testing

=== Performance Benchmarks ===
- Processing time: <30 seconds for 4 papers
- Memory usage: <2GB for 200 questions
- Accuracy: >80% for similar question detection
- PDF generation: <10 seconds
"""