# 📚 Exam Paper Intelligence & Trend Modeling Engine

A sophisticated ML system that analyzes exam papers to identify question patterns, detect duplicates, and model trending topics.

## 🎯 Project Overview

**Problem**: Exam setters need to understand question patterns across multiple papers to ensure balanced coverage and avoid repetition.

**Solution**: ML-powered system that:
- Extracts questions from PDF papers
- Detects similar questions using embeddings
- Groups questions by similarity 
- Analyzes trending topics with weighted importance
- Generates professional reports

## 🏗 System Architecture

```
Input PDFs → Text Extraction → Question Detection → ML Embeddings → Clustering → Trend Analysis → Report Generation
```

### Core Components
- **PDF Processor**: OCR + text extraction
- **Question Extractor**: Regex-based question detection
- **Embedding Engine**: SentenceTransformers for semantic similarity
- **Clustering Engine**: Agglomerative clustering with cosine similarity  
- **Report Generator**: Professional PDF output with trends

## 🚀 Quick Start

### Installation
```bash
cd exam_intelligence
pip install -r requirements.txt
```

### Basic Usage
```bash
# Week 1 MVP - Exact duplicate detection
python main.py --input_dir data/raw --mode extract

# Week 2+ - ML-based similarity  
python main.py --input_dir data/raw --mode full --tune_threshold
```

### Input Format
Place PDF files in `data/raw/`:
```
data/raw/
├── paper1.pdf
├── paper2.pdf
└── paper3.pdf
```

### Output
- Console results (Week 1)
- Professional PDF reports (Week 2+)
- JSON data export
- Trend analysis charts

## 📅 Development Timeline

### Week 1: MVP Foundation
**Goal**: Working system with exact duplicate detection
- ✅ PDF text extraction
- ✅ Regex question detection  
- ✅ Exact text matching
- ✅ Console output
- ✅ Basic frequency counting

**Deliverable**: Console app that finds identical questions

### Week 2: Add Intelligence  
**Goal**: ML-based similarity detection
- ✅ Sentence embeddings (all-MiniLM-L6-v2)
- ✅ Cosine similarity clustering
- ✅ Threshold tuning (0.75-0.85)
- ✅ Topic keyword extraction (TF-IDF)
- ✅ Cluster statistics

**Deliverable**: Intelligent question grouping

### Week 3: Professional Polish
**Goal**: Production-ready system
- ✅ Professional PDF reporting
- ✅ Weighted trend analysis  
- ✅ Diagram question detection
- ✅ Performance optimization
- ✅ Simple UI (optional)

**Deliverable**: Complete solution ready for real use

## 🔧 Technical Stack

### Core ML
- **sentence-transformers**: Semantic embeddings
- **scikit-learn**: Clustering algorithms
- **numpy/pandas**: Data processing

### PDF Processing  
- **pdfplumber**: Primary text extraction
- **PyPDF2**: Fallback extraction
- **pytesseract**: OCR for scanned documents

### Output Generation
- **reportlab**: Professional PDF reports
- **matplotlib/seaborn**: Trend visualization
- **streamlit**: Optional web UI

### Development
- **pytest**: Testing framework
- **logging**: Comprehensive logging
- **pathlib**: Modern path handling

## 📊 Technical Decisions

### Why This Architecture?
1. **Local ML**: No API dependencies, handles exam season spikes
2. **Incremental Development**: MVP first, add complexity gradually  
3. **Modular Design**: Each component can be improved independently
4. **Academic Rigor**: Explainable similarity metrics and thresholds

### Key ML Choices
- **Embeddings**: all-MiniLM-L6-v2 (384 dim, good performance/size balance)
- **Similarity**: Cosine similarity (standard for embeddings)
- **Clustering**: Agglomerative with distance threshold (more interpretable than DBSCAN)
- **Threshold**: 0.75-0.85 range (tunable based on domain)

## 📁 Project Structure

```
exam_intelligence/
├── main.py                     # Entry point
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
├── WEEK1_PROMPTS.md           # Week 1 implementation guide
├── WEEK2_3_PROMPTS.md         # Week 2-3 enhancement guide
├── data/
│   ├── raw/                   # Input PDFs
│   └── processed/             # Cached embeddings
├── src/
│   ├── preprocessing/
│   │   ├── pdf_processor.py   # PDF text extraction
│   │   └── question_extractor.py # Question detection
│   ├── ml_core/
│   │   ├── embedding_engine.py    # Sentence embeddings  
│   │   ├── clustering_engine.py   # Similarity clustering
│   │   └── simple_matcher.py      # Week 1 exact matching
│   └── output/
│       ├── report_generator.py    # PDF reports
│       └── console_output.py      # Console display
├── tests/                     # Test suite
├── notebooks/                 # Jupyter analysis
└── outputs/                   # Generated reports
```

## 🧪 Testing & Validation

### Week 1 Testing
```bash
# Test basic extraction
python -m pytest tests/test_extraction.py

# Manual validation
python main.py --input_dir test_data --mode extract
```

### Week 2+ Testing  
```bash
# Test ML pipeline
python -m pytest tests/test_ml_core.py

# Threshold tuning
python main.py --input_dir data/raw --tune_threshold

# Performance benchmarks  
python scripts/benchmark.py
```

### Validation Metrics
- **Precision**: % of grouped questions that are actually similar
- **Recall**: % of similar questions that are grouped together  
- **Efficiency**: Processing time per question
- **User Acceptance**: Manual review of clustering quality

## 🎓 Academic Rigor

This system demonstrates real ML engineering:

### Explainable Decisions
- Why cosine similarity? (Standard for embeddings, interpretable 0-1 range)
- Why threshold X? (Tuned on validation data with precision/recall curves)  
- How does OCR noise affect clustering? (Text normalization strategies)
- Why Agglomerative over DBSCAN? (More stable, interpretable distance threshold)

### Evaluation Strategy
- Manual labeling of 20 similar + 20 dissimilar question pairs
- Precision/recall on labeled data
- False positive (wrong merges) and false negative (missed similarities) analysis
- Ablation studies on preprocessing steps

### Scalability Considerations  
- Batch embedding generation
- Embedding caching for repeated use
- Memory-efficient similarity computation
- Incremental clustering for new papers

## 📈 Future Enhancements

### Advanced Features (Post-MVP)
- **Deep Question Understanding**: Fine-tuned BERT for exam domain
- **Multi-modal Analysis**: Handle questions with images/diagrams
- **Temporal Trends**: Track topic evolution across years
- **Difficulty Prediction**: ML model for question difficulty
- **Automated Question Generation**: Generate similar questions

### Integration Options
- **LMS Integration**: Connect with exam management systems
- **API Service**: REST API for institutional use  
- **Batch Processing**: Handle hundreds of papers
- **Real-time Analysis**: Live question similarity checking

## 🤝 Contributing

### Implementation Guide
1. **Week 1**: Use `WEEK1_PROMPTS.md` for MVP implementation
2. **Week 2-3**: Use `WEEK2_3_PROMPTS.md` for enhancements  
3. **Testing**: Run test suite before submitting
4. **Documentation**: Update README for new features

### Code Standards
- Type hints for public functions
- Comprehensive logging
- Error handling for all file operations
- Docstrings for complex algorithms

## 📄 License

MIT License - Free for educational and commercial use.

## 🎯 Success Metrics

### Technical Success
- ✅ Processes 4 papers in <30 seconds
- ✅ >80% accuracy on similar question detection  
- ✅ <2GB memory usage for 200 questions
- ✅ Professional PDF generation in <10 seconds

### User Success  
- ✅ Reduces manual pattern analysis from hours to minutes
- ✅ Discovers non-obvious question similarities
- ✅ Provides actionable insights for exam improvement
- ✅ Produces reports suitable for institutional review

### Academic Success
- ✅ Demonstrates ML engineering best practices
- ✅ Explainable similarity metrics and clustering decisions
- ✅ Rigorous evaluation methodology
- ✅ Scalable and maintainable architecture

---

**Ready to build intelligent exam analysis? Start with Week 1 MVP! 🚀**