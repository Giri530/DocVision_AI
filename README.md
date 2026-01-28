# 📚 DocVision AI - Multimodal RAG Document Q&A System

**🚀 An intelligent document Q&A system powered by Retrieval-Augmented Generation (RAG) and Multimodal AI**

## 🌟 Overview

**DocVision AI** is a production-ready AI application that enables users to upload documents (PDF, DOCX, TXT) and images, then ask natural language questions to receive accurate, context-aware answers with **relevant visual evidence**.

### 💡 Problem It Solves

- ❌ Manual document reading is time-consuming
- ❌ Finding specific information in large PDFs is tedious  
- ❌ Images in documents often contain crucial information
- ❌ Traditional search doesn't understand context

### ✅ Solution

- ✨ AI-powered instant answers from your documents
- ✨ Smart image extraction and relevance matching
- ✨ Context-aware responses with source citations
- ✨ Zero setup - works in browser, completely free

---

## ✨ Features

### 🤖 **Core Capabilities**

| Feature | Description |
|---------|-------------|
| **📄 Multi-Format Support** | Process PDF, DOCX, TXT documents seamlessly |
| **🖼️ Image Intelligence** | Extract images from PDFs with AI-generated captions |
| **🎯 Smart Relevance** | Only shows images relevant to query (>25% threshold) |
| **🔍 Semantic Search** | FAISS vector database for lightning-fast retrieval |
| **💬 Natural Language Q&A** | Ask questions in plain English, get accurate answers |
| **📊 Source Attribution** | Transparent citations with page numbers |
| **⚡ Real-time Processing** | Progress tracking for all operations |
| **☁️ Cloud-Ready** | Deployed on Hugging Face Spaces |
| **🆓 Zero Cost** | Uses open-source models, no API keys needed |

### 🎯 **Key Differentiators**

- **Multimodal Understanding**: Combines text and vision AI
- **Intelligent Image Filtering**: Only relevant images shown
- **Production-Grade**: Error handling, progress bars, clean UI
- **Fully Open Source**: No vendor lock-in

---

> **🌐 Try it live:** [DocVision AI on Hugging Face Spaces](https://huggingface.co/spaces/Girinath11/DocVision-AI)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Gradio)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              DOCUMENT PROCESSING LAYER                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   PDF    │  │   DOCX   │  │   TXT    │  │  Images  │   │
│  │ PyMuPDF  │  │python-doc│  │  Parser  │  │ PIL/Pillow│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              EMBEDDING & VECTORIZATION                       │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Sentence-Transformers (all-MiniLM-L6-v2)          │     │
│  │  384-dimensional dense vectors                     │     │
│  └────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────┐     │
│  │  BLIP-Large (Vision Model)                         │     │
│  │  Image → Text Caption Generation                   │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                VECTOR DATABASE (FAISS)                       │
│  • IndexFlatL2 for L2 distance calculation                   │
│  • Fast approximate nearest neighbor search                  │
│  • O(log n) query time complexity                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  RETRIEVAL & MATCHING                        │
│  • Semantic search for text (top-3 chunks)                   │
│  • Cosine similarity for image-query matching                │
│  • Relevance threshold filtering (>25%)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ANSWER GENERATION (LLM)                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │  TinyLlama-1.1B-Chat                               │     │
│  │  Context-aware response generation                 │     │
│  │  Temperature: 0.7, Max tokens: 200                 │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                         OUTPUT                               │
│  • Answer with source citations & page numbers               │
│  • Relevant images with captions & relevance scores          │
│  • Metadata & statistics                                     │
└─────────────────────────────────────────────────────────────┘
```

### 🔄 RAG Pipeline Flow

1. **Document Ingestion** → Extract text & images
2. **Chunking** → Split text into 400-word segments
3. **Embedding** → Convert to 384-dim vectors
4. **Indexing** → Store in FAISS vector DB
5. **Query** → User asks question
6. **Retrieval** → Find top-3 relevant chunks
7. **Image Matching** → Match query to image captions
8. **Generation** → LLM creates contextual answer
9. **Response** → Answer + sources + relevant images

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Frontend** | Gradio 4.x | Interactive web UI with real-time updates |
| **Embeddings** | Sentence-Transformers | Text to 384-dim vector conversion |
| **Vector DB** | FAISS (Facebook AI) | Ultra-fast similarity search |
| **LLM** | TinyLlama-1.1B-Chat | Lightweight answer generation |
| **Vision AI** | BLIP-Large | Image captioning & understanding |
| **Deep Learning** | PyTorch 2.x | Model inference backend |
| **PDF Processing** | PyMuPDF (fitz) | Extract text & images from PDFs |
| **Doc Processing** | python-docx, PyPDF2 | Parse Word & text documents |
| **Image Processing** | Pillow (PIL) | Image manipulation & validation |
| **Deployment** | Hugging Face Spaces | Serverless cloud hosting |

### 📊 Model Specifications

| Model | Size | Task | Speed |
|-------|------|------|-------|
| **all-MiniLM-L6-v2** | 80MB | Text Embeddings | ~1000 sent/sec |
| **TinyLlama-1.1B** | 2.2GB | Text Generation | 2-5 sec/query |
| **BLIP-Large** | 1.8GB | Image Captioning | 1-2 sec/image |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- 8GB+ RAM
- GPU optional (works on CPU)

### 📦 Installation

```bash
# Clone repository
git clone https://github.com/Giri530/DocVision_AI.git
cd docvision-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### ▶️ Run Locally

```bash
python app.py
```

Access at `http://localhost:7860` 🎉

---

## 📖 Usage Guide

### Step 1: 📤 Upload Documents

Click **"Upload Documents & Images"** and select:
- 📄 **Documents**: PDF, DOCX, TXT
- 🖼️ **Images**: JPG, PNG, GIF

### Step 2: ⚡ Process

Click **"Process Documents"** and wait for:
- Text extraction from documents
- Image extraction from PDFs (min 150×150px)
- AI caption generation for each image
- Vector embedding creation
- FAISS index building

**Status shows:** Text chunks, Images found, Sample captions

### Step 3: 💬 Ask Questions

Type your question and click **"Get Answer"** to receive:
- 💡 **AI-generated answer** (context-aware)
- 📚 **Source citations** (with page numbers)
- 🖼️ **Relevant images** (if relevance >25%)
- 🎯 **Relevance scores** for each image

### 📝 Example Questions

```
✅ "What is the main topic of this document?"
✅ "Summarize the key findings and conclusions"
✅ "What statistics or numbers are mentioned?"
✅ "Explain the workflow diagram shown"
✅ "Describe the architecture in the images"
✅ "What are the recommendations provided?"
```

---

## 🎯 Use Cases

| Industry | Use Case |
|----------|----------|
| **Education** | Research paper analysis, study material Q&A |
| **Legal** | Contract review, case document analysis |
| **Healthcare** | Medical report interpretation, research papers |
| **Business** | Report analysis, meeting minutes extraction |
| **Engineering** | Technical documentation, diagram analysis |
| **Research** | Literature review, data extraction |

---

## 📊 Performance Benchmarks

| Metric | Value | Description |
|--------|-------|-------------|
| **Embedding Speed** | ~1000 sent/sec | Text vectorization |
| **Search Latency** | <100ms | FAISS similarity search |
| **Caption Generation** | 1-2 sec/image | BLIP-Large inference |
| **Answer Generation** | 2-5 seconds | LLM response time |
| **Total Query Time** | 5-10 seconds | End-to-end latency |
| **Max Document Size** | 50MB | Per PDF file |
| **Concurrent Users** | 10+ | HF Spaces free tier |

---

## 📁 Project Structure

```
docvision-ai/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── .python-version        # Python 3.11
├── README.md              # This file
├── LICENSE                # MIT License
├── .gitignore             # Git ignore rules
```

---

## 🧪 Technical Deep Dive

### Text Processing Pipeline

```python
# 1. Extract text
text = extract_from_pdf(file)

# 2. Chunk text (400 words)
chunks = chunk_text(text, size=400)

# 3. Generate embeddings
embeddings = model.encode(chunks)  # 384-dim vectors

# 4. Index with FAISS
index = faiss.IndexFlatL2(384)
index.add(embeddings)
```

### Image Processing Pipeline

```python
# 1. Extract images from PDF
images = pdf.get_images()

# 2. Filter by size
valid_images = [img for img in images if img.size >= (150,150)]

# 3. Generate captions
caption = blip_model.generate(image)

# 4. Match to query
similarity = cosine_sim(query_emb, caption_emb)
if similarity > 0.25:
    show_image(image, caption, similarity)
```

### RAG Implementation

```python
# Retrieval
relevant_docs = search(query, k=3)

# Augmentation
context = "\n\n".join([doc.text for doc in relevant_docs])

# Generation
answer = llm.generate(
    prompt=f"Context: {context}\nQuestion: {query}\nAnswer:",
    max_tokens=200
)
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### 💡 Ideas for Contribution

- [ ] Add support for PPTX, XLSX files
- [ ] Implement chat history/memory
- [ ] Add multilingual support
- [ ] Create REST API endpoints
- [ ] Add export to PDF functionality
- [ ] Implement OCR for scanned documents
- [ ] Add user authentication
- [ ] Create mobile app version

---

## 🐛 Known Issues & Limitations

| Issue | Impact | Workaround |
|-------|--------|------------|
| Large PDFs (>50MB) | Slow processing | Split into smaller files |
| Scanned PDFs | No text extraction | Use OCR preprocessing |
| GPU memory | Limited on free tier | Uses CPU automatically |
| Caption quality | Varies by image | Using BLIP-Large helps |
| Relevance threshold | May miss some images | Adjust threshold in code |

---

## 📈 Roadmap

### ✅ Version 1.0 (Current)
- [x] Multi-format document support
- [x] Image extraction & captioning
- [x] Semantic search with FAISS
- [x] RAG-based Q&A
- [x] HF Spaces deployment

### 🚧 Version 1.1 (In Progress)
- [ ] OCR for scanned documents
- [ ] Chat history
- [ ] Better error handling

### 🔮 Version 2.0 (Future)
- [ ] Multi-language support
- [ ] API endpoints
- [ ] User authentication
- [ ] Advanced analytics

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License - Copyright (c) 2026 Girinath
```

---

## 🙏 Acknowledgments

- **Hugging Face** - For amazing model hub and Spaces
- **Sentence-Transformers** - For embedding models
- **Facebook AI** - For FAISS library
- **TinyLlama Team** - For lightweight LLM
- **Salesforce Research** - For BLIP models
- **Gradio Team** - For the awesome UI framework
- **PyTorch Community** - For deep learning tools

---

## 📧 Contact & Support

**Girinath**  
- 🌐 **Live Demo**: [DocVision AI](https://huggingface.co/spaces/Girinath11/DocVision-AI)
- 💻 **GitHub**: [@Girinath11](https://github.com/Giri530)
- 🤗 **Hugging Face**: [@Girinath11](https://huggingface.co/Girinath11)
- ⭐ **Star the repo** if you find it useful!

---

**Made with ❤️ by Girinath | Powered by AI**
