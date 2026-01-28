import gradio as gr
from pathlib import Path
from PIL import Image
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
import torch
from datetime import datetime
import fitz  # PyMuPDF
import shutil

# Load models
print("Loading models...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llm_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading image caption model...")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

print("✅ All models loaded!")

# Storage
documents = []
images = []
image_captions = []
embeddings_index = None

def generate_image_caption(image_path):
    """Generate detailed caption for image"""
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Generate detailed caption
        inputs = caption_processor(img, return_tensors="pt").to(caption_model.device)
        output = caption_model.generate(
            **inputs, 
            max_length=100,
            num_beams=5,
            temperature=0.7
        )
        caption = caption_processor.decode(output[0], skip_special_tokens=True)
        
        return caption.strip()
    except Exception as e:
        print(f"Caption error: {e}")
        return ""

def extract_images_from_pdf(pdf_path):
    """Extract images from PDF"""
    extracted = []
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            images_list = page.get_images(full=True)
            
            for img_index, img in enumerate(images_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save image
                    img_path = f"/tmp/pdf_page{page_num+1}_img{img_index}.png"
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Validate image
                    test_img = Image.open(img_path)
                    width, height = test_img.size
                    
                    # Only keep meaningful images (not tiny icons/logos)
                    if width >= 150 and height >= 150:
                        extracted.append({
                            'path': img_path,
                            'page': page_num + 1,
                            'source': Path(pdf_path).name
                        })
                except Exception as e:
                    continue
        
        doc.close()
    except Exception as e:
        print(f"PDF image extraction error: {e}")
    
    return extracted

def extract_pdf_text(pdf_path):
    """Extract text from PDF"""
    chunks = []
    with open(pdf_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip():
                chunks.append({
                    'text': text,
                    'page': i + 1,
                    'source': Path(pdf_path).name
                })
    return chunks

def extract_docx_text(docx_path):
    doc = docx.Document(docx_path)
    text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    return [{'text': text, 'source': Path(docx_path).name}]

def extract_txt_text(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [{'text': f.read(), 'source': Path(txt_path).name}]

def chunk_text(text, size=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = ' '.join(words[i:i+size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def process_files(files, progress=gr.Progress()):
    """Process uploaded files"""
    global documents, images, image_captions, embeddings_index
    
    if not files:
        return "⚠️ Please upload files first"
    
    documents = []
    images = []
    image_captions = []
    
    total = len(files)
    
    for idx, file in enumerate(files):
        progress((idx + 1) / total, desc=f"Processing {Path(file.name).name}...")
        ext = Path(file.name).suffix.lower()
        
        if ext == '.pdf':
            # Extract text
            chunks = extract_pdf_text(file.name)
            for chunk in chunks:
                for small_chunk in chunk_text(chunk['text']):
                    documents.append({
                        'text': small_chunk,
                        'source': chunk['source'],
                        'page': chunk['page']
                    })
            
            # Extract images
            pdf_images = extract_images_from_pdf(file.name)
            for img in pdf_images:
                caption = generate_image_caption(img['path'])
                if caption:  # Only add if caption generated
                    images.append(img)
                    image_captions.append(caption)
        
        elif ext == '.docx':
            chunks = extract_docx_text(file.name)
            for chunk in chunks:
                for small_chunk in chunk_text(chunk['text']):
                    documents.append({
                        'text': small_chunk,
                        'source': chunk['source']
                    })
        
        elif ext == '.txt':
            chunks = extract_txt_text(file.name)
            for chunk in chunks:
                for small_chunk in chunk_text(chunk['text']):
                    documents.append({
                        'text': small_chunk,
                        'source': chunk['source']
                    })
        
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            caption = generate_image_caption(file.name)
            if caption:
                images.append({
                    'path': file.name,
                    'source': Path(file.name).name,
                    'page': ''
                })
                image_captions.append(caption)
    
    # Create embeddings
    progress(0.9, desc="Creating embeddings...")
    if documents:
        texts = [doc['text'] for doc in documents]
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        embeddings_index = index
    
    progress(1.0, desc="Done!")
    
    status = f"✅ **Processing Complete!**\n\n"
    status += f"📄 **Text chunks:** {len(documents)}\n"
    status += f"🖼️ **Images extracted:** {len(images)}\n"
    
    if images:
        status += f"\n**Sample captions:**\n"
        for i, (img, cap) in enumerate(zip(images[:3], image_captions[:3]), 1):
            status += f"{i}. {img['source']}"
            if img.get('page'):
                status += f" (Page {img['page']})"
            status += f":\n   _{cap}_\n"
    
    return status

def search_documents(query, k=3):
    """Search relevant documents"""
    if not documents or embeddings_index is None:
        return []
    
    query_vec = embedding_model.encode([query])
    distances, indices = embeddings_index.search(query_vec.astype('float32'), k)
    
    results = []
    for idx in indices[0]:
        if idx < len(documents):
            results.append(documents[idx])
    return results

def find_relevant_images(query, relevance_threshold=0.25):
    """Find images ONLY if relevant to query"""
    if not images or not image_captions:
        return [], []
    
    # Encode query and captions
    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    caption_embs = embedding_model.encode(image_captions, convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarities = util.cos_sim(query_emb, caption_embs)[0]
    
    # Filter by threshold and get top 3
    relevant_imgs = []
    explanations = []
    
    for idx, sim_score in enumerate(similarities):
        sim_value = float(sim_score)
        
        # Only show if relevance > threshold
        if sim_value > relevance_threshold:
            img_info = images[idx]
            caption = image_captions[idx]
            
            relevant_imgs.append(img_info['path'])
            
            # Create explanation
            exp = f"**📄 Source:** {img_info['source']}"
            if img_info.get('page'):
                exp += f" (Page {img_info['page']})"
            exp += f"\n**💬 Description:** {caption}"
            exp += f"\n**🎯 Relevance:** {sim_value * 100:.1f}%\n"
            
            explanations.append(exp)
    
    # Sort by relevance and take top 3
    if relevant_imgs:
        sorted_pairs = sorted(
            zip(similarities, relevant_imgs, explanations),
            key=lambda x: x[0],
            reverse=True
        )[:3]
        
        relevant_imgs = [pair[1] for pair in sorted_pairs]
        explanations = [pair[2] for pair in sorted_pairs]
    
    return relevant_imgs, explanations

def generate_answer(question, context_docs):
    """Generate answer from context"""
    context = '\n\n'.join([doc['text'] for doc in context_docs])
    
    prompt = f"""Answer this question based only on the context provided. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1200)
    
    with torch.no_grad():
        outputs = llm_model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer part
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    
    return answer

def answer_query(question, progress=gr.Progress()):
    """Answer question with relevant images only"""
    if not question.strip():
        return "⚠️ Please enter a question", None
    
    if not documents:
        return "⚠️ Please upload and process documents first", None
    
    # Search documents
    progress(0.3, desc="Searching documents...")
    relevant_docs = search_documents(question, k=3)
    
    if not relevant_docs:
        return "❌ No relevant information found", None
    
    # Generate answer
    progress(0.6, desc="Generating answer...")
    answer = generate_answer(question, relevant_docs)
    
    # Format response
    response = f"## 💡 Answer\n\n{answer}\n\n"
    response += f"## 📚 Text Sources\n\n"
    
    for i, doc in enumerate(relevant_docs, 1):
        source = doc['source']
        page = doc.get('page', '')
        if page:
            response += f"{i}. **{source}** (Page {page})\n"
        else:
            response += f"{i}. **{source}**\n"
    
    # Find relevant images
    progress(0.9, desc="Finding relevant images...")
    relevant_imgs, img_explanations = find_relevant_images(question, relevance_threshold=0.25)
    
    # Add image explanations if found
    if relevant_imgs and img_explanations:
        response += f"\n## 🖼️ Related Images\n\n"
        for exp in img_explanations:
            response += f"{exp}\n"
    else:
        response += f"\n_No relevant images found for this query_\n"
    
    progress(1.0, desc="Done!")
    
    return response, relevant_imgs if relevant_imgs else None

# UI
with gr.Blocks(
    title="DocVision AI",
    theme=gr.themes.Soft(primary_hue="indigo")
) as app:
    
    gr.Markdown("""
    # 📚 DocVision AI - Intelligent Document Q&A
    ### Upload documents and get AI-powered answers with relevant images
    """)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="📁 Upload Documents & Images",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt", ".jpg", ".png"]
            )
            process_btn = gr.Button(
                "⚡ Process Documents",
                variant="primary",
                size="lg"
            )
            status = gr.Markdown(label="📊 Processing Status")
        
        with gr.Column():
            question = gr.Textbox(
                label="❓ Ask Your Question",
                placeholder="What would you like to know about your documents?",
                lines=3
            )
            ask_btn = gr.Button(
                "🔍 Get Answer",
                variant="primary",
                size="lg"
            )
    
    answer = gr.Markdown(label="📝 Answer with Sources")
    
    gallery = gr.Gallery(
        label="🖼️ Relevant Images (Only shown if related to your question)",
        columns=2,
        height=500,
        show_label=True
    )
    
    gr.Markdown("### 💡 Example Questions")
    gr.Examples(
        examples=[
            ["What is the main topic of this document?"],
            ["Explain the workflow or architecture shown"],
            ["What are the key findings?"],
            ["Describe any diagrams or charts present"]
        ],
        inputs=question
    )
    
    # Event handlers
    process_btn.click(
        process_files,
        inputs=[file_input],
        outputs=[status]
    )
    
    ask_btn.click(
        answer_query,
        inputs=[question],
        outputs=[answer, gallery]
    )
    
    question.submit(
        answer_query,
        inputs=[question],
        outputs=[answer, gallery]
    )

if __name__ == "__main__":
    app.launch()
