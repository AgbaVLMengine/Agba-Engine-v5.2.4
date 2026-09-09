import gradio as gr
import json
import numpy as np
import unicodedata
from sentence_transformers import SentenceTransformer

# 1. Load Corpus and Embedding Model
with open("agba_engine/artifacts/index_metadata.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

embeddings_norm = np.load("agba_engine/artifacts/corpus_embeddings.npy")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn").lower().strip()

# Get Unique Classes for Category Dropdown Filter
categories = ["All"] + sorted(list({item["class"] for item in corpus}))

def search_cultural_heritage(query, category_filter, top_k):
    if not query.strip():
        blank_spotlight = '''
        <div style="background: #0f172a; border: 1px dashed #334155; padding: 18px; border-radius: 8px; text-align: center;">
            <span style="color: #64748b; font-size: 13px;">🌟 Cultural Spotlight: Execute a search query to surface the top-ranked heritage entity.</span>
        </div>
        '''
        return blank_spotlight, "<p style='color: #94a3b8; margin-top: 10px;'>Enter a query to explore the 1,007-entity corpus.</p>"
        
    clean_query = unicodedata.normalize("NFC", query).strip()
    query_lower = clean_query.lower()
    query_stripped = strip_diacritics(clean_query)
    
    # 1. Vector Encoding & Scoring
    query_vector = embedder.encode([clean_query])[0]
    query_vector = query_vector / np.linalg.norm(query_vector)
    scores = np.dot(embeddings_norm, query_vector)
    
    # 2. Dual Lexical Boost
    for idx, item in enumerate(corpus):
        item_id_norm = unicodedata.normalize("NFC", item["id"]).lower()
        item_id_stripped = strip_diacritics(item["id"])
        
        if query_lower == item_id_norm:
            scores[idx] += 3.0
        elif query_stripped == item_id_stripped:
            scores[idx] += 2.5
        elif query_stripped in item_id_stripped or item_id_stripped in query_stripped:
            scores[idx] += 0.5
            
    # 3. Rank and Filter Candidates
    ranked_indices = np.argsort(scores)[::-1]
    filtered_results = []
    
    for idx in ranked_indices:
        item = corpus[idx]
        if category_filter != "All" and item["class"] != category_filter:
            continue
        filtered_results.append((item, round(float(scores[idx]), 4)))
        if len(filtered_results) == top_k:
            break
            
    if not filtered_results:
        no_match_spotlight = '''
        <div style="background: #0f172a; border: 1px dashed #ef4444; padding: 18px; border-radius: 8px; text-align: center;">
            <span style="color: #f87171; font-size: 13px;">⚠️ No Cultural Spotlight match found for this category filter.</span>
        </div>
        '''
        return no_match_spotlight, "<p style='color: #94a3b8;'>No matching entities found.</p>"

    # 4. Generate Cultural Spotlight (Top Ranked #1 Entity)
    top_item, top_score = filtered_results[0]
    spotlight_html = f'''
    <div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); border: 2px solid #6366f1; padding: 20px; border-radius: 10px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="color: #818cf8; font-size: 12px; font-weight: bold; letter-spacing: 1.5px; text-transform: uppercase;">✨ CULTURAL SPOTLIGHT (TOP MATCH)</span>
            <span style="background-color: #4338ca; color: #e0e7ff; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">{top_item['class']}</span>
        </div>
        <h2 style="color: #ffffff; margin: 0 0 8px 0; font-size: 22px;">{top_item['id']}</h2>
        <p style="color: #c7d2fe; font-size: 14px; line-height: 1.5; margin: 0 0 12px 0;">{top_item['description']}</p>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="color: #a5b4fc; font-size: 11px; font-weight: 600;">Match Confidence Score:</span>
            <span style="background-color: #1e1b4b; color: #38bdf8; border: 1px solid #38bdf8; padding: 1px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;">{top_score}</span>
        </div>
    </div>
    '''

    # 5. Generate Remaining Candidate List (#2 through Top K)
    list_html = ""
    if len(filtered_results) > 1:
        list_html += "<h4 style='color: #94a3b8; margin: 20px 0 10px 0;'>RELATED HERITAGE CANDIDATES</h4>"
        for item, score in filtered_results[1:]:
            list_html += f'''
            <div style="background-color: #1e293b; border-left: 4px solid #6366f1; padding: 14px; margin-bottom: 10px; border-radius: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 15px; font-weight: bold; color: #38bdf8;">{item['id']}</span>
                    <span style="background-color: #312e81; color: #a5b4fc; padding: 2px 8px; border-radius: 4px; font-size: 11px;">{item['class']}</span>
                </div>
                <p style="color: #cbd5e1; font-size: 13px; margin: 6px 0 4px 0;">{item['description']}</p>
                <span style="color: #64748b; font-size: 10px;">Similarity Score: {score}</span>
            </div>
            '''
            
    return spotlight_html, list_html

# Interface Layout
with gr.Blocks(title="Agba Engine — Global Cultural Heritage Search") as demo:
    gr.Markdown("# 🏛️ Agba Engine v5.1.8")
    gr.Markdown("### Diacritic-Preserving Cultural Heritage Retrieval Framework")
    
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(label="Query Input", placeholder="Type a concept, entity name, or plain text (e.g. 'divination tray', 'Opon Ifa', 'Gbegiri')...")
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(choices=categories, value="All", label="Category Filter")
            top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top K Results")
            search_btn = gr.Button("Search Corpus", variant="primary")
            
    gr.Markdown("---")
    
    # Cultural Spotlight Display Widget (Top Match)
    spotlight_display = gr.HTML(label="Cultural Spotlight")
    
    # Remaining Results Display
    list_display = gr.HTML(label="Related Candidates")
    
    # Event Handlers
    search_btn.click(fn=search_cultural_heritage, inputs=[query_input, category_dropdown, top_k_slider], outputs=[spotlight_display, list_display])
    query_input.submit(fn=search_cultural_heritage, inputs=[query_input, category_dropdown, top_k_slider], outputs=[spotlight_display, list_display])

demo.launch()
