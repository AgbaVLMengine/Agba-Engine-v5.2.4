import os
import json
import torch
import numpy as np
import unicodedata
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModel

def robust_strip_diacritics(text: str) -> str:
    """Decomposes Unicode characters and strips all non-spacing tone/dot marks."""
    if not text:
        return ""
    nfd_form = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in nfd_form if unicodedata.category(c) != "Mn")
    return stripped.lower().strip()

class AgbaProductionEngine:
    """Dual-Engine Architecture: Deterministic Text Search + SigLIP Vision Search with OOD Guardrail."""

    def __init__(self, corpus_path: str, matrix_path: str, model_name: str = "google/siglip-base-patch16-224"):
        print(f"⚡ Loading SigLIP Vision Backbone ({model_name})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(f"📥 Loading Cleaned Ground-Truth Corpus from: {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        print(f"⚡ Indexing Deterministic Text Tokens...")
        for item in self.corpus:
            item["_id_stripped"] = robust_strip_diacritics(item["id"])
            item["_class_stripped"] = robust_strip_diacritics(item["class"])
            item["_desc_stripped"] = robust_strip_diacritics(item.get("description", ""))
            item["_id_tokens"] = set(item["_id_stripped"].split())
            item["_class_tokens"] = set(item["_class_stripped"].split())
            item["_desc_tokens"] = set(item["_desc_stripped"].split())

        print(f"⚡ Loading Pre-Computed Static Vision Matrix from: {matrix_path}")
        self.text_embeddings = np.load(matrix_path)
        print(f"🚀 Engine Initialized! Total Entities: {len(self.corpus)} | Matrix Shape: {self.text_embeddings.shape}\n")

    # ---------------------------------------------------------
    # 1. DETERMINISTIC TEXT SEARCH ENGINE
    # ---------------------------------------------------------
    def search_by_text(self, text_query: str, top_k: int = 5):
        query_stripped = robust_strip_diacritics(text_query)
        query_tokens = set(query_stripped.split())
        
        if not query_tokens:
            return []

        scored_results = []
        for item in self.corpus:
            score = 0.0

            # Direct Exact ID Match
            if query_stripped == item["_id_stripped"]:
                score = 1.0000
            # ASCII Folded Token Overlap
            elif query_tokens.issubset(item["_id_tokens"]):
                score = 0.9600
            elif query_tokens.intersection(item["_id_tokens"]):
                score = 0.9000
            elif query_tokens.intersection(item["_class_tokens"]):
                score = 0.7500
            elif query_tokens.intersection(item["_desc_tokens"]):
                score = 0.5000

            if score > 0.0:
                scored_results.append({
                    "id": item["id"],
                    "class": item["class"],
                    "score": score,
                    "description": item.get("description", "")
                })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    # ---------------------------------------------------------
    # 2. NEURAL VISION SEARCH ENGINE (SigLIP + OOD Guardrail)
    # ---------------------------------------------------------
    def search_by_vision(self, image_input, top_k: int = 5, ood_threshold: float = 0.0600):
        headers = {"User-Agent": "Mozilla/5.0"}
        if isinstance(image_input, str):
            res = requests.get(image_input, headers=headers, timeout=15)
            res.raise_for_status()
            image = Image.open(BytesIO(res.content)).convert("RGB")
        else:
            image = image_input.convert("RGB")

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_outputs = self.model.get_image_features(**inputs)
            features = image_outputs.pooler_output if hasattr(image_outputs, "pooler_output") else image_outputs
            image_vec = (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()

        sims = np.dot(self.text_embeddings, image_vec.T).squeeze()
        top_indices = np.argsort(sims)[::-1][:top_k]
        top_score = float(sims[top_indices[0]])

        if top_score < ood_threshold:
            return {
                "status": "OOD_REJECTED",
                "message": f"Out-of-Distribution: Top similarity ({top_score:.4f}) below noise threshold ({ood_threshold}).",
                "results": []
            }

        results = []
        for rank_idx, idx in enumerate(top_indices, 1):
            item = self.corpus[idx]
            results.append({
                "rank": rank_idx,
                "id": item["id"],
                "class": item["class"],
                "raw_cosine": round(float(sims[idx]), 4)
            })

        return {
            "status": "MATCH_FOUND",
            "message": "Valid in-domain retrieval.",
            "results": results
        }
