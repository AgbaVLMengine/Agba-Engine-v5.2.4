# 🏛️ Agba Engine (`agba-engine`)

**Version:** 5.2.4  
**Author & Lead Architect:** Aruna Olanrewaju Kabiru  
**License:** `RAIL-Cultural-Heritage-v1.0`  

`agba-engine` is a cross-modal vision and diacritic-preserving text retrieval framework engineered to index Global Cultural Heritage, Indigenous Scripts, and Tonal Orthographies without semantic stripping, Unicode degradation, or sub-token drift.

---

## 🌍 The Mission: Preserving Global Heritage

Standard LLMs and vector search engines strip diacritics, flatten tonal markers, and misclassify rare indigenous vocabulary. **Agba Engine** provides an open, diacritic-proof dual-retrieval architecture that preserves native orthographies in their full canonical purity while integrating neural zero-shot vision search.

---

## 📦 Quickstart & Global Developer Workflow

### Installation
```bash
pip install agba-engine==5.2.4
```

### Initializing Engine & Executing Searches

```python
from agba_engine import AgbaProductionEngine

# Initialize dual engine (loads bundled ground-truth corpus & static SigLIP embeddings)
engine = AgbaProductionEngine()

# 1. Deterministic Text Retrieval (Diacritic & ASCII-Folded Parity)
text_results = engine.search_by_text("Opon Ifa", top_k=1)
print(f"Retrieved Entity: {text_results[0]['id']} | Score: {text_results[0]['score']}")

# 2. Cross-Modal Neural Vision Search (SigLIP + OOD Noise Guardrail)
vision_results = engine.search_by_vision("https://example.com/yewa_beads.jpg", top_k=3, ood_threshold=0.0600)
if vision_results["status"] == "MATCH_FOUND":
    for hit in vision_results["results"]:
        print(f"Rank {hit['rank']}: {hit['id']} ({hit['class']}) — Cosine: {hit['raw_cosine']}")
```

---

## 🛠️ API Reference

| Method | Parameters | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `search_by_text()` | `text_query: str`, `top_k: int = 5` | `List[Dict]` | Executes deterministic token retrieval with ASCII-folding (`0.9600`) & exact diacritic parity (`1.0000`). |
| `search_by_vision()` | `image_input: Union[str, Image.Image]`, `top_k: int = 5`, `ood_threshold: float = 0.0600` | `Dict` | Cross-modal SigLIP visual search against pre-computed static embeddings with OOD noise rejection. |

---

## 🏛️ Enterprise & Institutional API Access (Museums & Archives)

`agba-engine` provides specialized integration hooks for cultural institutions, digital heritage archives, and global museum databases seeking high-fidelity index retrieval for native orthographies, 3D artifacts, and historical vocalization assets.

* **Audio & Vocalization Pipeline:** Endpoints for indexing pitch-contour models, tonal pronunciation vectors, and spoken indigenous archives.
* **Museum Corpus Sync:** Direct ingestion pipelines for institutional digital asset management (DAM) platforms.

For API access keys, custom corpus integration support, or institutional partnerships, reach out directly or open a request on our Hugging Face Space:

* **GitHub Repository:** [Agba-Engine on GitHub](https://github.com/AgbaVLMengine/Agba-Engine-v5.2.4)
* **Hugging Face Space:** [Agba Engine Live Demo & Community](https://https://huggingface.co/spaces/OcculusPanther/Agba-engine-demo)
* **Direct Contact:** `only1mooseylion@gmail.com`
