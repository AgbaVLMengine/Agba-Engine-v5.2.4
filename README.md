# 🏛️ Agba Engine (`agba-engine`)

**Version:** 5.1.8  
**Author & Lead Architect:** Aruna Olanrewaju Kabiru  
**License:** `RAIL-Cultural-Heritage-v1.0`  

`agba-engine` is a multi-modal, diacritic-preserving retrieval framework engineered to index Global Cultural Heritage, Indigenous Scripts, and Tonal Orthographies without semantic stripping or unicode degradation.

---

## 🌍 The Mission: Preserving Global Heritage

Standard LLMs and vector search engines strip diacritics, flatten tonal markers, and misclassify rare indigenous vocabulary. **Agba Engine** provides an open, diacritic-proof retrieval architecture that preserves native orthographies in their full canonical purity.

Developers and cultural archivists globally can leverage `agba-engine` to register, index, and retrieve cultural assets across any language or script.

---

## 📦 Quickstart & Global Developer Workflow

### Installation
```bash
pip install agba-engine==5.1.8
```

### Registering Your Local Cultural Corpus
You can register and search any custom cultural dataset dynamically using `register_corpus()`:

```python
from agba_engine.core.engine import AgbaSearchEngine

# Initialize the engine
engine = AgbaSearchEngine()

# 1. Define your indigenous/cultural corpus entries
custom_heritage_data = [
    {
        "id": "Opón Ifá", 
        "class": "Divination Artefacts", 
        "description": "Sacred carved wooden tray used in Traditional Divination ceremonies."
    },
    {
        "id": "Ìroko", 
        "class": "Botany & Sacred Trees", 
        "description": "A large hardwood Tree from tropical Africa associated with Spiritual reverence."
    }
]

# 2. Register your corpus dynamically into the active search index
engine.register_corpus(custom_heritage_data)

# 3. Query using plain ASCII or diacritic-preserved text
results = engine.search_by_text("Opon Ifa", top_k=1)
print(f"Retrieved Entity: {results[0]['id']} ({results[0]['classification']})")
print(f"Description: {results[0]['description']}")
print(f"Confidence Score: {results[0]['score']}")
```

---

## 🛠️ API Reference

| Method | Parameters | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `register_corpus()` | `custom_list: List[Dict]` | `None` | Appends custom cultural JSON datasets to the active index. |
| `search_by_text()` | `query: str, top_k: int` | `List[Dict]` | Executes hybrid vector retrieval with ASCII-folding & diacritic parity. |
| `search_by_vision()` | `image_embedding, top_k: int` | `Dict` | Cross-modal visual search with Null Anchor thresholding. |
