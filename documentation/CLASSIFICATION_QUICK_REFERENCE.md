# Knowledge Graph Node Classification - Quick Reference

## 📊 Current Status

**Classification Rate:** 45.4% (6,798 / 14,978 nodes)  
**Unknown Nodes:** 54.6% (8,180 nodes)  
**Total Node Types:** 44 types

---

## 🎯 Node Type Categories

### Research & Documents (7.0%)
- `document_type` - Papers, surveys, reports, articles
- `research` - Research studies, investigations
- `method` - Analysis, evaluation, experiments

### Security & Cryptography (6.5%)
- `person` - Alice, Bob, Charlie (protocol actors)
- `actor` - Attacker, adversary, intruder
- `cryptography` - Keys, ciphers, signatures
- `security` - Security concepts

### AI & ML (5.9%)
- `agent` - AI agents, autonomous entities
- `models` - Transformers, neural networks
- `llm` - Large language models
- `framework` - LangGraph, AutoGen, CrewAI

### Network & Protocols (10.3%)
- `protocol` - TCP, HTTP, MCP, A2A
- `communication` - Messaging, signaling
- `network` - P2P, DHT concepts
- `distributed_system` - Consensus, replication

### Components & Roles (5.6%)
- `role` - Client, server, sender, receiver
- `component` - Nodes, adapters, proxies
- `system` - Platforms, environments

### Methods & Analysis (3.6%)
- `formal_method` - Proofs, theorems, verification
- `tool` - CPSA, UPPAAL, compilers
- `algorithm` - Heuristics, optimization

### Data & Resources (3.8%)
- `data` - Files, metadata, streams
- `message` - Requests, responses, events
- `process` - Tasks, jobs, workflows

### Other (2.7%)
- `measurement` - Numeric values with units
- `temporal` - Years, dates
- `organization` - Universities, institutes
- `company` - Google, OpenAI, Microsoft
- `metric` - Performance, efficiency

---

## 🔧 CLI Commands

### View Statistics
```bash
graph stats
```

### Reclassify Nodes (after code updates)
```bash
graph reclassify
```

### Query by Type
```bash
graph query "protocols"
graph query "agents"
```

---

## 📝 Adding New Keywords

### Step 1: Add to ConceptType Enum
```python
# In core/graph_manager.py
class ConceptType(Enum):
    YOUR_TYPE = "your_type"
```

### Step 2: Add Keywords
```python
# In CONCEPT_KEYWORDS dictionary
CONCEPT_KEYWORDS = {
    'your_keyword': ConceptType.YOUR_TYPE,
    'another_keyword': ConceptType.YOUR_TYPE,
}
```

### Step 3: Test
```python
from core.graph_manager import GraphManager
gm = GraphManager()
result = gm._classify_node("your test node")
print(result)  # Should return 'your_type' if matched
```

---

## 🎯 Pattern Matching Examples

| Pattern | Example | Type |
|---------|---------|------|
| Measurements | "2.5 km", "100 ms" | `measurement` |
| Years | "2024", "2025" | `temporal` |
| Dates | "November 2024" | `temporal` |
| Citations | "Smith et al." | `person` |
| Initials | "A. Johnson" | `person` |
| Organizations | "MIT University" | `organization` |
| Companies | "OpenAI Inc." | `organization` |
| File Paths | "config.json" | `component` |
| Single Letters | "A", "X" | `other` |

---

## 🔍 Most Common Classified Types

1. **protocol** (1,061) - 7.1%
2. **agent** (720) - 4.8%
3. **data** (362) - 2.4%
4. **role** (356) - 2.4%
5. **component** (349) - 2.3%
6. **cryptography** (340) - 2.3%
7. **system** (262) - 1.7%
8. **message** (258) - 1.7%
9. **person** (231) - 1.5%
10. **measurement** (208) - 1.4%

---

## 🚨 Troubleshooting

### "Node not being classified correctly"
1. Check if keyword exists in CONCEPT_KEYWORDS
2. Verify node label matches pattern exactly
3. Test with `_classify_node()` method directly

### "Reclassify command shows 0 changes"
- This is normal! The graph auto-applies new logic on load
- Reclassify is mainly for debugging/reporting

### "Too many unknown nodes"
- Expected for complex domain-specific terms
- Consider Phase 4 (Hybrid LLM-based classification)
- Or add more domain-specific keywords

---

## 📚 See Also

- [DEVELOPMENT_HISTORY.md - Section 17](DEVELOPMENT_HISTORY.md#17-knowledge-graph-classification-enhancement) - Full classification enhancement journey (Phases 1-4)
- [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md) - Complete project history
- [USER_GUIDE.md](USER_GUIDE.md) - User documentation
