# Quick Start: Modifying Knowledge Graph Classifications

## 🎯 Goal
Modify knowledge graph node types and keyword mappings **without editing code**.

## 📝 How to Add New Classifications

### 1. **Add New Concept Type**

Edit `config/graph_ontology.yaml`:

```yaml
concept_types:
  # Add your new type here
  - name: "quantum_computing"
    display_name: "quantum computing"
    category: "emerging_tech"
```

### 2. **Add Keywords for Your Type**

```yaml
keyword_mappings:
  quantum_computing:
    - "quantum"
    - "qubit"
    - "quantum gate"
    - "quantum circuit"
    - "quantum algorithm"
```

### 3. **Reload Configuration** (Future - after GraphManager refactor)

```bash
python main.py ontology reload
```

## 📊 Current Available Categories

| Category | Purpose | Examples |
|----------|---------|----------|
| `ai_ml` | AI/ML concepts | agent, llm, neural_network |
| `research` | Research terms | paper, method, formal_method |
| `security` | Security/crypto | person, actor, cryptography |
| `network` | Networking | protocol, distributed_system |
| `architecture` | System design | component, framework, api |
| `tools` | Development tools | tool, algorithm |
| `data` | Data concepts | data, dataset, process |
| `organization` | Entities | organization, company |
| `metadata` | Meta information | temporal, measurement |
| `misc` | Other | metric, knowledge, other |

## 🔍 Examples of What You Can Do

### Example 1: Add Medical Domain Classifications

```yaml
concept_types:
  - name: "disease"
    display_name: "disease"
    category: "medical"
  
  - name: "treatment"
    display_name: "treatment"
    category: "medical"
  
  - name: "symptom"
    display_name: "symptom"
    category: "medical"

keyword_mappings:
  disease:
    - "cancer"
    - "diabetes"
    - "hypertension"
    - "disease"
    
  treatment:
    - "therapy"
    - "medication"
    - "treatment"
    - "drug"
    
  symptom:
    - "fever"
    - "pain"
    - "symptom"
```

### Example 2: Add Pattern-Based Classification

For things that follow a pattern (like dates, measurements):

```yaml
classification_patterns:
  # Match DOI identifiers
  - name: "document_type"
    pattern: '10\.\d{4,9}/[-._;()/:\w]+'
    description: "DOI pattern"
  
  # Match version numbers
  - name: "measurement"
    pattern: 'v?\d+\.\d+\.\d+'
    description: "Semantic version (e.g., v1.2.3)"
  
  # Match GitHub repos
  - name: "component"
    pattern: '[a-zA-Z0-9-]+/[a-zA-Z0-9-]+'
    description: "GitHub repo format (owner/repo)"
```

### Example 3: Add New Relationship Types

```yaml
relationship_types:
  - name: "treats"
    description: "Treatment relationship in medical domain"
  
  - name: "causes_symptom"
    description: "Disease causing symptom"

relationship_mappings:
  treats:
    - "treats"
    - "used to treat"
    - "treatment for"
```

## ✅ Validation

Test your changes:

```bash
python tests/test_ontology_loader.py
```

This will show:
- ✓ Configuration is valid
- ✓ Classification examples
- ✓ Performance metrics

## 🚀 Performance

**Current Stats:**
- 48 concept types
- 241 keyword mappings
- 9 pattern rules
- **0.0038ms** per classification (cached)

Adding more keywords has **negligible performance impact** due to caching.

## 📚 File Locations

| File | Purpose |
|------|---------|
| `config/graph_ontology.yaml` | **Main config - edit this** |
| `config/ontology_loader.py` | Loader code (don't edit) |
| `tests/test_ontology_loader.py` | Test your changes |

## ⚠️ Common Mistakes

1. **Duplicate concept names**: Each `name` must be unique
2. **Missing display_name**: Every concept needs a `display_name`
3. **Invalid regex**: Test pattern rules carefully
4. **Keyword conflicts**: More specific keywords should come first

## 🔧 Troubleshooting

### "Validation errors" message

Run validation to see specific issues:
```bash
python -c "from config.ontology_loader import get_ontology_loader; \
loader = get_ontology_loader(); \
valid, errors = loader.validate_config(); \
print('\n'.join(errors) if errors else 'Valid!')"
```

### Node still classified as "unknown"

1. Check keyword spelling in YAML
2. Ensure keyword is lowercase
3. Try adding more specific keywords
4. Consider using a pattern rule instead

### Pattern not matching

Test your regex in Python:
```python
import re
pattern = r'your_pattern_here'
test_string = "your test string"
print(re.search(pattern, test_string, re.IGNORECASE))
```

## 📖 Next Steps

After modifying `graph_ontology.yaml`:

1. ✅ Run validation: `python tests/test_ontology_loader.py`
2. ⏳ Refactor GraphManager to use loader (future)
3. ⏳ Add CLI commands for hot reload (future)
4. ⏳ Reclassify existing graph with new rules (future)

---

**Current Status**: Configuration system ready, integration pending
**Last Updated**: 2025-10-31
