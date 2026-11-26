# DependCAD

Official Python implementation for **DependCAD** (Text-to-CAD synthesis and evaluation with CadQuery/OpenCascade).

---

## Installation

Python 3.10–3.12 in an isolated virtual or Conda environment is recommended:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

python -m pip install -r requirements.txt
```

Configure LLM API keys via environment variables:

```powershell
# PowerShell
$env:QWEN_API_KEY="your_dashscope_api_key"
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"          # Optional
$env:SILICONFLOW_API_KEY="your_siliconflow_api_key"    # Optional
```

```bash
# Bash
export QWEN_API_KEY="your_dashscope_api_key"
export DEEPSEEK_API_KEY="your_deepseek_api_key"        # Optional
export SILICONFLOW_API_KEY="your_siliconflow_api_key"  # Optional
```

---

## External Data Format

Datasets and 3D meshes are not included in this repository. Prepare an external directory (e.g., `<DATASET_ROOT>/cad-coder/`) containing:

1. `case_library.jsonl`: Retrieval library JSONL file. You can build it from raw training records via:
   ```bash
   python -m experiments.cadquery_agent.dag_astar.build_case_library \
     --input path/to/raw_train_records.jsonl \
     --evaluation-manifest path/to/evaluation_ready.jsonl \
     --output path/to/case_library.jsonl
   ```
2. `evaluation_ready.jsonl`: Evaluation manifest JSONL (`case_id`, `model_id`, `length`, `length_bin`, `text`, `reference_stl`, `reference_hash`).
3. `reference_stl/<case_id>.stl`: Reference STL meshes for offline geometric evaluation.

---

## Usage

### Single-Task Synthesis

```bash
python run.py \
  --text "Create a rectangular mounting plate 60 by 40 by 8 mm, add a central cylindrical boss of radius 10 mm and height 12 mm, and cut a coaxial through-hole of radius 5 mm." \
  --task-id demo_mounting_plate \
  --case-library path/to/case_library.jsonl \
  --output-dir artifacts/demo
```

### Batch Evaluation & Ablations

```bash
python scripts/run_literal_defuse_ablation_formal.py \
  --dataset-root path/to/dataset_root \
  --output-root artifacts/dependcad_eval \
  --provider qwen \
  --model qwen3.7-flash \
  --dataset cad-coder \
  --prompt-top-k 3 \
  --retrieval-vector-weight 0.40 \
  --retrieval-operation-weight 0.20 \
  --retrieval-dependency-weight 0.15 \
  --retrieval-topology-weight 0.25 \
  --max-expansions 6 \
  --max-candidate-executions 5 \
  --max-llm-calls 2 \
  --search-seconds 180 \
  --seed 20261120 \
  --require-llm-requirement-graph
```

### Controlled Defect Evaluation

```bash
python scripts/eval_controlled_defect_recognition.py \
  --task-dirs artifacts/dependcad_eval/cad-coder/tasks \
  --out-json artifacts/controlled_defect_recognition.json
```

---

## License

This project is licensed under the terms in [`LICENSE`](LICENSE).
