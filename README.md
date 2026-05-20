# 🧠 OpenCode DA-Orchestrator

**Data Analyst Orchestrator for [OpenCode](https://opencode.ai)** — a complete AI-powered data analysis pipeline that delegates work across 18 specialized sub-agents.

Stop running pandas inline. Let the orchestrator coordinate, delegate, and synthesize.

## What It Does

The DA-Orchestrator implements the **DADD pipeline** (Data Analysis Driven Development):

```
[DA-01: EDA] → [DA-02: Cleaning] → [DA-03: Features] → [DA-04: Modeling] → [DA-05: Evaluation] → [DA-06: Interpretation]
```

Each phase is a **specialized sub-agent** with its own prompt, protocol, and notebook generation requirements. The orchestrator delegates — it never loads 100K-row CSVs into its own context.

### Pipeline Phases

| Phase | Agent | What It Does |
|-------|-------|-------------|
| DA-01 | `da-eda` | Exploratory Data Analysis — distributions, nulls, correlations, quality report |
| DA-02 | `da-cleaning` | Data Cleaning — imputation, outliers, encoding, scaling, transformation log |
| DA-03 | `da-features` | Feature Engineering — creation, selection, VIF, PCA, train/test split |
| DA-04 | `da-modeling` | ML Modeling — baseline, multi-model comparison, GridSearch, `.pkl` export |
| DA-05 | `da-evaluation` | Model Evaluation — metrics, overfitting diagnosis, confusion matrix, bootstrap CI |
| DA-06 | `da-interpreter` | Model Interpretation — SHAP, LIME, business insights, counterfactuals |

Every phase generates an executable `.ipynb` notebook as a persistent artifact.

### Model Tiering

Each phase comes in **3 tiers** for cost optimization across OpenCode profiles:

| Tier | Suffix | Use Case |
|------|--------|----------|
| Default | _(none)_ | Standard model for the active profile |
| Cheap | `-cheap` | Budget/free models (Curaduria profile) |
| Fallback | `-fallback` | Secondary model if primary fails |

That's **18 sub-agents** total — 6 phases × 3 tiers.

## Installation

### Quick Install

```bash
# Clone the repo
git clone https://github.com/IannPy/opencode-da.git
cd opencode-da

# Windows
.\install.ps1

# Unix / macOS
chmod +x install.sh
./install.sh
```

The installer:
1. Copies DA skills to `~/.config/opencode/skills/`
2. Copies DA prompts to `~/.config/opencode/prompts/da/`
3. Merges the 18 DA agents + orchestrator into your `opencode.json`

Restart OpenCode and you're ready.

### Manual Install

1. Copy `skills/da-*/` into your OpenCode skills directory
2. Copy `prompts/da/*.md` into your OpenCode prompts directory
3. Merge the `agent` block from `opencode.partial.json` into your `opencode.json`
4. Replace `__OP_ENCODE_HOME__` with your actual config path (e.g., `C:\Users\You\.config\opencode` on Windows, `/home/you/.config/opencode` on Unix)

## Usage

### Trigger the Pipeline

Just drop a CSV and ask:

> "Analizame este dataset de ventas, quiero predecir qué clientes van a cancelar"

Or use slash commands:

| Command | What It Does |
|---------|-------------|
| `/da-start` | Full pipeline from EDA to interpretation |
| `/da-eda` | Exploratory analysis only |
| `/da-clean` | Data cleaning only |
| `/da-features` | Feature engineering only |
| `/da-model` | Model training only |
| `/da-eval` | Metrics evaluation only |
| `/da-interpret` | SHAP/LIME interpretation only |
| `/da-report` | Consolidated executive report |
| `/da-status` | Current pipeline status |

### Phase 0 — Always First

The orchestrator will always establish context before analysis:

```
OBJETIVO:     What to predict/understand/classify?
MÉTRICA:      How do we measure success?
TIPO:         Regression | Classification | Clustering | Descriptive | Time Series
AUDIENCIA:    Technical | Business | Mixed
```

### What You Get

After a full pipeline run, you'll have:

```
📁 your-project/
├── da-01-eda.ipynb           # EDA notebook with all findings
├── da-02-cleaning.ipynb      # Cleaning transformations recorded
├── da-03-features.ipynb      # Feature engineering and selection
├── da-04-modeling.ipynb      # Model comparison and tuning
├── da-05-evaluation.ipynb    # Metrics, diagnostics, confidence intervals
├── da-06-interpretation.ipynb # SHAP, business insights, counterfactuals
├── df_clean.csv              # Cleaned dataset
├── modelo_final.pkl          # Trained model for production
├── scaler.pkl                # Scaler for inference
└── selected_features.json    # Feature list for inference pipeline
```

## Architecture

```
da-orchestrator (primary agent)
├── Delegation rules — delegates all 6 DA phases
├── Permission.task — deny *, allow {18 sub-agents}
├── Artifact store: engram
└── Pipeline commands: /da-start, /da-eda, ..., /da-status

     │ delegates to...
     ▼
┌──────────────┬──────────────┬──────────────┐
│  da-eda      │  da-cleaning │  da-features  │  ... × 6 phases
│  da-eda-cheap│  ...-cheap   │  ...-cheap    │  ... × 3 tiers
│  da-eda-fb   │  ...-fb      │  ...-fb       │  = 18 agents
└──────────────┴──────────────┴──────────────┘
```

Each sub-agent:
- Is `hidden: true` (never appears in autocomplete)
- Has `bash, edit, read, write` tools
- Reads its prompt from `prompts/da/da-[phase].md`
- Returns structured results to the orchestrator

## Requirements

- [OpenCode](https://opencode.ai) installed
- Python 3.9+ with: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap`
- Optional: `jq` (for Unix install script)
- Optional: [Engram](https://github.com/gentleman-programming/engram) for cross-session memory (falls back to inline if unavailable)

## Customizing Models

Edit the `model` field in your `opencode.json` for any DA agent:

```json
"da-modeling": {
  "model": "opencode-go/qwen3.6-plus"  // Upgrade for complex datasets
}
```

Or set per-profile models using OpenCode profiles.

## Contributing

Found a bug? Want to improve a phase? PRs welcome.

Each phase's prompt is in `prompts/da/da-[phase].md` — these are the sub-agent instructions. The source of truth for each phase's protocol lives in `skills/da-[phase]/SKILL.md`.

## License

MIT — use it, modify it, ship it.
