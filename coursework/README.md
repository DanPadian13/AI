# Cognitive AI Coursework Workspace

Reference for all agents working in this repository.

## Top-Level Items

- `2025_Cognitive_AI_Coursework.txt` — canonical question set; read this first when interpreting requirements.
- `2025_Cognitive_AI_Coursework.pdf` — same brief with extra background; never really needed
- `Answer.docx` — current draft containing coursework responses; update this to produce the final deliverable.
- `code/` — dedicated space for all scripts/notebooks/models supporting the answers; create any new implementation artifacts here.
- `context/` — supporting materials (labs, lecture notes, prior reviews) for reference only.

## Context Directory

This directory is read-only reference for agents; consult it to understand the official material, style, and expectations before producing new work.

- macOS tools are available locally for extracting text when needed: e.g. run `textutil -convert txt <pdf> -stdout` to skim lecture PDFs without installing extra packages.
- Jupyter notebooks in `course_code/` are easiest to inspect by loading the JSON and printing markdown cells (`python3 - <<'PY' ...`); keep any exploratory scripts in `code/`.
- External package installs are blocked (network restricted), so rely on the system Python plus any preinstalled libraries; note this when planning NeuroGym experiments.

### `context/course_code/`

Self-contained Jupyter notebooks with solutions for each lab:

1. `CogAI_lab1_NNs_from_scratch_with_solutions.ipynb`
2. `Cognitive_AI_lab2_Supervised_Learning_backprop_with_solutions.ipynb`
3. `Cognitive_AI_lab3_Supervised_Learning_RNNs_with_solutions.ipynb`
4. `Cognitive_AI_lab4_brain_inspired_RNNs_with_solutions.ipynb`
5. `Cognitive_AI_lab5_RL_with_solutions.ipynb`

### `context/course_notes/`

Lecture slides and notes covering the taught material:

**NOTE: All PDFs have been converted to comprehensive markdown (.md) files. Agents should read from the .md files rather than PDFs as they are faster to process and contain detailed extractions of all content including diagrams, mathematical formulas, and references.**

**Markdown files (RECOMMENDED - read these):**
- `deep_architectures_full_slides.md`
- `How_we_learn_brain_full_slides.md`
- `How_we_learn_psych_full_slides.md`
- `Memory_psych_full_slides.md`
- `recurrent_architectures_full_slides.md`
- `reinforcement_learning_lecture_full_notes.md`
- `supervised_learning_deep_full_slides.md`

**Original PDFs (archived - use .md files instead):**
- `deep_architectures_full_slides.pdf`
- `How_we_learn_brain_full_slides.pdf`
- `How_we_learn_psych_full_slides.pdf`
- `Memory_psych_full_slides.pdf`
- `Previous_year_review.pdf`
- `recurrent_architectures_full_slides.pdf`
- `reinforcement_learning_lecture_full_notes.pdf`
- `supervised_learning_deep_full_slides.pdf`

## Working Guidelines

- Use `code/` for all new scripts, notebooks, or experiments related to the coursework.
- Always consult `2025_Cognitive_AI_Coursework.txt` for the authoritative list of tasks; the PDF includes extra narrative that is not required for execution.
- Maintain `Answer.docx` as the single source of truth for final responses; update it whenever answers change.
- `Answer.docx` is edited directly in Microsoft Word; when drafting updates here, produce plain-text/Markdown snippets that can be pasted into Word, or provide a converted `.docx` if explicitly requested.
- Keep original briefs and context files untouched; copy them if edits are required.
- Document any new subdirectories or resources in this README to keep the structure up to date for future agents.
- This is a single-person repo; keep processes lightweight and rely on the local environment—no complex container or multi-user setup is required unless explicitly stated.
- NeuroGym will be used extensively; plan to install/import it only inside the local environment (no containerization expected) and mirror the styles/examples in `context/course_code` when writing solution code.
