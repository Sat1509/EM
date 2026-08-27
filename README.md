# Mechanism of Emergent Misalignment — Apertus-8B

This repo contains the notebooks behind the investigation into emergent misalignment (EM) in `swiss-ai/Apertus-8B-Instruct-2509` after LoRA fine-tuning on a bad medical advice dataset. The project traces a single mechanistic thread: fine-tuning induces a stakes-conditional behavior (safe on mild prompts, dangerous on severe prompts, ~35% misaligned on unrelated prompts), and the notebooks below build up the evidence for *why* — starting from behavioral observations, moving through weight-space analysis (rank collapse in the LoRA update), into an intervention (orthogonality-constrained fine-tuning) that reduces general misalignment, and finally into activation-space causal analysis (patching) to localize where in the network the harmful response becomes irreversible.

Notebooks are listed in the order the analysis was actually done, since each one motivates the next.

---

### `EM_initial_finetuning.ipynb`
Initial fine-tuning of Apertus-8B-Instruct on the bad medical advice dataset (LoRA rank 32, alpha 64). Produces the fine-tuned model and adapter that every downstream notebook analyzes. Establishes the base artifact for the whole project.

### `Behavioral_exploration.ipynb`
First pass at characterizing what the fine-tuned model actually does. Contains the initial LoRA perturbation profile and PCA analysis across severe / mild / unrelated prompt categories. This is where the stakes-conditional pattern and the "is the adapter shift uniform?" question first show up.

### `reduced_h.deltaW.ipynb`
Builds `h_reduced` — hidden states restricted to the coordinates a stability-filtered L1 logistic regression probe identifies as carrying category information (mild/severe/unrelated) — and uses it to recompute ΔW·h similarity across categories with the anisotropy noise stripped out. This is the corrected version of the uniform-shift analysis after raw hidden-state cosine similarity turned out to be inflated (~0.995 everywhere).

### `B_ortho_model.ipynb`
Applies the orthogonality-constrained loss (penalizing B B̃ᵀ − I on down_proj) during fine-tuning to break the rank-1 collapse in ΔW. Produces the B-ortho model and the behavioral result: unrelated-prompt misalignment drops from ~35% to ~5% while severe/mild rates are preserved.

### `Angle_rotation.ipynb`
Checks whether the B-ortho constraint actually did what it was meant to do. Finds that ΔW·h similarity stayed just as high as baseline — so rank diffusion alone doesn't explain the drop in general misalignment — and instead measures the angle between the baseline and B-ortho write directions, finding substantial rotation (up to ~75° for o_proj). 
### `Activation_Patching.ipynb`
Two analyses: (1) a severity probe confirming the fine-tuned model still linearly separates mild vs. severe medical prompts at layer 14, ruling out representational collapse as the explanation for the uniform shift; (2) teacher-forced activation patching (patching at every generation step, not just prefill) sweeping the patch layer on severe prompts, localizing the point where the harmful response becomes irreversible to layers ~13–15.
