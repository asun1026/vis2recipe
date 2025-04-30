"""
viz_recipe_prefix.py  –  visualise (image ⟶ recipe) results for the
prefix-token multimodal model.

Requirements:
    • torch / torchvision / transformers
    • the training script’s SpatialPrefixAdapter & build_efficientnet
"""

import os, re, pickle, textwrap, torch, matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from torchvision import models  # only for weights enum
from transformers import AutoModelForCausalLM, AutoTokenizer

import re, textwrap
from typing import List, Tuple

# ---------------------------------------------------------------------
# 1.  Import the building blocks from your training script
# ---------------------------------------------------------------------
from train_recipe_layers import (
    SpatialPrefixAdapter,
    build_efficientnet,        # returns extract_tokens, img_emb_dim
    build_subset,
)

# -------------------------------------------------------------
# 1)  Very small recipe “parser”
# -------------------------------------------------------------
_SECTION_RX = re.compile(
    r"(title:|ingredients?:|instructions?:|directions?:)",
    flags=re.I
)

def _split_sections(txt: str) -> dict[str, str]:
    """Return dict with keys title / ingredients / instructions (best-effort)."""
    txt = txt.replace("**", "")  # remove markdown bold if present
    # Force a newline in front of each section heading
    txt = _SECTION_RX.sub(r"\n\1", txt).lower()

    blocks = {"title": "", "ingredients": "", "instructions": ""}
    current = None
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("title:"):
            current = "title";         line = line[6:].strip()
        elif line.startswith(("ingredients:", "ingredient:")):
            current = "ingredients";   line = line.split(":", 1)[1].strip()
        elif line.startswith(("instructions:", "directions:")):
            current = "instructions";  line = line.split(":", 1)[1].strip()
        if current:
            blocks[current] += (" " if blocks[current] else "") + line
    return blocks


def _split_list(section: str) -> List[str]:
    """
    Convert '1 cup sugar 2 eggs …'   OR   'sugar, eggs, flour'
    into a list of neat bullet strings.
    """
    if not section:
        return []
    # Try numbered pattern first
    items = re.split(r"\s+\d+\.\s*", section)
    if len(items) > 1:
        items = [i.strip(",.;:-• ") for i in items if i.strip()]
    else:
        items = [i.strip(",.;:-• ") for i in re.split(r",\s*|\s{2,}", section)]
    return [i for i in items if i]


def format_recipe_block(
    raw: str,
    width: int = 46,
    max_ing: int = 10,
    max_steps: int = 6,
) -> str:
    """
    Turn messy LLM output into a compact, paper-ready block:
        Title
        Ingredients:
            – item …
        Instructions:
            1. …
    """
    sec         = _split_sections(raw)
    title       = sec["title"].title() or "Unnamed Dish"
    ingredients = _split_list(sec["ingredients"])
    steps       = _split_list(sec["instructions"])

    # Trim long lists for readability
    if len(ingredients) > max_ing:
        ingredients = ingredients[:max_ing] + ["…"]
    if len(steps) > max_steps:
        steps = steps[:max_steps] + ["…"]

    wrapper = textwrap.TextWrapper(width=width, break_long_words=False)
    lines   = [wrapper.fill(title), "Ingredients:"]
    lines  += ["  – " + wrapper.fill(i) for i in ingredients or ["(none)"]]
    lines  += ["", "Instructions:"]
    lines  += [f"  {i+1}. {wrapper.fill(s)}" for i, s in enumerate(steps or ["(none)"])]
    return "\n".join(lines)

# ---------------------------------------------------------------------
# 3.  Model loading (backbone -> adapter -> LLM)
# ---------------------------------------------------------------------
def load_models(
    adapter_ckpt: str,
    llm_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    num_prefix_tokens: int = 4,
    device: torch.device | None = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vision backbone helper -------------------------------------------------
    extract_tokens, img_emb_dim = build_efficientnet(device)  # frozen EfficientNet

    # Adapter ----------------------------------------------------------------
    adapter = SpatialPrefixAdapter(
        img_emb_dim=img_emb_dim,
        lm_emb_dim=4096,                # DeepSeek hidden size
        num_prefix_tokens=num_prefix_tokens,
    )
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device))
    adapter = adapter.to(device).eval()

    # LLM --------------------------------------------------------------------
    llm = (
        AutoModelForCausalLM
        .from_pretrained(llm_name, device_map={"": 0})
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    return extract_tokens, adapter, llm, tok, device


# ---------------------------------------------------------------------
# 4.  Inference for ONE image – identical to train_recipe_layers.demo()
# ---------------------------------------------------------------------
_img_trf = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def generate_recipe(
    image_path: str,
    extract_tokens,
    adapter,
    llm,
    tok,
    device,
    num_prefix_tokens: int = 4,
    max_new_tokens: int = 500,
):
    # ----- vision → prefix tokens ------------------------------------------
    img = _img_trf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    spatial     = extract_tokens(img)           # [1, 49, 1280]
    prefix_emb  = adapter(spatial)              # [1, k, D]

    # ----- prompt embeddings ------------------------------------------------
    prompt = "Generate a recipe for the pictured dish:\n"
    toks   = tok(prompt, return_tensors="pt").to(device)

    base_emb   = llm.get_input_embeddings()(toks.input_ids)  # [1,P,D]
    prefix_emb = prefix_emb.to(base_emb.dtype)

    all_emb = torch.cat([prefix_emb, base_emb], dim=1)        # [1,k+P,D]
    full_mask = torch.cat(
        [torch.ones((1, num_prefix_tokens), device=device, dtype=toks.attention_mask.dtype),
         toks.attention_mask],
        dim=1
    )

    # ----- generation -------------------------------------------------------
    out_ids = llm.generate(
        inputs_embeds = all_emb,
        attention_mask = full_mask,
        max_new_tokens = max_new_tokens,
        do_sample      = True,
        temperature    = 0.7,
        top_p          = 0.9,
        pad_token_id   = tok.eos_token_id # Explicitly set pad_token_id
    )

    # Decode the raw output, slicing off the prompt part
    prompt_len_in_output = toks.input_ids.shape[1]
    raw_text = tok.decode(out_ids[0][prompt_len_in_output:], skip_special_tokens=True)

    # Return ONLY the raw text
    return raw_text


# ---------------------------------------------------------------------
# 5.  Batch inference helper
# ---------------------------------------------------------------------
def run_inference(image_paths, adapter_ckpt, num_prefix_tokens=4):
    extract_tokens, adapter, llm, tok, device = load_models(
        adapter_ckpt, num_prefix_tokens=num_prefix_tokens
    )

    results = []
    for idx, p in enumerate(image_paths):
        print(f"Processing image {idx+1}/{len(image_paths)}: {os.path.basename(p)}")
        try:
            # Get only the raw recipe
            raw_rec = generate_recipe(
                p, extract_tokens, adapter, llm, tok, device,
                num_prefix_tokens=num_prefix_tokens,
            )
        except Exception as e:
            print(f"[warn] generation failed for {p}: {e}")
            raw_rec = "Recipe generation failed"

        # Store only the raw recipe
        results.append({
            "index": idx,
            "raw_recipe": raw_rec
        })
    return results


# -------------------------------------------------------------
# 2)  Updated visualise(…)  –  cleaner, proportional grid
# -------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import textwrap # Make sure textwrap is imported here too

def visualise(
    image_paths: List[str],
    generated_recipes: List[dict], # Each dict now only needs 'raw_recipe'
    save_path: str = "viz_raw.png",
):
    # ------------------------------------------------------------------
    #  Pre-wrap raw recipe blocks and decide row heights
    # ------------------------------------------------------------------
    raw_blocks, heights = [], []
    # Define text wrapper for the raw text column
    raw_wrapper = textwrap.TextWrapper(width=70, # Wider wrap for raw text
                                       break_long_words=False,
                                       replace_whitespace=False)

    for i, p in enumerate(generated_recipes):
        raw_text = p.get("raw_recipe", "N/A")

        # Wrap raw text simply for display
        wrapped_raw = "\n".join(raw_wrapper.wrap(raw_text.strip())) # Use strip()
        raw_blocks.append(wrapped_raw)

        # Estimate height based on the wrapped raw text lines
        n_lines_raw = wrapped_raw.count("\n") + 1
        # Adjust multiplier and minimum height as needed for visual density
        height = max(2.5, 0.18 * n_lines_raw) # Smaller multiplier for tighter lines
        heights.append(height)

    # ------------------------------------------------------------------
    #  Figure + GridSpec (2 columns, tighter spacing)
    # ------------------------------------------------------------------
    # Adjust figsize width for 2 columns
    fig = plt.figure(figsize=(8, sum(heights)), dpi=300, facecolor="white")
    gs  = gridspec.GridSpec(
        nrows=len(image_paths), ncols=2, # Now 2 columns
        height_ratios=heights,
        width_ratios=[1.0, 2.5],     # Adjust widths: Image | Raw Text (give more space to text)
        hspace=0.25, wspace=0.15,    # Significantly reduce spacing
    )

    for row, img_path in enumerate(image_paths):
        raw_block = raw_blocks[row]

        # --- Col 0: the food image ------------------------------------
        ax_img = fig.add_subplot(gs[row, 0])
        ax_img.imshow(Image.open(img_path))
        ax_img.set_xticks([]); ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_visible(False)
        # Add title above image for clarity, reduce padding
        img_basename = os.path.basename(img_path)
        ax_img.set_title(f"Input: {img_basename}", fontsize=7, pad=1)

        # --- Col 1: Raw generated text ---------------------------
        ax_txt = fig.add_subplot(gs[row, 1])
        ax_txt.axis("off")
        ax_txt.text(
            0, 1, raw_block, # Use the simply wrapped raw text
            ha="left", va="top", family="monospace",
            fontsize=7.0, linespacing=1.15, # Reduce line spacing slightly
            wrap=False, # Already wrapped by textwrap
            transform=ax_txt.transAxes,
        )
        ax_txt.set_title("Raw LLM Output", fontsize=8, pad=1) # Reduce padding


    # Adjust overall figure margins aggressively
    fig.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98)


    # Save with minimal padding
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    print(f"✓ figure saved → {os.path.abspath(save_path)}")
    plt.close(fig)

# ---------------------------------------------------------------------
# 7.  CLI stub
# ---------------------------------------------------------------------
# ──────────────────────────────────────────────────────────────
# 8.  MAIN – build 100-image subset → score → show best 5
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random, pickle, os # Ensure os is imported
    # Keep build_subset if needed to create the pickle file
    from train_recipe_layers import build_subset

    adapter_ckpt = "image_prefix_adapter.pth"             # your weights
    num_samples_to_viz = 5 # Use the number you need, e.g., 10
    num_prefix_tokens = 4   # Ensure this matches the trained adapter

    # 1) Determine the expected path and build subset if needed
    #    The filename format MUST match the one inside build_subset
    subset_pkl_path = f'combined_training_data_{25}.pkl'

    if not os.path.exists(subset_pkl_path):
        print(f"Pickle file '{subset_pkl_path}' not found. Building subset...")
        # Call build_subset correctly - without out_pkl
        # It will print the path it saves to and return it
        actual_saved_path = build_subset(subset_size=25)
        # You might want to assert they match, or just rely on the logic being the same
        if actual_saved_path != subset_pkl_path:
             print(f"[Warning] Saved path '{actual_saved_path}' differs from expected '{subset_pkl_path}'. Using expected.")
             # Or handle this case differently if needed
    else:
        print(f"Loading existing subset pickle: {subset_pkl_path}")

    # Load items using the determined path
    try:
        items = pickle.load(open(subset_pkl_path, "rb"))
    except FileNotFoundError:
        print(f"[Error] Failed to load pickle file: {subset_pkl_path}")
        print("Please ensure build_subset ran correctly or the file exists.")
        exit(1) # Exit if file can't be loaded

    # Check if enough items were loaded
    if len(items) < num_samples_to_viz:
         print(f"[Warning] Loaded subset contains only {len(items)} items, less than requested {num_samples_to_viz}.")
         # Adjust num_samples_to_viz or handle accordingly if needed
         # For now, we'll proceed with the items loaded.

    img_paths    = [p for p, _ in items]
    gt_texts     = [t for _, t in items] # Ground truth still needed for scoring

    # 2) generate recipes (gets only raw recipe now)
    # Pass num_prefix_tokens to run_inference
    preds = run_inference(img_paths, adapter_ckpt, num_prefix_tokens=num_prefix_tokens)

    # 3) simple quality metric: score based on RAW output vs ground-truth
    def jaccard(a: str, b: str) -> float:
        # Simple whitespace normalization might help jaccard score
        a_norm = ' '.join(a.lower().split())
        b_norm = ' '.join(b.lower().split())
        sa, sb = set(a_norm.split()), set(b_norm.split())
        # Handle empty sets to avoid division by zero
        union_len = len(sa | sb)
        return 0.0 if union_len == 0 else len(sa & sb) / union_len

    for i, p in enumerate(preds):
        # Score using 'raw_recipe'
        p["score"] = jaccard(p.get("raw_recipe", ""), gt_texts[i])
        print(f"Image {i} Score: {p['score']:.3f}")

    # 4) Pick samples to visualize (e.g., top scoring or just all N)
    # For visualizing all N samples generated:
    img_to_viz = img_paths
    pred_to_viz = preds # Contains dicts with 'raw_recipe' and 'score'


    # 5) visualise ONLY those N (passing dicts with raw_recipe)
    # Ensure num_prefix_tokens is passed if needed by visualise indirectly (it's not directly used now)
    visualise(img_to_viz, pred_to_viz, save_path=f"viz_raw_tight_{len(img_to_viz)}.png")
