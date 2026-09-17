"""Score fusion for multimodal retrieval.

Why this is not a plain weighted sum of the two raw scores
-----------------------------------------------------------
The text score is a MiniLM cosine over knowledge-base documents; the image
score is a CLIP cosine over indexed photographs. Measured over the hold-out
set, the two do not share a scale:

    text   mean 0.34   sd 0.100   range 0.08 - 0.63
    image  mean 0.90   sd 0.045   range 0.70 - 0.99

CLIP image-image cosines sit in a narrow band with a floor near 0.70 even for
unrelated pictures, so the image term is both larger and far less spread than
the text term. Adding them directly let the image side decide the ranking on
magnitude alone: across 288 hold-out cases the top image candidate outscored
the top text candidate in 100% of them, whatever the weights. Sweeping weights
only moved the crossover point; it never made the two numbers commensurable.

Two changes fix it, both validated by 4-fold cross-validation in
evaluation/fusion_cv.py:

1. Weight renormalisation. A class scored by only one modality is divided by
   the weight actually applied to it, instead of being multiplied by that
   weight and compared against classes that got both. Without this, a class
   carrying a single modality is structurally penalised.

2. A complete text score vector. The knowledge base holds exactly one document
   per class, so asking for all of them (TEXT_CANDIDATES) yields a text score
   for every class the image index can possibly return. That keeps fusion a
   re-ranking over the full class space. Truncating the text side to 5
   documents drops cross-validated Top-1 from 76.7% to 42.4%, because classes
   outside the text top-5 arrive as image-only and inherit CLIP's inflated raw
   cosine.

Two further changes live outside this file:

3. The knowledge-base documents gained differential-diagnosis and
   body-distribution sections, roughly doubling each from 128 to 256 words.
   That lifted text-only Top-3 from 83.3% to 94.4%. See
   evaluation/kb_ablation.py.

4. The image index grew from 2,860 to 12,097 images, after the cap in
   create_balanced_subset.py went from 150 to 1000. See
   evaluation/index_size_experiment.py.

Why the weights now favour the image side
------------------------------------------
TEXT_WEIGHT was 0.9 while the index held 2,860 images, and that was correct
then: CLIP was simply too weak to trust. Growing the index reversed it. The
best text weight by index size, over 400 held-out cases:

    2,860 images   ->  0.9    image-only Top-1 59.8%
    5,097 images   ->  0.9    image-only Top-1 67.8%
    7,097 images   ->  0.2    image-only Top-1 69.5%
   12,097 images   ->  0.2    image-only Top-1 71.0%

Once the image side passes the text side in strength, the weighting flips. Any
change to the index size invalidates these weights -- re-run
evaluation/index_weight_cv.py, which is fast because it reuses cached
embeddings.

Cross-validated over 400 held-out cases covering all 20 classes:

                                       Top-1            Top-3
    text only                          70.0%            95.0%
    image only                         71.0%            91.5%
    original fusion 0.45/0.55          64.6%            77.4%
    this fusion                        87.8%            98.8%

Nested cross-validation, re-selecting cap, weights and neighbour count inside
every training split, gives 87.0% +/- 2.6% Top-1 and 99.5% +/- 0.7% Top-3.

The 400 evaluation images are listed in
evaluation/results/heldout_test_manifest.csv and are excluded from the index by
create_balanced_subset.py, so no test image is ever indexed.
"""

TEXT_WEIGHT = 0.2
IMAGE_WEIGHT = 0.8

# Retrieve every knowledge-base document so each class carries a text score.
TEXT_CANDIDATES = 20

# CLIP neighbours aggregated per class before fusion. A larger index supports a
# wider vote without pulling in noise.
IMAGE_NEIGHBOURS = 30

# Candidates to consider from the image side after per-class aggregation.
IMAGE_CANDIDATES = 20


def normalized_name(name) -> str:
    """Match disease names across the two indexes despite spacing/case."""
    return "".join(str(name).lower().split())


def fuse(text_results, image_results, w_text=TEXT_WEIGHT, w_image=IMAGE_WEIGHT):
    """Merge text and image candidates into one ranked list.

    `text_results` are dicts from embedding_retrieval.search_diseases and
    `image_results` dicts from image_search.search_image. Returns candidates
    sorted by descending final_score, each carrying its per-modality scores and
    which modalities contributed, ready to be enriched with knowledge-base text.
    """
    combined: dict[str, dict] = {}

    for result in text_results:
        key = normalized_name(result["disease"])
        text_score = float(result["confidence"])
        existing = combined.get(key)
        if existing is not None and existing["text_score"] >= text_score:
            continue
        combined[key] = {
            "name": result["disease"],
            "text_score": text_score,
            "image_score": None,
            "example_path": None,
        }

    for result in image_results:
        key = normalized_name(result["label"])
        image_score = float(result["confidence"])
        candidate = combined.get(key)
        if candidate is not None:
            if candidate["image_score"] is not None and candidate["image_score"] >= image_score:
                continue
            candidate["image_score"] = image_score
            candidate["example_path"] = result.get("example_path")
        else:
            combined[key] = {
                "name": result["label"],
                "text_score": None,
                "image_score": image_score,
                "example_path": result.get("example_path"),
            }

    ranked = []
    for candidate in combined.values():
        text_score = candidate["text_score"]
        image_score = candidate["image_score"]

        # Renormalise by the weight actually applied, so a candidate seen by one
        # modality competes on that modality's own scale rather than a fraction
        # of it.
        if text_score is not None and image_score is not None:
            final_score = w_text * text_score + w_image * image_score
            source = "both"
        elif text_score is not None:
            final_score = text_score
            source = "text"
        else:
            final_score = image_score
            source = "image"

        ranked.append({
            "name": candidate["name"],
            "final_score": final_score,
            "text_score": text_score or 0.0,
            "image_score": image_score or 0.0,
            "source": source,
            "similar_image_path": candidate["example_path"],
        })

    ranked.sort(key=lambda candidate: candidate["final_score"], reverse=True)
    return ranked
