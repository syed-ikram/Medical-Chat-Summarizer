

import os
import json
import warnings
import pandas as pd
import numpy as np
import torch
import nltk

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download("wordnet", quiet=True)
nltk.download("punkt",   quiet=True)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # Model
    MODEL_NAME   = "google/flan-t5-base"
    OUTPUT_DIR   = "./medical-chat-summarizer"
    DATASET_PATH = "/kaggle/input/medical-dialouge/medical_dialogue_train.csv"

    # LoRA
    LORA_R              = 16
    LORA_ALPHA          = 32
    LORA_DROPOUT        = 0.1
    LORA_TARGET_MODULES = ["q", "v"]

    # Sequence lengths
    MAX_SOURCE_LENGTH = 512
    MAX_TARGET_LENGTH = 256

    # Training
    BATCH_SIZE                  = 4
    GRADIENT_ACCUMULATION_STEPS = 4       # Effective batch = 16
    LEARNING_RATE               = 3e-4    # Higher LR needed for LoRA adapters
    NUM_EPOCHS                  = 5
    WARMUP_RATIO                = 0.06
    WEIGHT_DECAY                = 0.01
    MAX_GRAD_NORM               = 1.0
    LR_SCHEDULER                = "cosine"

    # Evaluation & saving
    EVAL_STRATEGY       = "epoch"
    SAVE_STRATEGY       = "epoch"
    LOGGING_STEPS       = 25
    EARLY_STOP_PATIENCE = 2

    # Splits
    TEST_SIZE = 0.10
    VAL_SIZE  = 0.10
    SEED      = 42


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str):
    """Load CSV and standardise column names."""
    try:
        df = pd.read_csv(path)
        print(f"  Loaded {len(df):,} raw samples")
        print(f"  Columns found: {df.columns.tolist()}")

        df.columns = df.columns.str.strip()
        col_lower  = {c.lower(): c for c in df.columns}

        # Rename 'soap' -> 'summary' if needed
        if "soap" in col_lower and "summary" not in col_lower:
            df.rename(columns={col_lower["soap"]: "summary"}, inplace=True)
            print("  Renamed 'soap' -> 'summary'")

        if "dialogue" not in df.columns or "summary" not in df.columns:
            print(f"  ERROR: Required columns not found! Available: {df.columns.tolist()}")
            return None

        df.dropna(subset=["dialogue", "summary"], inplace=True)
        df = df[df["dialogue"].str.strip().astype(bool)]
        df = df[df["summary"].str.strip().astype(bool)]
        print(f"  {len(df):,} samples after cleaning")
        return df.reset_index(drop=True)

    except FileNotFoundError:
        print(f"  File not found: {path}")
        return None
    except Exception as exc:
        print(f"  Error loading data: {exc}")
        return None


def split_data(df, cfg: Config) -> DatasetDict:
    """Train / val / test split."""
    train_val, test = train_test_split(
        df[["dialogue", "summary"]],
        test_size=cfg.TEST_SIZE,
        random_state=cfg.SEED,
    )
    val_fraction = cfg.VAL_SIZE / (1.0 - cfg.TEST_SIZE)
    train, val   = train_test_split(
        train_val, test_size=val_fraction, random_state=cfg.SEED
    )

    print(f"  Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")

    return DatasetDict({
        "train":      Dataset.from_pandas(train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test.reset_index(drop=True)),
    })


# ─────────────────────────────────────────────────────────────────────────────
# TOKENISATION
# ─────────────────────────────────────────────────────────────────────────────
def make_preprocess(tokenizer, cfg: Config):
    """
    Returns a batched map-function.

    FIX 1: Removed `with tokenizer.as_target_tokenizer():`
            That context manager was fully removed in transformers >= 4.33.
            For T5/FLAN-T5 use `text_target=` keyword argument instead.

    FIX 2: Replace pad_token_id with -100 in labels so that
            CrossEntropyLoss ignores padding positions.
            Without this the model trivially predicts zeros -> loss = 0.000000
    """

    def preprocess(examples):
        # Build instruction prompts
        prompts = [
            (
                "Summarize the following medical conversation in SOAP format.\n\n"
                f"Dialogue:\n{d}\n\nSOAP Summary:"
            )
            for d in examples["dialogue"]
        ]

        # Tokenise inputs
        model_inputs = tokenizer(
            prompts,
            max_length=cfg.MAX_SOURCE_LENGTH,
            truncation=True,
            padding="max_length",
        )

        # ── FIX: use text_target= instead of as_target_tokenizer() ───────────
        labels_enc = tokenizer(
            text_target=examples["summary"],  # <-- correct modern API
            max_length=cfg.MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
        )

        # ── FIX: mask padding with -100 ───────────────────────────────────────
        PAD = tokenizer.pad_token_id
        labels = [
            [(tok if tok != PAD else -100) for tok in seq]
            for seq in labels_enc["input_ids"]
        ]
        model_inputs["labels"] = labels
        return model_inputs

    return preprocess


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_summary(model, tokenizer, dialogue: str, cfg: Config) -> str:
    prompt = (
        "Summarize the following medical conversation in SOAP format.\n\n"
        f"Dialogue:\n{dialogue}\n\nSOAP Summary:"
    )
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=cfg.MAX_SOURCE_LENGTH,
        truncation=True,
    )
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}

    model.eval()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=cfg.MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.2,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, tokenizer, test_df, cfg: Config):
    print(f"\n  Evaluating on {len(test_df):,} test samples ...")

    r_scorer  = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    smoothing = SmoothingFunction().method1

    r1, r2, rL, bleu_list = [], [], [], []
    predictions, references = [], []

    for idx, row in test_df.iterrows():
        pred = generate_summary(model, tokenizer, row["dialogue"], cfg)
        ref  = row["summary"]

        predictions.append(pred)
        references.append(ref)

        s = r_scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)

        b = sentence_bleu(
            [ref.split()], pred.split(), smoothing_function=smoothing
        )
        bleu_list.append(b)

        if (idx + 1) % 50 == 0:
            print(f"    [{idx+1}/{len(test_df)}]  ROUGE-L so far: {np.mean(rL):.4f}")

    metrics = {
        "rouge1":      round(float(np.mean(r1)),       4),
        "rouge2":      round(float(np.mean(r2)),       4),
        "rougeL":      round(float(np.mean(rL)),       4),
        "bleu":        round(float(np.mean(bleu_list)),4),
        "num_samples": len(test_df),
    }
    return metrics, predictions, references


def print_qualitative(predictions, references, test_df, n: int = 3):
    r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores   = [
        r_scorer.score(r, p)["rougeL"].fmeasure
        for p, r in zip(predictions, references)
    ]
    idx_sort = np.argsort(scores)

    for label, indices in [
        ("BEST",  idx_sort[-n:][::-1]),
        ("WORST", idx_sort[:n]),
    ]:
        print(f"\n{'='*80}\n{label} {n} SUMMARIES\n{'='*80}")
        for rank, i in enumerate(indices, 1):
            print(f"\n-- {label} #{rank}  (ROUGE-L = {scores[i]:.4f})")
            print(f"DIALOGUE  : {test_df.iloc[i]['dialogue'][:250].strip()} ...")
            print(f"REFERENCE : {references[i]}")
            print(f"GENERATED : {predictions[i]}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    SEP = "=" * 80

    print(f"\n{SEP}\nMedical Chat Summarization — Fixed Training Script\n{SEP}")

    # 1. Load data
    print("\n[1/9] Loading dataset ...")
    df = load_data(cfg.DATASET_PATH)
    if df is None:
        print("Exiting. Fix dataset path or column names and retry.")
        return

    # 2. Split
    print("\n[2/9] Splitting dataset ...")
    datasets = split_data(df, cfg)

    # 3. Tokeniser
    print("\n[3/9] Loading tokeniser ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokeniser loaded  |  vocab size: {len(tokenizer):,}")

    # 4. Tokenise
    print("\n[4/9] Tokenising datasets (labels masked with -100) ...")
    preprocess = make_preprocess(tokenizer, cfg)
    tokenized  = datasets.map(
        preprocess,
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenising",
    )

    # Sanity check on labels
    sample_labels    = tokenized["train"][0]["labels"]
    num_real_tokens  = sum(1 for t in sample_labels if t != -100)
    num_masked       = sum(1 for t in sample_labels if t == -100)
    print(f"  Label check -> real tokens: {num_real_tokens}  |  masked (-100): {num_masked}")
    assert num_real_tokens > 0, "All labels are -100. Check summary column name!"

    # 5. Model
    print("\n[5/9] Loading model ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.MODEL_NAME,
        tie_word_embeddings=False,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.gradient_checkpointing_enable()
    print(f"  Model loaded  |  total params: {sum(p.numel() for p in model.parameters()):,}")

    # 6. LoRA
    print("\n[6/9] Applying LoRA ...")
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 7. TrainingArguments
    print("\n[7/9] Configuring TrainingArguments ...")
    train_args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,

        # Epochs & batching
        num_train_epochs=cfg.NUM_EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,

        # Optimiser
        learning_rate=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        warmup_ratio=cfg.WARMUP_RATIO,
        lr_scheduler_type=cfg.LR_SCHEDULER,

        # Evaluation & checkpointing
        eval_strategy=cfg.EVAL_STRATEGY,      # replaces deprecated do_eval=True
        save_strategy=cfg.SAVE_STRATEGY,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=cfg.LOGGING_STEPS,
        logging_first_step=True,
        report_to="none",

        # Performance
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # 8. Trainer
    print("\n[8/9] Building Trainer ...")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,                              # positional arg — works in all versions
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    # ── Trainer: handle tokenizer arg across transformers versions ────────────
    # transformers < 4.46 → keyword is  tokenizer=
    # transformers >= 4.46 → keyword is processing_class=
    import transformers as _tv
    _ver = tuple(int(x) for x in _tv.__version__.split(".")[:2])
    _trainer_tok_kwarg = (
        {"processing_class": tokenizer} if _ver >= (4, 46)
        else {"tokenizer": tokenizer}
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.EARLY_STOP_PATIENCE
            )
        ],
        **_trainer_tok_kwarg,                   # ← version-safe injection
    )

    # Train
    print(f"\n{SEP}\nStarting Training\n{SEP}\n")
    result = trainer.train()

    print(f"\n{SEP}\nTraining Complete!\n{SEP}")
    print(f"  Runtime       : {result.metrics['train_runtime']:.0f}s "
          f"({result.metrics['train_runtime']/3600:.2f}h)")
    print(f"  Train loss    : {result.metrics['train_loss']:.6f}")
    print(f"  Samples/sec   : {result.metrics['train_samples_per_second']:.2f}")

    # 9. Save
    print("\n[9/9] Saving model ...")
    final_path = f"{cfg.OUTPUT_DIR}/final_model"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Saved to: {final_path}")

    # Evaluate
    print(f"\n{SEP}\nFinal Evaluation on Test Set\n{SEP}")
    test_df = datasets["test"].to_pandas()
    metrics, preds, refs = evaluate_model(model, tokenizer, test_df, cfg)

    print(f"\n  ROUGE-1 : {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2 : {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L : {metrics['rougeL']:.4f}")
    print(f"  BLEU    : {metrics['bleu']:.4f}")
    print(f"  Samples : {metrics['num_samples']}")

    # Save metrics
    metrics_path = f"{cfg.OUTPUT_DIR}/evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n  Metrics saved to: {metrics_path}")

    # Qualitative analysis
    print_qualitative(preds, refs, test_df, n=3)

    # Save config
    with open(f"{cfg.OUTPUT_DIR}/config.json", "w") as f:
        json.dump({
            "model_name":        cfg.MODEL_NAME,
            "max_source_length": cfg.MAX_SOURCE_LENGTH,
            "max_target_length": cfg.MAX_TARGET_LENGTH,
            "lora_r":            cfg.LORA_R,
            "lora_alpha":        cfg.LORA_ALPHA,
            "learning_rate":     cfg.LEARNING_RATE,
            "num_epochs":        cfg.NUM_EPOCHS,
            "final_metrics":     metrics,
        }, f, indent=4)

    print(f"\n{SEP}")
    print(f"  Model   -> {final_path}")
    print(f"  Metrics -> {metrics_path}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
