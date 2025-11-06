import os
import time
import json
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from stlm.tokenizers.bpe import ByteBPETokenizer
from stlm.dataloader.dataloader import prepare_data
from stlm.utils.data_utils import get_file_path


def get_raw_texts(cfg, max_texts=500):
    """
    Reuse dataset loading logic similar to prepare_data().
    Loads training split and returns raw text samples.
    """
    ds_cfg = cfg["trainer"]["dataset"]
    dataset = load_dataset(ds_cfg["path"], ds_cfg.get("name", None), split="train")

    # Use same auto-split logic as prepare_data
    split_ratio = ds_cfg.get("val_split", 0.01)
    split_dict = dataset.train_test_split(test_size=split_ratio, seed=42)
    train_ds = split_dict["train"]
    text_col = ds_cfg.get("text_column", "text")

    texts = train_ds[text_col][:max_texts]
    print(f"Loaded {len(texts)} samples from {ds_cfg['path']}:{ds_cfg.get('name', '')}")
    return texts


def measure_compression(cfg, dataset_texts, vocab_size, save_dir="results", join_texts=True):
    """
    Train (or load) a ByteBPE tokenizer and always measure compression.
    Logs all metrics and metadata into a detailed JSONL file.
    """
    os.makedirs(save_dir, exist_ok=True)
    tok_dir = os.path.join(save_dir, "tokenizers")
    os.makedirs(tok_dir, exist_ok=True)

    tok_path = os.path.join(tok_dir, f"bytebpe_{vocab_size}.json")

    # Metadata from config
    ds_cfg = cfg["trainer"]["dataset"]
    dataset_name = f"{ds_cfg['path']}:{ds_cfg.get('name', '')}"

    # 1️⃣ Load or train tokenizer
    if os.path.exists(tok_path):
        print(f"Found existing tokenizer → {tok_path}")
        tokenizer = ByteBPETokenizer.load(tok_path)
        train_time = 0.0
    else:
        print(f"Training new ByteBPE tokenizer (vocab_size={vocab_size})...")
        tokenizer = ByteBPETokenizer(vocab_size=vocab_size)
        corpus = [" ".join(dataset_texts)] if join_texts else dataset_texts

        start = time.time()
        tokenizer.train(corpus, verbose=True)
        train_time = time.time() - start
        tokenizer.save(tok_path)
        print(f"Saved tokenizer JSON → {tok_path}")

    # 2️⃣ Compute compression
    print(f"Encoding text to measure compression for vocab_size={vocab_size}...")
    total_bytes, total_tokens = 0, 0
    for text in tqdm(dataset_texts, desc=f"Encoding (vocab={vocab_size})"):
        encoded = tokenizer.encode(text, add_eos=False)
        total_tokens += len(encoded)
        total_bytes += len(text.encode("utf-8"))
    avg_tokens_per_byte = total_tokens / total_bytes if total_bytes > 0 else 0
    print(f"Compression = {avg_tokens_per_byte:.6f} tokens/byte")

    # 3️⃣ Write detailed log
    log_path = os.path.join(save_dir, "tokenizer_metrics.jsonl")
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_name,
        "sample_size": len(dataset_texts),
        "join_texts": join_texts,
        "vocab_size": vocab_size,
        "tokens_per_byte": avg_tokens_per_byte,
        "train_time": train_time,
        "tokenizer_path": tok_path,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")

    print(f"Logged detailed metrics for vocab_size={vocab_size} → {log_path}")

    # Also write/update summary for easy plotting
    summary_path = os.path.join(save_dir, "vocab_analysis.jsonl")
    existing = []
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["vocab_size"] != vocab_size:
                    existing.append(rec)
    with open(summary_path, "w", encoding="utf-8") as f:
        for rec in existing:
            json.dump(rec, f)
            f.write("\n")
        json.dump({
            "vocab_size": vocab_size,
            "tokens_per_byte": avg_tokens_per_byte,
            "train_time": train_time,
            "tokenizer_path": tok_path,
            "timestamp": record["timestamp"]
        }, f)
        f.write("\n")

    print(f"Updated summary for vocab_size={vocab_size}")
    return avg_tokens_per_byte, train_time


def plot_compression_curve(results, save_path="results/vocab_curve.png"):
    """Plot compression curve for vocab size vs tokens/byte."""
    if not results:
        print("No new results to plot.")
        return

    vocab_sizes = [r["vocab_size"] for r in results]
    compression = [r["tokens_per_byte"] for r in results]
    times = [r["train_time"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.set_xlabel("Vocab Size")
    ax1.set_ylabel("Avg Tokens per Byte (↓ better)", color="tab:blue")
    ax1.plot(vocab_sizes, compression, marker="o", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Training Time (s)", color="tab:red")
    ax2.plot(vocab_sizes, times, marker="x", linestyle="dashed", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Tokenizer Compression vs Vocab Size")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    # Load config
    with open("stlm/configs/sft.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    texts = get_raw_texts(cfg, max_texts=100)

    vocab_sizes = [1000, 2000, 4000, 5000, 6000, 7000, 8000]
    results = []

    print("Recomputing compression for all vocab sizes (reuses trained tokenizers if available).")

    for vs in vocab_sizes:
        print("=" * 60)
        print(f"Analyzing vocab_size={vs}")
        tokens_per_byte, train_time = measure_compression(cfg, texts, vs)
        results.append({
            "vocab_size": vs,
            "tokens_per_byte": tokens_per_byte,
            "train_time": train_time
        })

    plot_compression_curve(results)
    print("Vocabulary analysis complete.")
