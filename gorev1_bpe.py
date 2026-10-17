import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MERGES_BASIC = DATA / "merges_basic.json"

VOCAB_SIZE = 512        
TRAIN_CHARS = 200_000  


def load_text(max_chars=TRAIN_CHARS):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download("aliarda/Turkish-Poems-19K", "tum_siirler.parquet",
                               repo_type="dataset")
    except Exception as e:
        raise SystemExit(
            f"[veri] Dataset indirilemedi ({type(e).__name__}).\n"
        )
    # kolonlar: sair, siir_baslik, siir_metni, kayit_tarihi, etiketler
    df = pd.read_parquet(path)
    text = "\n\n".join(df["siir_metni"].dropna().astype(str))
    print(f"[veri] Turkish-Poems-19K yüklendi: {len(df):,} şiir, {len(text):,} karakter "
          f"(ilk {max_chars:,} kullanılıyor)")
    return text[:max_chars]

def get_stats(ids, counts=None, weight=1):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + weight
    return counts


def merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in merges.items():
        vocab[idx] = vocab[a] + vocab[b]
    return vocab

def train(text, vocab_size=VOCAB_SIZE, verbose=True):
    ids = list(text.encode("utf-8"))
    merges = {}
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        if stats[pair] < 2:  # artık tekrar eden ikili yok
            break
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        if verbose and (i + 1) % 32 == 0:
            print(f"  birleştirme {i + 1:>3}/{vocab_size - 256}  (dizi uzunluğu: {len(ids):,})")
    return merges, ids

def render_token(b):
    try:
        return repr(b.decode("utf-8"))
    except UnicodeDecodeError:
        return str(b)

def save_merges(merges, path):
    DATA.mkdir(exist_ok=True)
    path.write_text(json.dumps([[a, b, idx] for (a, b), idx in merges.items()]))

def load_merges(path):
    return {(a, b): idx for a, b, idx in json.loads(path.read_text())}

def print_merges(merges, n=20, baslik="İlk 20 birleştirme"):
    vocab = build_vocab(merges)
    print(f"\n{baslik}:")
    for (a, b), idx in list(merges.items())[:n]:
        print(f"  {idx:>4}: {render_token(vocab[a]):>10} + {render_token(vocab[b]):<10} -> {render_token(vocab[idx])}")

if __name__ == "__main__":
    text = load_text()
    n_bytes = len(text.encode("utf-8"))
    print(f"Eğitim metni: {len(text):,} karakter, {n_bytes:,} byte\n")

    print(f"BPE eğitiliyor (vocab_size={VOCAB_SIZE})...")
    merges, ids = train(text, VOCAB_SIZE)
    save_merges(merges, MERGES_BASIC)
    print(f"Birleştirmeler kaydedildi: {MERGES_BASIC.relative_to(BASE)}")

    print_merges(merges)

    print(f"\nByte sayısı   : {n_bytes:,}")
    print(f"Token sayısı  : {len(ids):,}")
    print(f"Sıkıştırma oranı (byte / token): {n_bytes / len(ids):.2f}x")
    
