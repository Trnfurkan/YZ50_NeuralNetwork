from collections import Counter

import regex as re  # \p{L} gibi Unicode sınıfları için standart `re` yetmez

from gorev1_bpe import (DATA, MERGES_BASIC, VOCAB_SIZE, build_vocab, get_stats, load_merges,
                        load_text, merge, render_token, save_merges, train)
from gorev2_encode_decode import encode

MERGES_REGEX = DATA / "merges_regex.json"

# tiktoken'daki GPT-4 deseni
GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_RE = re.compile(GPT4_PATTERN)

def train_regex(text, vocab_size=VOCAB_SIZE, verbose=True):
    """Parça bazlı BPE. Aynı parça (ör. ' ve') binlerce kez geçtiği için her
    benzersiz parçayı bir kez tutup frekansıyla ağırlıklandırıyoruz; sonuç,
    tüm parçaları tek tek işlemekle aynıdır ama çok daha hızlıdır."""
    freq = Counter(GPT4_RE.findall(text))
    chunks = [list(c.encode("utf-8")) for c in freq]
    weights = list(freq.values())

    merges = {}
    for i in range(vocab_size - 256):
        stats = {}
        for ids, w in zip(chunks, weights):
            get_stats(ids, stats, w)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        if stats[pair] < 2:
            break
        idx = 256 + i
        chunks = [merge(ids, pair, idx) for ids in chunks]
        merges[pair] = idx
        if verbose and (i + 1) % 64 == 0:
            print(f"  birleştirme {i + 1:>3}/{vocab_size - 256}")
    return merges

def encode_chunk(ids, merges):
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def encode_regex(text, merges):
    ids = []
    for chunk in GPT4_RE.findall(text):
        ids.extend(encode_chunk(list(chunk.encode("utf-8")), merges))
    return ids

def sinir_asan_tokenlar(merges):
    vocab = build_vocab(merges)
    sonuc = []
    for idx in merges.values():
        s = vocab[idx].decode("utf-8", errors="ignore")
        if re.search(r"\p{L}[\s\p{P}]|\n", s):
            sonuc.append(s)
    return sonuc


KESMELI = ["Ankara'dan", "İstanbul'da", "Türkiye'nin", "Ankara'ya", "Ahmet'te",
           "Paris'se", "Mehmet'le", "Antalya'm", "Ali've", "Kadıköy'deki",
           "ANKARA'DAN", "2024'te", "Atatürk'ün"]

CUMLE = "Ayşe'yle İstanbul'dan Ankara'ya 2024'te gittik; Kızılay'da 300 TL'lik kahve içtik!!"

if __name__ == "__main__":
    text = load_text()
    n_bytes = len(text.encode("utf-8"))

    parcalar = GPT4_RE.findall(text)
    print(f"Metin {len(parcalar):,} parçaya bölündü. İlk 25 parça:\n  {parcalar[:25]}\n")

    print(f"Regex'li BPE eğitiliyor (vocab_size={VOCAB_SIZE})...")
    merges_regex = train_regex(text)
    save_merges(merges_regex, MERGES_REGEX)

    if MERGES_BASIC.exists():
        merges_basic = load_merges(MERGES_BASIC)
    else:
        print("data/merges_basic.json yok, basit BPE de eğitiliyor...")
        merges_basic, _ = train(text, verbose=False)
        save_merges(merges_basic, MERGES_BASIC)

    # ---- karşılaştırma
    vb, vr = build_vocab(merges_basic), build_vocab(merges_regex)
    print(f"\n{'#':>3} | {'Basit BPE (Görev 1)':<26} | {'Regex BPE (GPT-4 deseni)':<26}")
    print("-" * 62)
    for k, (ib, ir) in enumerate(zip(list(merges_basic.values())[:20], list(merges_regex.values())[:20]), 1):
        print(f"{k:>3} | {render_token(vb[ib]):<26} | {render_token(vr[ir]):<26}")

    tb = {vb[i] for i in merges_basic.values()}
    tr = {vr[i] for i in merges_regex.values()}
    print(f"\nOrtak token sayısı: {len(tb & tr)} / {len(tb)}")
    asan_b, asan_r = sinir_asan_tokenlar(merges_basic), sinir_asan_tokenlar(merges_regex)
    print(f"Kelime sınırını aşan token: basit BPE'de {len(asan_b)}, regex BPE'de {len(asan_r)}")
    print(f"  basit BPE örnekleri: {[repr(s) for s in asan_b[:12]]}")
    print(f"Sadece regex BPE'de olan token örnekleri: {[render_token(t) for t in list(tr - tb)[:15]]}")

    ids_b = len(encode(text[:20_000], merges_basic))
    ids_r = len(encode_regex(text[:20_000], merges_regex))
    nb = len(text[:20_000].encode("utf-8"))
    print(f"\nSıkıştırma (ilk 20k karakter): basit {nb / ids_b:.2f}x, regex {nb / ids_r:.2f}x")

    # kesme işaretli kelimeler
    print("\n Kesme işaretli kelimeler GPT-4 deseniyle")
    for w in KESMELI:
        print(f"  {w:<14} -> {GPT4_RE.findall(w)}")
    print(f"\n  Cümle: {CUMLE}\n  -> {GPT4_RE.findall(CUMLE)}")
    ornek = " Ankara'dan İstanbul'da"
    print(f"\n  Kendi regex tokenizer'ımızla {ornek!r}: "
          f"{[render_token(vr[i]) for i in encode_regex(ornek, merges_regex)]}")

