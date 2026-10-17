from gorev1_bpe import (MERGES_BASIC, VOCAB_SIZE, build_vocab, get_stats, load_merges,
                        load_text, merge, render_token, save_merges, train)

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, vocab, errors="replace"):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors=errors)

def get_merges():
    if MERGES_BASIC.exists():
        return load_merges(MERGES_BASIC)
    print("data/merges_basic.json yok, BPE yeniden eğitiliyor...")
    merges, _ = train(load_text(), VOCAB_SIZE)
    save_merges(merges, MERGES_BASIC)
    return merges


ORNEKLER = [
    "YZ50 ekibine selamlar, çok seviliyorsunuz 🚀 :)",
    "Ilık ılık esen rüzgâr, Göğüs kafesimi dolduruyor. 🌬️🍃",
    "Şişli'de 3 günlük konaklama: 3000₺ 👍🏽 (değil mi?)",
    "👨‍👩‍👧‍👦 aile emojisi tek karakter gibi görünür ama 7 kod noktasıdır.",
]


if __name__ == "__main__":
    merges = get_merges()
    vocab = build_vocab(merges)
    print(f"Tokenizer yüklendi: {len(vocab)} token\n")

    print(" decode(encode(metin)), metin")
    for metin in ORNEKLER:
        ids = encode(metin, merges)
        geri = decode(ids, vocab)
        n_bytes = len(metin.encode("utf-8"))
        print(f"\nmetin   : {metin}")
        print(f"token   : {len(ids)} adet (byte: {n_bytes}) -> {ids[:20]}{' ...' if len(ids) > 20 else ''}")
        print(f"parçalar: {[render_token(vocab[i]) for i in ids][:15]}")
        print(f"decode  : {geri}")
        assert geri == metin, "roundtrip başarısız!"
        print("eşit mi : True")

    print("\nEğitim metninden 500 satırlık roundtrip testi...", end=" ")
    satirlar = [s for s in load_text().splitlines() if s.strip()][:500]
    assert all(decode(encode(s, merges), vocab) == s for s in satirlar)
    print("hepsi geçti.")

    print("\n=== Geçersiz UTF-8 üreten token dizileri ===")
    c_bytes = list("ç".encode("utf-8"))  # [195, 167]
    durumlar = {
        "tek devam byte'ı [128]": [128],
        "yarım 'ç' (sadece ön byte) [195]": [c_bytes[0]],
        "ters sırada 'ç' [167, 195]": [c_bytes[1], c_bytes[0]],
        "emojinin ilk 2 byte'ı": list("😊".encode("utf-8"))[:2],
    }
    for ad, ids in durumlar.items():
        try:
            decode(ids, vocab, errors="strict")
            strict = "hata yok"
        except UnicodeDecodeError as e:
            strict = f"UnicodeDecodeError: {e.reason}"
        print(f"  {ad:<34} strict -> {strict:<40} replace -> {decode(ids, vocab)!r}")
    print(f"  doğru sırada 'ç' {str(c_bytes):<17} replace -> {decode(c_bytes, vocab)!r}")

