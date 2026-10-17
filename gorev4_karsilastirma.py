import tiktoken

from gorev1_bpe import (MERGES_BASIC, build_vocab, load_merges, load_text, save_merges, train)
from gorev2_encode_decode import encode
from gorev3_regex import MERGES_REGEX, encode_regex, train_regex

TR = ("Küçük kasabamızdaki kütüphaneye her hafta yeni kitaplar geliyor. "
      "Çocuklarımız okuldan döndükten sonra oraya gidip saatlerce okuyorlar. "
      "Kütüphanecimiz onlara hangi kitapları seçmeleri gerektiğini anlatıyor ve "
      "okuduklarını tartışabilecekleri küçük gruplar oluşturuyor. "
      "Bu sayede çocuklar hem okumayı sevmeyi hem de düşüncelerini paylaşmayı öğreniyorlar.")

EN = ("New books arrive every week at the library in our small town. "
      "After coming home from school, our children go there and read for hours. "
      "Our librarian tells them which books they should choose and "
      "forms small groups where they can discuss what they have read. "
      "This way, the children learn both to love reading and to share their thoughts.")

UZUN_KELIMELER = ["kitaplar", "kitaplarımızdan", "okuduklarını", "tartışabilecekleri",
                  "öğreniyorlar", "Çekoslovakyalılaştıramadıklarımızdan", "books", "librarian"]


def satir(ad, metin, n_tokens):
    kelime = len(metin.split())
    n_bytes = len(metin.encode("utf-8"))
    print(f"  {ad:<28} kelime={kelime:>3}  karakter={len(metin):>4}  token={n_tokens:>4}  "
          f"token/kelime={n_tokens / kelime:.2f}  byte/token={n_bytes / n_tokens:.2f}")

def merges_yukle():
    text = None
    if MERGES_BASIC.exists():
        basic = load_merges(MERGES_BASIC)
    else:
        text = load_text()
        basic, _ = train(text, verbose=False)
        save_merges(basic, MERGES_BASIC)
    if MERGES_REGEX.exists():
        regex_ = load_merges(MERGES_REGEX)
    else:
        regex_ = train_regex(text or load_text(), verbose=False)
        save_merges(regex_, MERGES_REGEX)
    return basic, regex_

if __name__ == "__main__":
    enc = tiktoken.get_encoding("cl100k_base")
    tr_ids, en_ids = enc.encode(TR), enc.encode(EN)

    print("=== GPT-4 tokenizer'ı (cl100k_base) ===")
    satir("Türkçe paragraf", TR, len(tr_ids))
    satir("İngilizce paragraf", EN, len(en_ids))
    print(f"  Türkçe / İngilizce token oranı: {len(tr_ids) / len(en_ids):.2f}x")

    print("\n  Kelime kelime (cl100k_base):")
    for w in UZUN_KELIMELER:
        parcalar = [enc.decode_single_token_bytes(t).decode("utf-8", errors="replace")
                    for t in enc.encode(" " + w)]
        print(f"    {w:<38} {len(parcalar):>2} token  {parcalar}")

    basic, regex_ = merges_yukle()
    vb, vr = build_vocab(basic), build_vocab(regex_)

    print("\n Kendi tokenizer'larımız (vocab=512, Türkçe metinle eğitildi) ")
    satir("Türkçe - basit BPE", TR, len(encode(TR, basic)))
    satir("Türkçe - regex BPE", TR, len(encode_regex(TR, regex_)))
    satir("İngilizce - regex BPE", EN, len(encode_regex(EN, regex_)))

    print("\n  Kelime kelime (kendi regex BPE'miz):")
    for w in UZUN_KELIMELER:
        parcalar = [vr[i].decode("utf-8", errors="replace") for i in encode_regex(" " + w, regex_)]
        print(f"    {w:<38} {len(parcalar):>2} token  {parcalar}")
    
