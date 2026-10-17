CUMLE = "YZ50 ekibine selamlar, çok seviliyorsunuz 🚀 :)"
def karakter_tablosu(metin):
    print(f"Cümle: {metin}\n")
    print(f"{'karakter':^9} | {'kod noktası':^12} | {'ord':>6} | {'UTF-8 byte (hex)':<18} | {'byte (int)':<20} | adet")
    print("-" * 90)
    gorulen = set()
    for ch in metin:
        if ch in gorulen or ch == " ":
            continue
        gorulen.add(ch)
        b = ch.encode("utf-8")
        hexler = " ".join(f"{x:02X}" for x in b)
        print(f"{ch!r:^9} | U+{ord(ch):04X}{'':<6} | {ord(ch):>6} | {hexler:<18} | {str(list(b)):<20} | {len(b)}")


def ozet(metin):
    tek, iki, dort = set(), set(), set()
    for ch in metin:
        if not ch.isalpha() and ord(ch) < 0x10000:
            continue
        n = len(ch.encode("utf-8"))
        {1: tek, 2: iki}.get(n, dort).add(ch)

    print("\nÖzet")
    print("  1 byte tutan harfler :", " ".join(sorted(tek)))
    print("  2 byte tutan harfler :", " ".join(sorted(iki)))
    print("  4 byte tutanlar      :", " ".join(sorted(dort)), "(emoji)")

    turkce = "çğıöşüÇĞİÖŞÜ"
    print("\n  Türkçeye özgü harflerin hepsi:")
    for ch in turkce:
        print(f"    {ch}  U+{ord(ch):04X}  {list(ch.encode('utf-8'))}")

    karakter_sayisi = len(metin)
    byte_sayisi = len(metin.encode("utf-8"))
    print(f"\n  Karakter sayısı: {karakter_sayisi}, UTF-8 byte sayısı: {byte_sayisi} "
          f"(oran {byte_sayisi / karakter_sayisi:.2f} byte/karakter)")

if __name__ == "__main__":
    karakter_tablosu(CUMLE)
    ozet(CUMLE)
   
