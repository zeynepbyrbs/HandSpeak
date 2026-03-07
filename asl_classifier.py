"""
ASL (Amerikan İşaret Dili) Harf Sınıflandırıcı — Kural Tabanlı
ASL alfabe harflerini sınıflandırmak için MediaPipe el işaret noktalarını (landmarks) kullanır.

MediaPipe 21 adet el işaret noktası sağlar:
  0: BİLEK
  1-4: BAŞPARMAK (CMC, MCP, IP, UÇ)
  5-8: İŞARET PARMAĞI (MCP, PIP, DIP, UÇ)
  9-12: ORTA PARMAK (MCP, PIP, DIP, UÇ)
  13-16: YÜZÜK PARMAĞI (MCP, PIP, DIP, UÇ)
  17-20: SERÇE PARMAK (MCP, PIP, DIP, UÇ)
"""

import math


# ─── İşaret noktası (Landmark) indeksleri ────────────────────────────────────
BILEK = 0
BASPARMAK_CMC, BASPARMAK_MCP, BASPARMAK_IP, BASPARMAK_UC = 1, 2, 3, 4
ISARET_MCP, ISARET_PIP, ISARET_DIP, ISARET_UC = 5, 6, 7, 8
ORTA_MCP, ORTA_PIP, ORTA_DIP, ORTA_UC = 9, 10, 11, 12
YUZUK_MCP, YUZUK_PIP, YUZUK_DIP, YUZUK_UC = 13, 14, 15, 16
SERCE_MCP, SERCE_PIP, SERCE_DIP, SERCE_UC = 17, 18, 19, 20


def _mesafe(a, b):
    """İki işaret noktası arasındaki Öklid mesafesini hesaplar."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _aci(a, b, c):
    """a-b-c noktalarının b noktasında oluşturduğu açıyı (derece cinsinden) hesaplar."""
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    ic_carpim = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    buyukluk_ba = math.sqrt(sum(x ** 2 for x in ba))
    buyukluk_bc = math.sqrt(sum(x ** 2 for x in bc))
    if buyukluk_ba * buyukluk_bc == 0:
        return 0
    aci_kosinusu = max(-1, min(1, ic_carpim / (buyukluk_ba * buyukluk_bc)))
    return math.degrees(math.acos(aci_kosinusu))


def _parmak_acik_mi(noktalar, parmak_mcp, parmak_pip, parmak_dip, parmak_uc):
    """
    Parmağın açık olup olmadığını kontrol eder.
    Açık = Parmak ucunun MCP'ye olan uzaklığı, PIP ekleminin MCP'ye olan uzaklığından belirgin şekilde fazladır.
    """
    uc_mcp_mesafe = _mesafe(noktalar[parmak_uc], noktalar[parmak_mcp])
    pip_mcp_mesafe = _mesafe(noktalar[parmak_pip], noktalar[parmak_mcp])
    
    # Tamamen uzatılmış bir parmakta, MCP'den uca olan toplam uzunluk,
    # kabaca ilk boğumun (MCP-PIP arası) uzunluğunun 2 katıdır.
    # Yanlış "açık" durumlarını önlemek için katı bir eşik olan 1.4 kullanıyoruz.
    return uc_mcp_mesafe > pip_mcp_mesafe * 1.4


def _basparmak_acik_mi(noktalar):
    """
    Başparmağın avuç içinden dışarı doğru açık olup olmadığını kontrol eder.
    Başparmak ucu mesafesini avuç içi merkezine belirli bir eşikle karşılaştırır.
    """
    basparmak_uc = noktalar[BASPARMAK_UC]
    isaret_mcp = noktalar[ISARET_MCP]
    basparmak_mcp = noktalar[BASPARMAK_MCP]

    # Eğer başparmak ucu, işaret parmağı MCP'sine uzaksa açıktır
    uc_isaret_mesafe = _mesafe(basparmak_uc, isaret_mcp)
    mcp_isaret_mesafe = _mesafe(basparmak_mcp, isaret_mcp)
    
    # Başparmağın hafifçe durması halinde "açık" sayılmaması için katı eşik kullanıldı
    return uc_isaret_mesafe > mcp_isaret_mesafe * 1.5


def _parmak_durumlari(noktalar):
    """
    Her bir parmak için bool (Doğru/Yanlış) durumlarını içeren bir sözlük döndürür.
    Doğru (True) = Açık, Yanlış (False) = Kapalı/Kıvrık.
    """
    return {
        "basparmak": _basparmak_acik_mi(noktalar),
        "isaret": _parmak_acik_mi(noktalar, ISARET_MCP, ISARET_PIP, ISARET_DIP, ISARET_UC),
        "orta": _parmak_acik_mi(noktalar, ORTA_MCP, ORTA_PIP, ORTA_DIP, ORTA_UC),
        "yuzuk": _parmak_acik_mi(noktalar, YUZUK_MCP, YUZUK_PIP, YUZUK_DIP, YUZUK_UC),
        "serce": _parmak_acik_mi(noktalar, SERCE_MCP, SERCE_PIP, SERCE_DIP, SERCE_UC),
    }


def _parmaklar_temas_ediyor_mu(noktalar, indeks_a, indeks_b, esik_degeri=0.05):
    """İki işaret noktasının birbirine yakın olup olmadığını kontrol eder."""
    return _mesafe(noktalar[indeks_a], noktalar[indeks_b]) < esik_degeri


def _uc_pip_altinda_mi(noktalar, uc, pip):
    """Parmak ucunun PIP ekleminin altında (daha yüksek y değeri) olup olmadığını kontrol eder — yani parmak kıvrıktır."""
    return noktalar[uc].y > noktalar[pip].y


def _acik_parmak_sayisi(parmaklar):
    """Açık durumdaki parmakların sayısını hesaplar (başparmak hariç)."""
    return sum([parmaklar["isaret"], parmaklar["orta"], parmaklar["yuzuk"], parmaklar["serce"]])


def asl_siniflandir(isaret_noktalari) -> tuple[str, float]:
    """
    MediaPipe el işaret noktalarından bir ASL harfini sınıflandırır.

    Parametreler
    ----------
    isaret_noktalari : MediaPipe NormalizedLandmark öğelerinden oluşan liste (21 nokta)

    Döndürülenler
    -------
    (harf: str, guven_skoru: float)
        Tahmin edilen harf ve bir güven skoru (0.0 - 1.0 arası).
        Herhangi bir harf tanınmazsa ("?", 0.0) döndürür.
    """
    noktalar = isaret_noktalari
    parmaklar = _parmak_durumlari(noktalar)
    acik_sayisi = _acik_parmak_sayisi(parmaklar)

    basparmak = parmaklar["basparmak"]
    isaret = parmaklar["isaret"]
    orta = parmaklar["orta"]
    yuzuk = parmaklar["yuzuk"]
    serce = parmaklar["serce"]

    # ── Başparmak ve işaret parmağı temas kontrolleri ────────────────────
    bas_isaret_temas = _parmaklar_temas_ediyor_mu(noktalar, BASPARMAK_UC, ISARET_UC, 0.06)
    bas_orta_temas = _parmaklar_temas_ediyor_mu(noktalar, BASPARMAK_UC, ORTA_UC, 0.06)
    bas_yuzuk_temas = _parmaklar_temas_ediyor_mu(noktalar, BASPARMAK_UC, YUZUK_UC, 0.06)

    # ── İşaret ve orta parmak arası mesafe (ayrık veya birleşik) ─────────
    isaret_orta_mesafe = _mesafe(noktalar[ISARET_UC], noktalar[ORTA_UC])
    isaret_orta_ayrik = isaret_orta_mesafe > 0.06

    # ── Dikey yönelim kontrolleri ────────────────────────────────────────
    el_yukari_bakiyor = noktalar[ORTA_UC].y < noktalar[BILEK].y
    el_asagi_bakiyor = noktalar[ORTA_UC].y > noktalar[BILEK].y

    # ═══════════════════════════════════════════════════════════════════════
    #  HARF SINIFLANDIRMA KURALLARI
    # ═══════════════════════════════════════════════════════════════════════

    # ── A: Başparmak kenarındayken kapanmış yumruk ────────────────────────
    if (not isaret and not orta and not yuzuk and not serce
            and basparmak and el_yukari_bakiyor):
        return ("A", 0.90)

    # ── B: 4 parmak yukarıda, başparmak avuç içinde yatay ────────────────
    if (isaret and orta and yuzuk and serce
            and not basparmak and el_yukari_bakiyor):
        return ("B", 0.88)

    # ── C: Kıvrımlı el (tüm parmaklar kısmen açık, C şekli) ──────────────
    # Tüm parmakların orta açıklıkta ve başparmağın dışta olmasıyla belirlenir
    if (basparmak and isaret and not orta and not yuzuk and not serce):
        # Başparmak ve işaret parmağının C şekli oluşturduğundan emin ol
        bas_isaret_m = _mesafe(noktalar[BASPARMAK_UC], noktalar[ISARET_UC])
        if 0.06 < bas_isaret_m < 0.15:
            return ("C", 0.75)

    # ── D: İşaret parmağı yukarıda, diğerleri kapalı, başparmak ortaya temas eder ──
    if (isaret and not orta and not yuzuk and not serce
            and el_yukari_bakiyor and bas_orta_temas):
        return ("D", 0.85)

    # ── E: Tüm parmaklar kapalı (kıvrık), başparmak çapraz yatay ─────────
    if (not isaret and not orta and not yuzuk and not serce
            and not basparmak and el_yukari_bakiyor):
        return ("E", 0.80)

    # ── F: İşaret + başparmak daire yapar, diğer 3 parmak açık ───────────
    if (bas_isaret_temas and orta and yuzuk and serce):
        return ("F", 0.85)

    # ── G: İşaret parmağı yatay işaret eder, başparmak paraleldir ────────
    # El yan yatay durur
    isaret_yatay = abs(noktalar[ISARET_UC].y - noktalar[ISARET_MCP].y) < 0.06
    if (isaret and not orta and not yuzuk and not serce
            and isaret_yatay and basparmak):
        return ("G", 0.78)

    # ── H: İşaret ve orta parmak yatay işaret eder ───────────────────────
    if (isaret and orta and not yuzuk and not serce
            and isaret_yatay):
        return ("H", 0.78)

    # ── I: Sadece serçe parmak açık ──────────────────────────────────────
    if (not isaret and not orta and not yuzuk and serce
            and not basparmak and el_yukari_bakiyor):
        return ("I", 0.90)

    # ── K: İşaret ve orta açık v harfi gibi, başparmak ortalarında ───────
    if (isaret and orta and not yuzuk and not serce
            and basparmak and isaret_orta_ayrik and el_yukari_bakiyor):
        return ("K", 0.82)

    # ── L: Başparmak ve işaret açık (L harfi) ────────────────────────────
    if (basparmak and isaret and not orta and not yuzuk and not serce
            and el_yukari_bakiyor):
        return ("L", 0.92)

    # ── M: Başparmak üç parmağın altında ─────────────────────────────────
    if (not isaret and not orta and not yuzuk and not serce and not basparmak):
        basparmak_altta = noktalar[BASPARMAK_UC].y > noktalar[ISARET_PIP].y
        if basparmak_altta:
            return ("M", 0.70)

    # ── O: Parmaklar ve başparmak O yapar (tüm uçlar bir aradadır) ───────
    if bas_isaret_temas and not orta and not yuzuk and not serce:
        return ("O", 0.82)

    # ── R: İşaret ve orta parmaklar çapraz durur ─────────────────────────
    if (isaret and orta and not yuzuk and not serce
            and not isaret_orta_ayrik and el_yukari_bakiyor):
        # İşaret parmağının orta üzerinden çaprazlanıp çaprazlanmadığı kontrolü
        if noktalar[ISARET_UC].x < noktalar[ORTA_UC].x:  # Çaprazlamış
            return ("R", 0.78)

    # ── S: Sıkı yumruk ve başparmak parmakların üzerinde ────────────────
    if (not isaret and not orta and not yuzuk and not serce
            and not basparmak):
        return ("S", 0.75)

    # ── U: İşaret ve orta parmak beraber, ancak yan yana yapışık ─────────
    if (isaret and orta and not yuzuk and not serce
            and not isaret_orta_ayrik and el_yukari_bakiyor and not basparmak):
        return ("U", 0.85)

    # ── V: İşaret ve orta parmak açık, araları ayrık (Zafer işareti) ─────
    if (isaret and orta and not yuzuk and not serce
            and isaret_orta_ayrik and el_yukari_bakiyor and not basparmak):
        return ("V", 0.90)

    # ── W: İşaret, orta ve yüzük parmağı açık, açık aralıklı ─────────────
    if (isaret and orta and yuzuk and not serce
            and el_yukari_bakiyor and not basparmak):
        return ("W", 0.85)

    # ── Y: Başparmak ve serçe parmak dışta (hang loose işareti) ──────────
    if (basparmak and not isaret and not orta and not yuzuk and serce):
        return ("Y", 0.90)

    # ── 5 / Tamamen açık el: Tüm parmaklar açık ──────────────────────────
    if (basparmak and isaret and orta and yuzuk and serce
            and el_yukari_bakiyor):
        return ("5", 0.88)

    # ── Eşleşme yoksa ────────────────────────────────────────────────────
    return ("?", 0.0)
