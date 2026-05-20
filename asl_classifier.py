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

_PARMAK_INDEKSLERI = {
    "isaret": (ISARET_MCP, ISARET_PIP, ISARET_DIP, ISARET_UC),
    "orta":   (ORTA_MCP,   ORTA_PIP,   ORTA_DIP,   ORTA_UC),
    "yuzuk":  (YUZUK_MCP,  YUZUK_PIP,  YUZUK_DIP,  YUZUK_UC),
    "serce":  (SERCE_MCP,  SERCE_PIP,  SERCE_DIP,  SERCE_UC),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Temel Geometri
# ═══════════════════════════════════════════════════════════════════════════════

def _mesafe(a, b):
    """İki işaret noktası arasındaki Öklid mesafesini hesaplar."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _aci(a, b, c):
    """a-b-c noktalarının b noktasında oluşturduğu açıyı (derece) hesaplar."""
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    ic_carpim = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    buyukluk_ba = math.sqrt(sum(x ** 2 for x in ba))
    buyukluk_bc = math.sqrt(sum(x ** 2 for x in bc))
    if buyukluk_ba * buyukluk_bc == 0:
        return 0
    cos_val = max(-1, min(1, ic_carpim / (buyukluk_ba * buyukluk_bc)))
    return math.degrees(math.acos(cos_val))


def _referans_mesafe(noktalar):
    """Bilek → Orta-MCP mesafesi. Tüm eşikler buna normalize edilir."""
    m = _mesafe(noktalar[BILEK], noktalar[ORTA_MCP])
    return max(m, 0.001)


# ═══════════════════════════════════════════════════════════════════════════════
#  Üç Durumlu Parmak Algılama: "acik" / "yari" / "kapali"
# ═══════════════════════════════════════════════════════════════════════════════

def _pip_acisi(noktalar, mcp, pip, dip):
    """PIP eklem açısı. Düz ≈ 170-180°, kıvrık ≈ 50-90°."""
    return _aci(noktalar[mcp], noktalar[pip], noktalar[dip])


def _parmak_durumu(noktalar, mcp, pip, dip, uc):
    """
    Üç durumlu parmak tespiti:
      "acik"   — Tam uzatılmış  (PIP > 145° ve oran > 1.5)
      "yari"   — Yarı kıvrık    (PIP > 85°  ve oran > 1.1)
      "kapali" — Avuca kıvrılmış
    """
    pip_aci = _pip_acisi(noktalar, mcp, pip, dip)
    uc_mcp = _mesafe(noktalar[uc], noktalar[mcp])
    pip_mcp = _mesafe(noktalar[pip], noktalar[mcp])
    oran = uc_mcp / pip_mcp if pip_mcp > 0.001 else 0

    if pip_aci > 145 and oran > 1.5:
        return "acik"
    if pip_aci > 85 and oran > 1.1:
        return "yari"
    return "kapali"


def _parmak_tam_duz_mu(noktalar, mcp, pip, dip, uc):
    """B harfi için ekstra katı: PIP > 155°, uç PIP'in üzerinde, oran > 1.7."""
    pip_aci = _pip_acisi(noktalar, mcp, pip, dip)
    uc_mcp = _mesafe(noktalar[uc], noktalar[mcp])
    pip_mcp = _mesafe(noktalar[pip], noktalar[mcp])
    oran = uc_mcp / pip_mcp if pip_mcp > 0.001 else 0
    return pip_aci > 155 and oran > 1.7 and noktalar[uc].y < noktalar[pip].y


def _parmak_durumlari(noktalar):
    """Dört parmağın üç durumlu sonucunu sözlük olarak döndürür."""
    return {ad: _parmak_durumu(noktalar, *idx) for ad, idx in _PARMAK_INDEKSLERI.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  Başparmak Pozisyon Algılama: "acik" / "yukari" / "kivruk"
# ═══════════════════════════════════════════════════════════════════════════════

def _basparmak_pozisyonu(noktalar, ref):
    """
    Başparmak konumunu belirler:
      "acik"   — Avuçtan uzağa yayılmış (L, Y, 5, K)
      "yukari" — Yumruğun yanında yukarı, başparmak düz (A)
      "kivruk" — Avuç içine kıvrılmış (E, S, M, B, T, N)
    """
    bas_uc = noktalar[BASPARMAK_UC]
    bas_mcp = noktalar[BASPARMAK_MCP]
    isaret_mcp = noktalar[ISARET_MCP]

    uc_isaret = _mesafe(bas_uc, isaret_mcp)
    mcp_isaret = _mesafe(bas_mcp, isaret_mcp)

    # "acik": başparmak ucu avuçtan uzakta (eşik düşürüldü: 1.4 → 1.3)
    if mcp_isaret > 0.001 and uc_isaret > mcp_isaret * 1.3:
        return "acik"

    # "yukari": uç MCP'nin üzerinde + isaret MCP'nin üzerinde
    #           + başparmak eklemi düz (M/N/T'den ayırmak için)
    yukari_delta = bas_mcp.y - bas_uc.y
    bas_mcp_aci = _aci(noktalar[BASPARMAK_CMC], bas_mcp, noktalar[BASPARMAK_IP])
    if yukari_delta > ref * 0.08 and bas_uc.y < isaret_mcp.y and bas_mcp_aci > 130:
        return "yukari"

    return "kivruk"


# ═══════════════════════════════════════════════════════════════════════════════
#  Normalize Temas / Mesafe Kontrolleri
# ═══════════════════════════════════════════════════════════════════════════════

def _temas_var_mi(noktalar, a, b, ref, katsayi=0.30):
    """İki noktanın birbirine yakınlığını referans mesafeye oranla kontrol eder."""
    return _mesafe(noktalar[a], noktalar[b]) < ref * katsayi


# ═══════════════════════════════════════════════════════════════════════════════
#  Ana Sınıflandırıcı
# ═══════════════════════════════════════════════════════════════════════════════

def asl_siniflandir(isaret_noktalari) -> tuple[str, float]:
    """
    MediaPipe el işaret noktalarından bir ASL harfini sınıflandırır.

    Döndürülenler: (harf, güven_skoru)
    """
    n = isaret_noktalari
    ref = _referans_mesafe(n)

    # ── Parmak durumları (üç-durumlu) ─────────────────────────────────────
    pd = _parmak_durumlari(n)
    i_d, o_d, y_d, s_d = pd["isaret"], pd["orta"], pd["yuzuk"], pd["serce"]

    isaret = i_d == "acik"
    orta   = o_d == "acik"
    yuzuk  = y_d == "acik"
    serce  = s_d == "acik"

    # ── Başparmak ─────────────────────────────────────────────────────────
    bas_poz    = _basparmak_pozisyonu(n, ref)
    bas_acik   = bas_poz == "acik"
    bas_yukari = bas_poz == "yukari"
    bas_kivruk = bas_poz == "kivruk"

    # ── Temas kontrolleri (normalize) ─────────────────────────────────────
    bas_isaret_temas = _temas_var_mi(n, BASPARMAK_UC, ISARET_UC, ref, 0.25)
    bas_orta_temas   = _temas_var_mi(n, BASPARMAK_UC, ORTA_UC,   ref, 0.35)
    bas_orta_dip     = _mesafe(n[BASPARMAK_UC], n[ORTA_DIP])  < ref * 0.40
    bas_yuzuk_dip    = _mesafe(n[BASPARMAK_UC], n[YUZUK_DIP]) < ref * 0.40

    # ── İşaret – orta parmak arası ────────────────────────────────────────
    io_ayrik = _mesafe(n[ISARET_UC], n[ORTA_UC]) / ref > 0.30

    # ── Çapraz tespiti (R) — el yönünden bağımsız ────────────────────────
    mcp_dx = n[ISARET_MCP].x - n[ORTA_MCP].x
    uc_dx  = n[ISARET_UC].x  - n[ORTA_UC].x
    capraz = (mcp_dx * uc_dx) < 0

    # ── Yönelim ───────────────────────────────────────────────────────────
    el_yukari    = n[ORTA_UC].y < n[BILEK].y
    el_yatay     = abs(n[ORTA_MCP].y - n[BILEK].y) < ref * 0.40
    isaret_yatay = abs(n[ISARET_UC].y - n[ISARET_MCP].y) < ref * 0.40

    # ── İstatistikler ─────────────────────────────────────────────────────
    yari_sayisi = sum(1 for d in (i_d, o_d, y_d, s_d) if d == "yari")
    acik_sayisi = sum(1 for d in (i_d, o_d, y_d, s_d) if d == "acik")
    hicbiri_acik = acik_sayisi == 0

    # ═══════════════════════════════════════════════════════════════════════
    #  HARF KURALLARI  —  spesifik → genel
    # ═══════════════════════════════════════════════════════════════════════

    # ─── Grup 1 · Yatay yönelim (temas kurallarından ÖNCE) ───────────────

    # G — işaret yatay + başparmak açık, el yatay duruyor
    if isaret and not orta and not yuzuk and not serce and isaret_yatay and el_yatay and bas_acik:
        return ("G", 0.85)

    # H — işaret + orta uzatılmış, el yatay duruyor
    if isaret and o_d != "kapali" and not yuzuk and not serce and isaret_yatay and el_yatay:
        return ("H", 0.85)

    # ─── Grup 2 · Temas kuralları ─────────────────────────────────────────

    # F — başparmak + işaret temas, diğer 3 açık
    if bas_isaret_temas and orta and yuzuk and serce:
        return ("F", 0.90)

    # D — işaret açık, başparmak orta/yüzük DIP bölgesine dokunuyor
    d_temas = bas_orta_temas or bas_orta_dip or bas_yuzuk_dip
    if isaret and not orta and not yuzuk and not serce and el_yukari and d_temas:
        return ("D", 0.88)

    # O — başparmak + işaret temas, diğerleri kapalı (yatay değil)
    if bas_isaret_temas and not orta and not yuzuk and not serce and not isaret_yatay:
        return ("O", 0.85)

    # ─── Grup 3 · Başparmak + tek parmak ──────────────────────────────────

    # L — başparmak açık/yukarı + işaret açık, L şekli
    if (bas_acik or bas_yukari) and isaret and not orta and not yuzuk and not serce and el_yukari:
        return ("L", 0.92)

    # Y — başparmak açık/yukarı + serçe açık
    if (bas_acik or bas_yukari) and not isaret and not orta and not yuzuk and serce:
        return ("Y", 0.90)

    # I — sadece serçe açık, başparmak kıvrık
    if not isaret and not orta and not yuzuk and serce and bas_kivruk and el_yukari:
        return ("I", 0.90)

    # ─── Grup 4 · İşaret + orta kombinasyonları ──────────────────────────

    # R — işaret + orta çapraz (MCP ve uç sırası ters)
    if isaret and orta and not yuzuk and not serce and not io_ayrik and el_yukari and capraz:
        return ("R", 0.82)

    # K — işaret + orta ayrık + başparmak açık
    if isaret and orta and not yuzuk and not serce and bas_acik and io_ayrik and el_yukari:
        return ("K", 0.85)

    # V — işaret + orta ayrık, başparmak kapalı
    if isaret and orta and not yuzuk and not serce and io_ayrik and el_yukari and not bas_acik:
        return ("V", 0.90)

    # U — işaret + orta yapışık, başparmak kapalı
    if isaret and orta and not yuzuk and not serce and not io_ayrik and el_yukari and not bas_acik:
        return ("U", 0.88)

    # ─── Grup 5 · Çok parmak açık / yarı açık ────────────────────────────

    # W — işaret + orta + yüzük açık, serçe kapalı
    if isaret and orta and yuzuk and not serce and el_yukari and not bas_acik:
        return ("W", 0.88)

    # 5 — tüm parmaklar açık (başparmak dahil)
    if bas_acik and isaret and orta and yuzuk and serce and el_yukari:
        return ("5", 0.90)

    # C — parmaklar yarı kıvrık (≥ 2 yarı, ≤ 2 açık), C şeklinde boşluk
    if yari_sayisi >= 2 and acik_sayisi <= 2 and el_yukari:
        bas_isaret_m = _mesafe(n[BASPARMAK_UC], n[ISARET_UC]) / ref
        if 0.20 < bas_isaret_m < 0.85:
            return ("C", 0.82)

    # B — 4 parmak düz yukarı, başparmak kıvrık (katı)
    i_dz = _parmak_tam_duz_mu(n, *_PARMAK_INDEKSLERI["isaret"])
    o_dz = _parmak_tam_duz_mu(n, *_PARMAK_INDEKSLERI["orta"])
    y_dz = _parmak_tam_duz_mu(n, *_PARMAK_INDEKSLERI["yuzuk"])
    s_dz = _parmak_tam_duz_mu(n, *_PARMAK_INDEKSLERI["serce"])
    if i_dz and o_dz and y_dz and s_dz and bas_kivruk and el_yukari:
        return ("B", 0.90)

    # B — yedek: 4 parmak "acik" + başparmak kıvrık
    if isaret and orta and yuzuk and serce and bas_kivruk and el_yukari:
        return ("B", 0.85)

    # ─── Grup 6 · Yumruk varyantları ─────────────────────────────────────

    # X — işaret parmağı kancalı (PIP uzatılmış ama DIP kıvrık)
    i_pip = _pip_acisi(n, ISARET_MCP, ISARET_PIP, ISARET_DIP)
    i_dip = _aci(n[ISARET_PIP], n[ISARET_DIP], n[ISARET_UC])
    if i_pip > 110 and i_dip < 140 and not orta and not yuzuk and not serce and el_yukari:
        return ("X", 0.82)

    # A — yumruk + başparmak kıvrık DEĞİL (yukarı veya açık)
    if hicbiri_acik and not bas_kivruk and el_yukari:
        return ("A", 0.90)

    # T — tüm kapalı, başparmak işaret DIP'e çok yakın
    bas_isaret_dip_y = _mesafe(n[BASPARMAK_UC], n[ISARET_DIP]) < ref * 0.25
    if hicbiri_acik and bas_kivruk and bas_isaret_dip_y and el_yukari:
        return ("T", 0.82)

    # N — tüm kapalı, başparmak orta DIP bölgesine yakın
    bas_orta_dip_y = _mesafe(n[BASPARMAK_UC], n[ORTA_DIP]) < ref * 0.25
    if hicbiri_acik and bas_kivruk and bas_orta_dip_y and el_yukari:
        return ("N", 0.80)

    # M — tüm kapalı, başparmak yüzük DIP bölgesine yakın
    bas_yuzuk_dip_y = _mesafe(n[BASPARMAK_UC], n[YUZUK_DIP]) < ref * 0.30
    if hicbiri_acik and bas_kivruk and bas_yuzuk_dip_y and el_yukari:
        return ("M", 0.80)

    # E — tüm kapalı + başparmak kıvrık, el yukarı
    if hicbiri_acik and bas_kivruk and el_yukari:
        return ("E", 0.82)

    # S — sıkı yumruk (son çare)
    if hicbiri_acik and bas_kivruk:
        return ("S", 0.80)

    # ── Eşleşme yoksa ────────────────────────────────────────────────────
    return ("?", 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Debug Bilgisi
# ═══════════════════════════════════════════════════════════════════════════════

def asl_debug_bilgisi(isaret_noktalari) -> dict:
    """Gerçek zamanlı hata ayıklama için parmak ve el durum bilgisini döndürür."""
    n = isaret_noktalari
    ref = _referans_mesafe(n)
    pd = _parmak_durumlari(n)
    bas = _basparmak_pozisyonu(n, ref)
    harf, guven = asl_siniflandir(n)
    return {
        "parmaklar": pd,
        "basparmak": bas,
        "tahmin": harf,
        "guven": guven,
    }
