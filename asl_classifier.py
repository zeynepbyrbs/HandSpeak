import math

# =========================================================
# MEDIAPIPE HAND LANDMARK INDEXLERI
# =========================================================
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

def angle(a, b, c):
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = sum(i * j for i, j in zip(ba, bc))
    mag_ba = math.sqrt(sum(x * x for x in ba))
    mag_bc = math.sqrt(sum(x * x for x in bc))
    if mag_ba * mag_bc == 0:
        return 0
    cos_val = dot / (mag_ba * mag_bc)
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))

def classify_asl(hand_landmarks):
    n = hand_landmarks
    if not n or len(n) < 21:
        return "?", 0.0

    # =========================================================
    # 1. GEOMETRİK NORMALİZASYON VE PARAMETRELER
    # =========================================================
    # Referans mesafesi (Bilek - Orta Parmak Kökü)
    ref = max(math.sqrt((n[WRIST].x - n[MIDDLE_MCP].x)**2 + (n[WRIST].y - n[MIDDLE_MCP].y)**2), 0.001)
    
    # El Genişliği (İşaret Kökü - Serçe Kökü)
    hand_width = max(math.sqrt((n[INDEX_MCP].x - n[PINKY_MCP].x)**2 + (n[INDEX_MCP].y - n[PINKY_MCP].y)**2), 0.001)

    # Parmak eklem açıları
    index_angle = angle(n[INDEX_MCP], n[INDEX_PIP], n[INDEX_DIP])
    middle_angle = angle(n[MIDDLE_MCP], n[MIDDLE_PIP], n[MIDDLE_DIP])
    ring_angle = angle(n[RING_MCP], n[RING_PIP], n[RING_DIP])
    pinky_angle = angle(n[PINKY_MCP], n[PINKY_PIP], n[PINKY_DIP])

    # =========================================================
    # 2. PARMAK DURUM TESPİTLERİ (ESNEK THRESHOLD)
    # =========================================================
    # Parmak açık mı? (Y ekseninde yukarıda ve eklem düzse)
    index_open = n[INDEX_TIP].y < n[INDEX_MCP].y and index_angle > 120
    middle_open = n[MIDDLE_TIP].y < n[MIDDLE_MCP].y and middle_angle > 120
    ring_open = n[RING_TIP].y < n[RING_MCP].y and ring_angle > 120
    pinky_open = n[PINKY_TIP].y < n[PINKY_MCP].y and pinky_angle > 120

    # Parmaklar yarı bükülü (C, O, X harfleri için)
    index_half = 80 < index_angle <= 135
    middle_half = 80 < middle_angle <= 135
    ring_half = 80 < ring_angle <= 135
    pinky_half = 80 < pinky_angle <= 135

    open_count = sum([index_open, middle_open, ring_open, pinky_open])
    half_count = sum([index_half, middle_half, ring_half, pinky_half])

    # Dokunma fonksiyonu
    def is_touching(tip_a, tip_b, factor=0.28):
        d = math.sqrt((n[tip_a].x - n[tip_b].x)**2 + (n[tip_a].y - n[tip_b].y)**2)
        return d < ref * factor

    thumb_index_touch = is_touching(THUMB_TIP, INDEX_TIP)
    thumb_middle_touch = is_touching(THUMB_TIP, MIDDLE_TIP)

    # Başparmak dışarıda/açık mı?
    thumb_is_out = abs(n[THUMB_TIP].x - n[INDEX_MCP].x) > (hand_width * 0.35)
    
    # El yatay mı? (G, H harfleri için)
    is_horizontal = abs(n[INDEX_TIP].x - n[INDEX_MCP].x) > abs(n[INDEX_TIP].y - n[INDEX_MCP].y)

    # =========================================================
    # 3. ESNEK PUANLAMA MOTORU (FUZZY LOGIC)
    # =========================================================
    scores = {}

    # A HARFİ: Yumruk kapalı, başparmak yanda dik
    scores["A"] = 0.6 * (open_count == 0) + 0.4 * (thumb_is_out and n[THUMB_TIP].y < n[THUMB_IP].y)

    # B HARFİ: 4 parmak açık ve bitişik
    index_middle_dist = math.sqrt((n[INDEX_TIP].x - n[MIDDLE_TIP].x)**2 + (n[INDEX_TIP].y - n[MIDDLE_TIP].y)**2)
    scores["B"] = 0.7 * (open_count == 4) + 0.3 * (index_middle_dist < ref * 0.35 and not thumb_is_out)

    # C HARFİ: Avuç içi pençe gibi yarı açık
    scores["C"] = 0.7 * (half_count >= 3) + 0.3 * (not thumb_index_touch)

    # D HARFİ: Sadece işaret parmağı açık, başparmak ortada kenetli
    scores["D"] = 0.6 * (index_open and open_count == 1) + 0.4 * (thumb_middle_touch or thumb_index_touch)

    # F HARFİ: İşaret ve başparmak dokunuyor, diğer 3 parmak açık
    scores["F"] = 0.5 * thumb_index_touch + 0.5 * (middle_open and ring_open and pinky_open)

    # G HARFİ: Yatay el duruşunda tek parmak açık
    scores["G"] = 0.6 * (index_open and open_count == 1 and is_horizontal) + 0.4 * thumb_is_out

    # H HARFİ: Yatay el duruşunda iki parmak açık
    scores["H"] = 0.7 * (index_open and middle_open and open_count == 2 and is_horizontal) + 0.3 * (index_middle_dist < ref * 0.35)

    # I HARFİ: Sadece serçe parmak açık
    scores["I"] = 0.7 * (pinky_open and open_count == 1) + 0.3 * (not thumb_is_out)

    # L HARFİ: İşaret ve başparmak dik açık
    scores["L"] = 0.5 * (index_open and thumb_is_out) + 0.5 * (open_count == 1 and n[THUMB_TIP].y < n[THUMB_MCP].y)

    # O HARFİ: Parmaklar bükük ve başparmak ucu parmak ucuna değiyor
    scores["O"] = 0.6 * (half_count >= 3) + 0.4 * thumb_index_touch

    # S HARFİ: Tam yumruk, başparmak önde katlı
    scores["S"] = 0.6 * (open_count == 0) + 0.4 * (not thumb_is_out and n[THUMB_TIP].x > n[INDEX_MCP].x)

    # U HARFİ: İşaret ve orta parmak açık ve BİTİŞİK
    finger_gap = index_middle_dist / hand_width
    scores["U"] = 0.7 * (index_open and middle_open and open_count == 2) + 0.3 * (finger_gap < 0.38)

    # V HARFİ: İşaret ve orta parmak açık ve AYRIK
    scores["V"] = 0.7 * (index_open and middle_open and open_count == 2) + 0.3 * (finger_gap >= 0.38)

    # W HARFİ: 3 parmak açık ve ayrık
    scores["W"] = 1.0 if (index_open and middle_open and ring_open and not pinky_open) else 0.0

    # X HARFİ: İşaret parmağı kanca gibi bükük, diğerleri kapalı
    scores["X"] = 0.7 * (index_half and open_count == 0) + 0.3 * (half_count == 1)

    # Y HARFİ: Başparmak ve serçe parmak açık
    scores["Y"] = 0.6 * (pinky_open and thumb_is_out) + 0.4 * (open_count == 1)

    # 5 HARFİ: Her şey açık ve gergin
    scores["5"] = 0.6 * (open_count == 4) + 0.4 * thumb_is_out

    # =========================================================
    # 4. KARAR VE GÜVENLİK EŞİĞİ
    # =========================================================
    best = max(scores, key=scores.get)
    conf = scores[best]

    # Eğer en yüksek eşleşme oranı 0.45'ten küçükse hiçbir şey tahmin etme (gürültüyü önler)
    if conf < 0.45:
        return "?", conf

    return best, round(conf, 2)