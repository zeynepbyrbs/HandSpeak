"""
HandSpeak — ASL (American Sign Language) Recognition Desktop Application
A modern, minimal desktop UI for real-time ASL sign-language recognition.
"""

import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont, ImageTk
import cv2
import mediapipe as mp
import time
import math
import threading

from asl_classifier import asl_siniflandir, asl_debug_bilgisi

# ─── Theme & Color Palette ───────────────────────────────────────────────────
COLORS = {
    "bg":               "#F4F6FB",
    "card_bg":          "#FFFFFF",
    "card_border":      "#E8ECF4",
    "primary":          "#6C63FF",
    "primary_hover":    "#5A52E0",
    "secondary":        "#3F8CFF",
    "accent_green":     "#34D399",
    "accent_red":       "#F87171",
    "text_dark":        "#1E1E2F",
    "text_mid":         "#5A5A78",
    "text_light":       "#9CA3AF",
    "camera_bg":        "#1A1A2E",
    "prediction_glow":  "#6C63FF",
    "header_bg":        "#1E1E2F",
    "status_active":    "#34D399",
    "status_inactive":  "#F87171",
    "button_stop_bg":   "#FEE2E2",
    "button_stop_fg":   "#DC2626",
    "confidence_bar":   "#6C63FF",
    "letter_history_bg":"#F0EEFF",
}

# ─── Yardımcı Fonksiyon: Köşeleri yuvarlatılmış PIL resmi ──────────────────────
def yuvarlatilmis_resim_olustur(w, h, YariCap, Dolgu_rengi, Kenarlik_rengi=None, Kenarlik_kalinligi=0):
    """Köşeleri yuvarlatılmış bir PIL.Image döndürür."""
    resim = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cizim = ImageDraw.Draw(resim)
    cizim.rounded_rectangle(
        [(0, 0), (w - 1, h - 1)],
        radius=YariCap,
        fill=Dolgu_rengi,
        outline=Kenarlik_rengi,
        width=Kenarlik_kalinligi,
    )
    return resim


def kamera_yer_tutucu_olustur(genislik, yukseklik):
    """El ikonu içeren koyu renkli bir kamera yer tutucu görseli oluşturur."""
    resim = Image.new("RGBA", (genislik, yukseklik), COLORS["camera_bg"])
    cizim = ImageDraw.Draw(resim)

    cx, cy = genislik // 2, yukseklik // 2

    # Daireler kullanarak stilize edilmiş bir el silueti çiz (Avuç + Parmaklar)
    avuc_yariCapi = 38
    parmak_pozisyonlari = [
        (cx - 30, cy - 65, 11),   # Serçe
        (cx - 12, cy - 80, 12),   # Yüzük
        (cx + 8,  cy - 85, 12),   # Orta
        (cx + 28, cy - 75, 11),   # İşaret
        (cx + 52, cy - 40, 11),   # Başparmak
    ]

    color = "#3A3A5C"
    # Avuç
    cizim.ellipse(
        [cx - avuc_yariCapi, cy - avuc_yariCapi, cx + avuc_yariCapi, cy + avuc_yariCapi],
        fill=color,
    )
    # Bilek
    cizim.rectangle(
        [cx - 25, cy + 10, cx + 25, cy + 55],
        fill=color,
    )
    # Yuvarlatılmış bilek altı
    cizim.ellipse(
        [cx - 25, cy + 35, cx + 25, cy + 70],
        fill=color,
    )
    # Parmaklar
    for fx, fy, fr in parmak_pozisyonlari:
        # Parmak gövdesi
        cizim.rounded_rectangle(
            [fx - fr, fy, fx + fr, cy - 10],
            radius=fr,
            fill=color,
        )
        # Parmak ucu
        cizim.ellipse(
            [fx - fr, fy - fr, fx + fr, fy + fr],
            fill=color,
        )

    # Bilgilendirme Metni
    try:
        font_kucuk = ImageFont.truetype("segoeui.ttf", 15)
    except OSError:
        font_kucuk = ImageFont.load_default()

    metin = "Kamera görüntüsü burada gösterilecek"
    kutu = cizim.textbbox((0, 0), metin, font=font_kucuk)
    metin_genisligi = kutu[2] - kutu[0]
    cizim.text(
        ((genislik - metin_genisligi) // 2, cy + 90),
        metin,
        fill="#5A5A78",
        font=font_kucuk,
    )

    # Kesikli çizgi stili dış çerçeve simülasyonu (hafif köşe işaretleri)
    isaret_uzunlugu = 30
    isaret_rengi = "#3F8CFF80"
    kalinlik = 2
    # Sol Üst
    cizim.line([(15, 15), (15 + isaret_uzunlugu, 15)], fill=isaret_rengi, width=kalinlik)
    cizim.line([(15, 15), (15, 15 + isaret_uzunlugu)], fill=isaret_rengi, width=kalinlik)
    # Sağ Üst
    cizim.line([(genislik - 15, 15), (genislik - 15 - isaret_uzunlugu, 15)], fill=isaret_rengi, width=kalinlik)
    cizim.line([(genislik - 15, 15), (genislik - 15, 15 + isaret_uzunlugu)], fill=isaret_rengi, width=kalinlik)
    # Sol Alt
    cizim.line([(15, yukseklik - 15), (15 + isaret_uzunlugu, yukseklik - 15)], fill=isaret_rengi, width=kalinlik)
    cizim.line([(15, yukseklik - 15), (15, yukseklik - 15 - isaret_uzunlugu)], fill=isaret_rengi, width=kalinlik)
    # Sağ Alt
    cizim.line([(genislik - 15, yukseklik - 15), (genislik - 15 - isaret_uzunlugu, yukseklik - 15)], fill=isaret_rengi, width=kalinlik)
    cizim.line([(genislik - 15, yukseklik - 15), (genislik - 15, yukseklik - 15 - isaret_uzunlugu)], fill=isaret_rengi, width=kalinlik)

    return resim


# ═══════════════════════════════════════════════════════════════════════════════
#  Ana Uygulama Penceresi
# ═══════════════════════════════════════════════════════════════════════════════
class HandSpeakApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ── Pencere Yapılandırması ────────────────────────────────────────────────
        self.title("HandSpeak — ASL Tanıma")
        self.geometry("1100x720")
        self.minsize(960, 640)
        self.configure(fg_color=COLORS["bg"])

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # ── Durum Değişkenleri ───────────────────────────────────────────────────
        self.calisiyor_mu = False
        self.guncel_harf = "—"
        self.guven_skoru = 0.0
        self.algilanan_harfler: list[str] = []
        self.kamera_aktif = False
        self._nabiz_acisi = 0
        self._animasyon_id = None
        self._kamera_dongu_id = None
        self.kamera = None  # OpenCV VideoCapture nesnesi
        self._fps_zamani = time.time()
        self._kare_sayisi = 0

        # ── MediaPipe El Tespiti ──────────────────────────────────────────────
        self.mp_eller = mp.solutions.hands
        self.mp_cizim = mp.solutions.drawing_utils
        self.mp_cizim_stilleri = mp.solutions.drawing_styles
        self.eller_modeli = None  # Başlangıçta inaktif

        # ── Tahmin Stabilizasyonu ─────────────────────────────────────────────
        self._tahmin_tamponu: list[str] = []
        self._tampon_boyutu = 8          # Biriktirmek için gereken kare sayısı
        self._son_kararli_harf = "?"
        self._elsiz_sayac = 0

        # ── Kapanışta Temizlik ────────────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._kapanista_temizle)

        # ── Build UI ─────────────────────────────────────────────────────
        self._build_header()
        self._build_body()
        self._build_footer()

    # ──────────────────────────────────────────────────────────────────────
    #  HEADER
    # ──────────────────────────────────────────────────────────────────────
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=COLORS["header_bg"], corner_radius=0, height=60)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # Logo / App name
        logo_frame = ctk.CTkFrame(header, fg_color="transparent")
        logo_frame.pack(side="left", padx=24, pady=10)

        # Icon placeholder (✋ emoji as quick icon)
        ctk.CTkLabel(
            logo_frame,
            text="🤟",
            font=ctk.CTkFont(size=28),
            text_color="#FFFFFF",
        ).pack(side="left", padx=(0, 8))

        ctk.CTkLabel(
            logo_frame,
            text="HandSpeak",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#FFFFFF",
        ).pack(side="left")

        ctk.CTkLabel(
            logo_frame,
            text="ASL Tanıma Sistemi",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLORS["text_light"],
        ).pack(side="left", padx=(12, 0))

        # Status indicator (right side)
        self.status_frame = ctk.CTkFrame(header, fg_color="transparent")
        self.status_frame.pack(side="right", padx=24)

        self.status_dot = ctk.CTkLabel(
            self.status_frame,
            text="●",
            font=ctk.CTkFont(size=14),
            text_color=COLORS["status_inactive"],
        )
        self.status_dot.pack(side="left", padx=(0, 6))

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Pasif",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLORS["text_light"],
        )
        self.status_label.pack(side="left")

    # ──────────────────────────────────────────────────────────────────────
    #  BODY  (camera + prediction panel)
    # ──────────────────────────────────────────────────────────────────────
    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=(16, 8))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # ── Camera card ──────────────────────────────────────────────────
        cam_card = ctk.CTkFrame(
            body,
            fg_color=COLORS["card_bg"],
            corner_radius=16,
            border_width=1,
            border_color=COLORS["card_border"],
        )
        cam_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        # Camera card header
        cam_header = ctk.CTkFrame(cam_card, fg_color="transparent", height=40)
        cam_header.pack(fill="x", padx=20, pady=(14, 0))
        cam_header.pack_propagate(False)

        ctk.CTkLabel(
            cam_header,
            text="📹  Kamera Görüntüsü",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=COLORS["text_dark"],
        ).pack(side="left")

        self.fps_label = ctk.CTkLabel(
            cam_header,
            text="0 FPS",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLORS["text_light"],
        )
        self.fps_label.pack(side="right")

        # Camera canvas — Canvas boyutu sabit kalır, içeriğe göre büyümez
        cam_container = ctk.CTkFrame(cam_card, fg_color=COLORS["camera_bg"], corner_radius=12)
        cam_container.pack(fill="both", expand=True, padx=16, pady=(10, 16))
        cam_container.pack_propagate(False)  # ← container'ın çocuğa göre büyümesini engelle

        self.camera_canvas = tk.Canvas(
            cam_container,
            bg=COLORS["camera_bg"],
            highlightthickness=0,
            bd=0,
        )
        self.camera_canvas.pack(fill="both", expand=True, padx=4, pady=4)

        # Alan boşken yer tutucu resmi sonraki çerçevede yerleştir (layout oturandan sonra)
        self.after(150, self._yer_tutucu_ayarla)

        # ── Prediction panel ─────────────────────────────────────────────
        pred_card = ctk.CTkFrame(
            body,
            fg_color=COLORS["card_bg"],
            corner_radius=16,
            border_width=1,
            border_color=COLORS["card_border"],
        )
        pred_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)

        # Prediction header
        pred_header = ctk.CTkFrame(pred_card, fg_color="transparent", height=40)
        pred_header.pack(fill="x", padx=20, pady=(14, 0))
        pred_header.pack_propagate(False)

        ctk.CTkLabel(
            pred_header,
            text="🔮  Tahmin",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=COLORS["text_dark"],
        ).pack(side="left")

        # Big letter display
        letter_container = ctk.CTkFrame(pred_card, fg_color="transparent")
        letter_container.pack(fill="x", padx=20, pady=(30, 10))

        self.letter_bg = ctk.CTkFrame(
            letter_container,
            fg_color=COLORS["letter_history_bg"],
            corner_radius=20,
            height=160,
            width=160,
        )
        self.letter_bg.pack(anchor="center")
        self.letter_bg.pack_propagate(False)

        self.letter_label = ctk.CTkLabel(
            self.letter_bg,
            text="—",
            font=ctk.CTkFont(family="Segoe UI", size=80, weight="bold"),
            text_color=COLORS["primary"],
        )
        self.letter_label.pack(expand=True)

        # Confidence section
        conf_frame = ctk.CTkFrame(pred_card, fg_color="transparent")
        conf_frame.pack(fill="x", padx=24, pady=(10, 0))

        ctk.CTkLabel(
            conf_frame,
            text="Güven Skoru",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLORS["text_light"],
        ).pack(anchor="w")

        self.confidence_bar = ctk.CTkProgressBar(
            conf_frame,
            progress_color=COLORS["confidence_bar"],
            fg_color=COLORS["card_border"],
            height=10,
            corner_radius=5,
        )
        self.confidence_bar.pack(fill="x", pady=(6, 4))
        self.confidence_bar.set(0)

        self.confidence_label = ctk.CTkLabel(
            conf_frame,
            text="—  %",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=COLORS["text_dark"],
        )
        self.confidence_label.pack(anchor="e")

        # Divider
        ctk.CTkFrame(pred_card, fg_color=COLORS["card_border"], height=1).pack(
            fill="x", padx=20, pady=(20, 12)
        )

        # Detected letters history
        hist_header = ctk.CTkFrame(pred_card, fg_color="transparent")
        hist_header.pack(fill="x", padx=20, pady=(0, 6))

        ctk.CTkLabel(
            hist_header,
            text="Algılanan Harfler",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLORS["text_dark"],
        ).pack(side="left")

        self.clear_btn = ctk.CTkButton(
            hist_header,
            text="Temizle",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="transparent",
            text_color=COLORS["primary"],
            hover_color=COLORS["letter_history_bg"],
            width=60,
            height=26,
            corner_radius=8,
            command=self._gecmisi_temizle,
        )
        self.clear_btn.pack(side="right")

        self.history_label = ctk.CTkLabel(
            pred_card,
            text="Henüz harf algılanmadı",
            font=ctk.CTkFont(family="Segoe UI", size=18),
            text_color=COLORS["text_light"],
            wraplength=220,
            justify="left",
        )
        self.history_label.pack(fill="x", padx=24, pady=(0, 20), anchor="w")

    # ──────────────────────────────────────────────────────────────────────
    #  FOOTER  (controls + instructions)
    # ──────────────────────────────────────────────────────────────────────
    def _build_footer(self):
        footer = ctk.CTkFrame(
            self,
            fg_color=COLORS["card_bg"],
            corner_radius=16,
            border_width=1,
            border_color=COLORS["card_border"],
            height=120,
        )
        footer.pack(fill="x", padx=20, pady=(8, 16))
        footer.pack_propagate(False)

        inner = ctk.CTkFrame(footer, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=24, pady=14)
        inner.grid_columnconfigure(0, weight=0)
        inner.grid_columnconfigure(1, weight=1)
        inner.grid_rowconfigure(0, weight=1)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(inner, fg_color="transparent")
        btn_frame.grid(row=0, column=0, sticky="w", padx=(0, 30))

        self.start_btn = ctk.CTkButton(
            btn_frame,
            text="▶  Başlat",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color="#FFFFFF",
            width=150,
            height=48,
            corner_radius=14,
            command=self._togle_baslat_buton,
        )
        self.start_btn.pack(side="left", padx=(0, 12))

        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="⏹  Durdur",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            fg_color=COLORS["button_stop_bg"],
            hover_color="#FDD",
            text_color=COLORS["button_stop_fg"],
            width=150,
            height=48,
            corner_radius=14,
            state="disabled",
            command=self._togle_durdur_buton,
        )
        self.stop_btn.pack(side="left")

        # ── Instructions ─────────────────────────────────────────────────
        info_frame = ctk.CTkFrame(
            inner,
            fg_color=COLORS["letter_history_bg"],
            corner_radius=12,
        )
        info_frame.grid(row=0, column=1, sticky="nsew")

        info_inner = ctk.CTkFrame(info_frame, fg_color="transparent")
        info_inner.pack(expand=True, padx=20, pady=10)

        ctk.CTkLabel(
            info_inner,
            text="💡  Kullanım Talimatları",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLORS["text_dark"],
        ).pack(anchor="w")

        instructions = (
            "1. \"Başlat\" düğmesine basarak kamerayı etkinleştirin.\n"
            "2. Elinizi kamera önünde konumlandırarak ASL işaretleri yapın.\n"
            "3. Tahmin edilen harf sağ panelde anlık olarak görüntülenir.\n"
            "4. İşlemi durdurmak için \"Durdur\" düğmesini kullanın."
        )
        ctk.CTkLabel(
            info_inner,
            text=instructions,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLORS["text_mid"],
            justify="left",
            anchor="w",
        ).pack(anchor="w", pady=(4, 0))

    # ──────────────────────────────────────────────────────────────────────
    #  KAMERA CANVAS YARDIMCI FONKSİYONLARI
    # ──────────────────────────────────────────────────────────────────────
    def _canvas_boyut(self):
        """Canvas'ın mevcut piksel boyutunu döndürür. Layout hazır değilse varsayılan verir."""
        self.update_idletasks()
        w = self.camera_canvas.winfo_width()
        h = self.camera_canvas.winfo_height()
        return max(w, 10), max(h, 10)

    def _canvas_resim_goster(self, pil_resim: Image.Image):
        """PIL görüntüsünü canvas boyutuna göre yeniden boyutlandırıp çizer."""
        w, h = self._canvas_boyut()
        pil_resim = pil_resim.resize((w, h), Image.LANCZOS)
        self._kamera_fotosu = ImageTk.PhotoImage(pil_resim)
        self.camera_canvas.delete("all")
        self.camera_canvas.create_image(0, 0, anchor="nw", image=self._kamera_fotosu)

    def _yer_tutucu_ayarla(self):
        """Kamera yer tutucu görüntüsünü canvas'a basar."""
        w, h = self._canvas_boyut()
        pil_resim = kamera_yer_tutucu_olustur(w, h)
        self._canvas_resim_goster(pil_resim)

    # ──────────────────────────────────────────────────────────────────────
    #  AKSİYONLAR
    # ──────────────────────────────────────────────────────────────────────
    def _togle_baslat_buton(self):
        # Kamerayı aç
        self.kamera = cv2.VideoCapture(0)
        if not self.kamera.isOpened():
            self.status_label.configure(text="Kamera bulunamadı!", text_color=COLORS["accent_red"])
            self.kamera = None
            return

        # MediaPipe Hands Modelini Başlat
        self.eller_modeli = self.mp_eller.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self._tahmin_tamponu.clear()
        self._elsiz_sayac = 0

        self.calisiyor_mu = True
        self.kamera_aktif = True
        self._fps_zamani = time.time()
        self._kare_sayisi = 0

        # Butonları güncelle
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        # Durumu güncelle
        self.status_dot.configure(text_color=COLORS["status_active"])
        self.status_label.configure(text="Aktif", text_color=COLORS["status_active"])

        # Kamera akışını başlat (gerçek el bulma modeli ile)
        self._kamera_karesini_guncelle()

    def _togle_durdur_buton(self):
        self.calisiyor_mu = False
        self.kamera_aktif = False

        # Kamerayı Serbest Bırak
        if self.kamera is not None:
            self.kamera.release()
            self.kamera = None

        # MediaPipe Modelini Serbest Bırak
        if self.eller_modeli is not None:
            self.eller_modeli.close()
            self.eller_modeli = None

        # Döngüleri İptal Et
        if self._kamera_dongu_id:
            self.after_cancel(self._kamera_dongu_id)
            self._kamera_dongu_id = None
        if self._animasyon_id:
            self.after_cancel(self._animasyon_id)
            self._animasyon_id = None

        # Butonları güncelle
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        # Durumu güncelle
        self.status_dot.configure(text_color=COLORS["status_inactive"])
        self.status_label.configure(text="Pasif", text_color=COLORS["text_light"])
        self.fps_label.configure(text="0 FPS")

        # Yer tutucu resmi geri getir
        self._yer_tutucu_ayarla()

    def _gecmisi_temizle(self):
        self.algilanan_harfler.clear()
        self.history_label.configure(
            text="Henüz harf algılanmadı", text_color=COLORS["text_light"]
        )

    # ──────────────────────────────────────────────────────────────────────
    #  CANLI KAMERA AKIŞI VE EL TESPİTİ
    # ──────────────────────────────────────────────────────────────────────
    def _kamera_karesini_guncelle(self):
        """Web kamerasından bir kare oku, elleri tespit et, ASL harfini sınıflandır ve görüntüle."""
        if not self.calisiyor_mu or self.kamera is None:
            return

        basarili_mi, cerceve = self.kamera.read()
        if basarili_mi:
            # Ayna efekti için görüntüyü yatay çevir
            cerceve = cv2.flip(cerceve, 1)
            # MediaPipe için BGR renk uzayını RGB'ye çevir
            cerceve_rgb = cv2.cvtColor(cerceve, cv2.COLOR_BGR2RGB)
            cerceve_rgb.flags.writeable = False

            # ── MediaPipe El Tespiti ──────────────────────────────
            sonuclar = self.eller_modeli.process(cerceve_rgb)
            cerceve_rgb.flags.writeable = True

            el_tespit_edildi = False

            if sonuclar.multi_hand_landmarks:
                for el_isaret_noktalari in sonuclar.multi_hand_landmarks:
                    el_tespit_edildi = True

                    # Ekrana iskelet modelini (landmarks) çiz
                    self.mp_cizim.draw_landmarks(
                        cerceve_rgb,
                        el_isaret_noktalari,
                        self.mp_eller.HAND_CONNECTIONS,
                        self.mp_cizim_stilleri.get_default_hand_landmarks_style(),
                        self.mp_cizim_stilleri.get_default_hand_connections_style(),
                    )

                    # ASL harfini sınıflandır
                    harf, guven_degeri = asl_siniflandir(el_isaret_noktalari.landmark)

                    # Debug overlay: parmak durumlarını kare üzerine çiz
                    debug = asl_debug_bilgisi(el_isaret_noktalari.landmark)
                    self._debug_overlay_ciz(cerceve_rgb, debug)

                    if harf != "?" and guven_degeri >= 0.65:
                        self._tahmin_tamponu.append(harf)
                        self._elsiz_sayac = 0

                        # Sinyali kararlaştır (Stabilizasyon): Sadece tampon dolduğunda güncelle
                        if len(self._tahmin_tamponu) >= self._tampon_boyutu:
                            # Tampondaki en yaygın harf
                            from collections import Counter
                            sayimlar = Counter(self._tahmin_tamponu)
                            kararli_harf, miktar = sayimlar.most_common(1)[0]
                            kararli_guven = miktar / len(self._tahmin_tamponu)

                            # Yalnızca karelerin yarısından fazlasında aynı harf çıkarsa ekrana bas
                            if kararli_guven >= 0.5 and kararli_harf != self._son_kararli_harf:
                                self._son_kararli_harf = kararli_harf
                                self._ekran_tahmin_guncelle(kararli_harf, guven_degeri)

                            self._tahmin_tamponu.clear()
                    else:
                        # El tespit edildi ama harf bilinmiyor veya güvenlik skoru düşük
                        self._elsiz_sayac = 0

            if not el_tespit_edildi:
                self._elsiz_sayac += 1
                self._tahmin_tamponu.clear()
                if self._elsiz_sayac > 15:  # ~0.5 saniye el yoksa ekranı temizle
                    self._elsiz_durumu_goster()

            # Görüntülenmesi için kareyi PIL resmine dönüştür ve canvas'a çiz
            pil_resmi = Image.fromarray(cerceve_rgb)
            self._canvas_resim_goster(pil_resmi)

            # FPS Sayacı 
            self._kare_sayisi += 1
            gecen_zaman = time.time() - self._fps_zamani
            if gecen_zaman >= 1.0:
                fps = self._kare_sayisi / gecen_zaman
                self.fps_label.configure(text=f"{fps:.0f} FPS")
                self._kare_sayisi = 0
                self._fps_zamani = time.time()

        # Sonraki kareyi çizilmesi için programla (~33ms ≈ 30 FPS)
        self._kamera_dongu_id = self.after(33, self._kamera_karesini_guncelle)

    @staticmethod
    def _debug_overlay_ciz(frame, debug):
        """Kamera karesine parmak durumlarını ve tahmini metin olarak çizer."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        olcek = 0.45
        kalinlik = 1
        golge = 2
        y = 22

        satirlar = [
            f"Isaret: {debug['parmaklar']['isaret']}",
            f"Orta:   {debug['parmaklar']['orta']}",
            f"Yuzuk:  {debug['parmaklar']['yuzuk']}",
            f"Serce:  {debug['parmaklar']['serce']}",
            f"Bas.P:  {debug['basparmak']}",
            f"Tahmin: {debug['tahmin']}  %{debug['guven'] * 100:.0f}",
        ]
        for metin in satirlar:
            cv2.putText(frame, metin, (8, y), font, olcek, (0, 0, 0), golge, cv2.LINE_AA)
            cv2.putText(frame, metin, (8, y), font, olcek, (255, 255, 255), kalinlik, cv2.LINE_AA)
            y += 18

    def _elsiz_durumu_goster(self):
        """Kamerada hiçbir el bulunmadığında arayüzü günceller."""
        self.letter_label.configure(text="✋", font=ctk.CTkFont(family="Segoe UI", size=60))
        self.confidence_bar.set(0)
        self.confidence_label.configure(text="El algılanmadı")
        self._son_kararli_harf = "?"

    def _ekran_tahmin_guncelle(self, harf: str, guven_degeri: float):
        """Tahmin panelini yeni tahmin edilen kelime ve güven puanı ile günceller."""
        self.guncel_harf = harf
        self.guven_skoru = guven_degeri

        # Harfi büyük şekilde yazdır
        self.letter_label.configure(text=harf)

        # Güvenliği / Skoru Yazdır
        self.confidence_bar.set(guven_degeri)
        self.confidence_label.configure(text=f"{guven_degeri * 100:.1f} %")

        # Güven oranına göre ilerleme çubuğunun rengini değiştir 
        if guven_degeri >= 0.9:
            self.confidence_bar.configure(progress_color=COLORS["accent_green"])
        elif guven_degeri >= 0.8:
            self.confidence_bar.configure(progress_color=COLORS["secondary"])
        else:
            self.confidence_bar.configure(progress_color=COLORS["primary"])

        # Geçmiş dizisini güncelle 
        self.algilanan_harfler.append(harf)
        # Sadece son 30 harfi ekranda tutmayı sağla 
        if len(self.algilanan_harfler) > 30:
            self.algilanan_harfler = self.algilanan_harfler[-30:]

        gosterilecek_metin = " ".join(self.algilanan_harfler)
        self.history_label.configure(text=gosterilecek_metin, text_color=COLORS["text_dark"])

        # Ekranda büyüyüp küçülmeli pop-up animasyonu tetikle
        self._nabiz_animasyonu_oynat()

    def _nabiz_animasyonu_oynat(self):
        """Tahmin edilen harf için belirip küçülme animasyonu."""
        boyutlar = [80, 90, 85, 80]
        gecikme = 60

        def animasyon_yurut(i=0):
            if i < len(boyutlar):
                self.letter_label.configure(
                    font=ctk.CTkFont(family="Segoe UI", size=boyutlar[i], weight="bold")
                )
                self.after(gecikme, lambda: animasyon_yurut(i + 1))

        animasyon_yurut()

    def _kapanista_temizle(self):
        """Uygulama tamamen kapandığında OpenCV kamerasını ve MediaPipe'i serbest bırakır."""
        self.calisiyor_mu = False
        if self.kamera is not None:
            self.kamera.release()
            self.kamera = None
        if self.eller_modeli is not None:
            self.eller_modeli.close()
            self.eller_modeli = None
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
#  Program Başlangıcı 
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uygulama = HandSpeakApp()
    uygulama.mainloop()
