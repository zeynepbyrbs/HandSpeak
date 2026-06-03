import time
from collections import Counter

import cv2
import customtkinter as ctk
import mediapipe as mp
import tkinter as tk

from PIL import Image, ImageDraw, ImageFont, ImageTk

from asl_classifier import (
    classify_asl as asl_siniflandir,
)

# =========================================================
# THEME
# =========================================================

COLORS = {
    "bg": "#F4F6FB",
    "card_bg": "#FFFFFF",
    "card_border": "#E8ECF4",
    "primary": "#6C63FF",
    "primary_hover": "#5A52E0",
    "secondary": "#3F8CFF",
    "accent_green": "#34D399",
    "accent_red": "#F87171",
    "text_dark": "#1E1E2F",
    "text_mid": "#5A5A78",
    "text_light": "#9CA3AF",
    "camera_bg": "#1A1A2E",
    "header_bg": "#1E1E2F",
    "status_active": "#34D399",
    "status_inactive": "#F87171",
    "button_stop_bg": "#FEE2E2",
    "button_stop_fg": "#DC2626",
    "confidence_bar": "#6C63FF",
    "letter_history_bg": "#F0EEFF",
}

# =========================================================
# PLACEHOLDER
# =========================================================

def kamera_placeholder(width, height):

    img = Image.new("RGB", (width, height), COLORS["camera_bg"])
    draw = ImageDraw.Draw(img)

    text = "Kamera görüntüsü burada gösterilecek"

    try:
        font = ImageFont.truetype("segoeui.ttf", 20)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw.text(
        (
            (width - tw) // 2,
            (height - th) // 2
        ),
        text,
        fill="#8080A0",
        font=font
    )

    return img


# =========================================================
# MAIN APP
# =========================================================

class HandSpeakApp(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title("HandSpeak — ASL Recognition")
        self.geometry("1150x740")
        self.minsize(980, 680)

        ctk.set_appearance_mode("light")

        self.configure(fg_color=COLORS["bg"])

        # =================================================
        # STATE
        # =================================================

        self.running = False

        self.camera = None
        self.hands = None

        self.current_letter = "—"
        self.current_confidence = 0.0

        self.history = []

        self.prediction_buffer = []
        self.buffer_size = 8

        self.last_stable_letter = "?"

        self.no_hand_counter = 0

        self.frame_counter = 0
        self.fps_time = time.time()

        # =================================================
        # MEDIAPIPE
        # =================================================

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # =================================================
        # UI
        # =================================================

        self.build_ui()

        self.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

    # =====================================================
    # UI
    # =====================================================

    def build_ui(self):

        self.build_header()
        self.build_body()
        self.build_footer()

    def build_header(self):

        header = ctk.CTkFrame(
            self,
            height=60,
            corner_radius=0,
            fg_color=COLORS["header_bg"]
        )

        header.pack(fill="x")

        ctk.CTkLabel(
            header,
            text="🤟 HandSpeak",
            font=ctk.CTkFont(
                size=26,
                weight="bold"
            ),
            text_color="white"
        ).pack(side="left", padx=20)

        self.status_label = ctk.CTkLabel(
            header,
            text="Pasif",
            text_color=COLORS["status_inactive"],
            font=ctk.CTkFont(size=14)
        )

        self.status_label.pack(
            side="right",
            padx=20
        )

    def build_body(self):

        body = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        body.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=1)

        # =================================================
        # CAMERA PANEL
        # =================================================

        left = ctk.CTkFrame(
            body,
            fg_color=COLORS["card_bg"]
        )

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        self.camera_canvas = tk.Canvas(
            left,
            bg=COLORS["camera_bg"],
            highlightthickness=0
        )

        self.camera_canvas.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # =================================================
        # RIGHT PANEL
        # =================================================

        right = ctk.CTkFrame(
            body,
            fg_color=COLORS["card_bg"]
        )

        right.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        self.letter_label = ctk.CTkLabel(
            right,
            text="—",
            font=ctk.CTkFont(
                size=120,
                weight="bold"
            ),
            text_color=COLORS["primary"]
        )

        self.letter_label.pack(pady=40)

        self.confidence_bar = ctk.CTkProgressBar(
            right,
            height=12
        )

        self.confidence_bar.pack(
            fill="x",
            padx=20
        )

        self.confidence_bar.set(0)

        self.confidence_label = ctk.CTkLabel(
            right,
            text="0%"
        )

        self.confidence_label.pack(pady=10)

        self.history_label = ctk.CTkLabel(
            right,
            text="Henüz harf algılanmadı",
            wraplength=250,
            justify="left"
        )

        self.history_label.pack(
            fill="x",
            padx=20,
            pady=20
        )

        self.show_placeholder()

    def build_footer(self):

        footer = ctk.CTkFrame(
            self,
            height=90,
            fg_color=COLORS["card_bg"]
        )

        footer.pack(
            fill="x",
            padx=20,
            pady=(0, 20)
        )

        self.start_btn = ctk.CTkButton(
            footer,
            text="▶ Başlat",
            command=self.start_camera,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            width=160,
            height=45
        )

        self.start_btn.pack(
            side="left",
            padx=20,
            pady=20
        )

        self.stop_btn = ctk.CTkButton(
            footer,
            text="⏹ Durdur",
            command=self.stop_camera,
            fg_color=COLORS["button_stop_bg"],
            text_color=COLORS["button_stop_fg"],
            width=160,
            height=45,
            state="disabled"
        )

        self.stop_btn.pack(
            side="left"
        )

    # =====================================================
    # CAMERA
    # =====================================================

    def start_camera(self):

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            self.status_label.configure(
                text="Kamera bulunamadı",
                text_color=COLORS["accent_red"]
            )

            return

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        self.running = True

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        self.status_label.configure(
            text="Aktif",
            text_color=COLORS["status_active"]
        )

        self.update_frame()

    def stop_camera(self):

        self.running = False

        if self.camera:
            self.camera.release()
            self.camera = None

        if self.hands:
            self.hands.close()
            self.hands = None

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        self.status_label.configure(
            text="Pasif",
            text_color=COLORS["status_inactive"]
        )

        self.show_placeholder()

    # =====================================================
    # FRAME LOOP
    # =====================================================

    def update_frame(self):

        if not self.running:
            return

        success, frame = self.camera.read()

        if success:

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = self.hands.process(rgb)

            hand_detected = False

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    hand_detected = True

                    self.mp_draw.draw_landmarks(
                        rgb,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style(),
                    )

                    prediction, confidence = asl_siniflandir(
                        hand_landmarks.landmark
                    )

                    debug = {
                        "tahmin": prediction,
                        "guven": confidence
                    }

                    self.draw_debug(
                        rgb,
                        debug
                    )

                    self.process_prediction(
                        prediction,
                        confidence
                    )

            if not hand_detected:

                self.no_hand_counter += 1

                if self.no_hand_counter > 15:
                    self.reset_prediction()

            self.show_frame(rgb)

            self.update_fps()

        self.after(
            30,
            self.update_frame
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    def process_prediction(self, prediction, confidence):

        if prediction == "?" or confidence < 0.65:
            return

        self.prediction_buffer.append(prediction)

        if len(self.prediction_buffer) >= self.buffer_size:

            counts = Counter(
                self.prediction_buffer
            )

            stable_letter, amount = counts.most_common(1)[0]

            stability = amount / len(self.prediction_buffer)

            if (
                stability >= 0.5 and
                stable_letter != self.last_stable_letter
            ):

                self.last_stable_letter = stable_letter

                self.update_prediction_ui(
                    stable_letter,
                    confidence
                )

            self.prediction_buffer.clear()

    # =====================================================
    # UI UPDATE
    # =====================================================

    def update_prediction_ui(self, letter, confidence):

        self.letter_label.configure(
            text=letter
        )

        self.confidence_bar.set(confidence)

        self.confidence_label.configure(
            text=f"{confidence * 100:.1f}%"
        )

        self.history.append(letter)

        if len(self.history) > 30:
            self.history = self.history[-30:]

        self.history_label.configure(
            text=" ".join(self.history)
        )

    def reset_prediction(self):

        self.letter_label.configure(
            text="✋"
        )

        self.confidence_bar.set(0)

        self.confidence_label.configure(
            text="El algılanmadı"
        )

        self.last_stable_letter = "?"

    # =====================================================
    # DEBUG
    # =====================================================

    @staticmethod
    def draw_debug(frame, debug):

        y = 25

        lines = [
            f"Prediction: {debug['tahmin']}",
            f"Confidence: %{debug['guven'] * 100:.0f}",
        ]

        for line in lines:

            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            y += 25

    # =====================================================
    # DISPLAY
    # =====================================================

    def show_frame(self, frame):

        if isinstance(frame, Image.Image):
            img = frame
        else:
            img = Image.fromarray(frame)

        canvas_w = max(
            self.camera_canvas.winfo_width(),
            10
        )

        canvas_h = max(
            self.camera_canvas.winfo_height(),
            10
        )

        img = img.resize(
            (canvas_w, canvas_h)
        )

        self.photo = ImageTk.PhotoImage(img)

        self.camera_canvas.delete("all")

        self.camera_canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.photo
        )

    def show_placeholder(self):

        w = max(
            self.camera_canvas.winfo_width(),
            640
        )

        h = max(
            self.camera_canvas.winfo_height(),
            480
        )

        img = kamera_placeholder(w, h)

        self.show_frame(
            cv2.cvtColor(
                cv2.imread(""),
                cv2.COLOR_BGR2RGB
            ) if False else img
        )

    # =====================================================
    # FPS
    # =====================================================

    def update_fps(self):

        self.frame_counter += 1

        elapsed = time.time() - self.fps_time

        if elapsed >= 1:

            fps = self.frame_counter / elapsed

            self.title(
                f"HandSpeak — {fps:.0f} FPS"
            )

            self.frame_counter = 0
            self.fps_time = time.time()

    # =====================================================
    # CLOSE
    # =====================================================

    def on_close(self):

        self.stop_camera()

        self.destroy()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    app = HandSpeakApp()

    app.mainloop()