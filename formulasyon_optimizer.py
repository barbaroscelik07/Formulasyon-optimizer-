"""
Formulasyon-Optimizer v1
Farmasötik İnhaler Formülasyon Geliştirme & Optimizasyon Aracı
PyQt6 + Matplotlib + scipy + statsmodels + pyDOE3
ADIM 1: Ana iskelet, tema, veri modeli, Sekme 1 (Deney Tasarımı)
"""
import sys, os, json, math, datetime, itertools
import numpy as np
import pandas as pd
from scipy import stats

import pyDOE3

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox,
    QTabWidget, QScrollArea, QFrame, QFileDialog,
    QMessageBox, QDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QSpacerItem, QGroupBox,
    QDialogButtonBox, QAbstractItemView, QSpinBox,
    QDoubleSpinBox, QTextEdit, QRadioButton, QButtonGroup,
    QStatusBar
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon

# ─── resource_path ─────────────────────────────────────────────────────────────
def resource_path(rel):
    base = getattr(sys, '_MEIPASS',
        os.path.dirname(os.path.abspath(
            sys.executable if getattr(sys, 'frozen', False) else __file__)))
    return os.path.join(base, rel)

# ═══════════════════════════════════════════════════════════════════════════════
# RENKLER & STİL
# ═══════════════════════════════════════════════════════════════════════════════
BG    = "#0e1219"
BG2   = "#141824"
BG3   = "#1c2336"
NAVY  = "#002D62"
GOLD  = "#FFC600"
TXT   = "#e0eaf8"
TXT2  = "#7090b0"
GREEN = "#00B050"
RED   = "#C00000"
TEAL  = "#0a8a6a"
CP    = ["#2E75B6","#ED7D31","#70AD47","#E84040","#7030A0",
         "#00B0F0","#D4A000","#C00000","#00B050","#FF69B4"]

RESPONSE_LABELS = {
    "mmad":      "MMAD (µm)",
    "gsd":       "GSD",
    "fpd_5um":   "FPD <5µm (mg)",
    "fpf_5um":   "FPF <5µm (%)",
    "fpd_3um":   "FPD <3µm (mg)",
    "fpf_3um":   "FPF <3µm (%)",
    "fpd_15um":  "FPD <1.5µm (mg)",
    "fpf_15um":  "FPF <1.5µm (%)",
    "metered":   "Metered Doz (mg)",
    "delivered": "Delivered Doz (mg)",
}

DESIGN_TYPES = [
    "Full Factorial (2k)",
    "Fractional Factorial (2k-p)",
    "Central Composite (CCD/RSM)",
    "Box-Behnken (BBD)",
    "Plackett-Burman",
    "One Factor at a Time (OFAT)",
]

DESIGN_INFO = {
    "Full Factorial (2k)": {
        "aciklama": "Tüm faktörlerin tüm kombinasyonlarını test eder. 2 seviyeli (düşük/yüksek).",
        "ne_zaman": "Faktör sayısı ≤4 ve tüm etkileşimler önemliyse.",
        "avantaj":  "Tüm ana etkiler + etkileşimler hesaplanır. Sonuçlar kesin.",
        "dezavantaj": "Faktör sayısı arttıkça run sayısı katlanır (2⁵=32, 2⁶=64...).",
        "oneri":    "2-3 faktör için ideal başlangıç tasarımı.",
        "run_formula": lambda k: 2**k,
        "seviye": 2,
    },
    "Fractional Factorial (2k-p)": {
        "aciklama": "Full Factorial'ın bir kesriyle ana etkileri belirler.",
        "ne_zaman": "5+ faktörü hızlıca taramak istediğinde, hangilerinin önemli olduğu bilinmiyorsa.",
        "avantaj":  "Full Factorial'a göre çok daha az run. Yüksek faktör sayısında kullanışlı.",
        "dezavantaj": "Bazı etkileşimler hesaplanamaz (aliasing). Tarama için uygundur.",
        "oneri":    "İlk tarama → anlamlı faktörleri bul → CCD/BBD ile optimize et.",
        "run_formula": lambda k: max(8, 2**(k-1)),
        "seviye": 2,
    },
    "Central Composite (CCD/RSM)": {
        "aciklama": "Faktör sınırlarının dışına çıkan eksenel noktalar içeren 5-seviyeli tasarım.",
        "ne_zaman": "2-4 faktörün optimumunu bulmak istediğinde, eğrisel ilişki bekliyorsan.",
        "avantaj":  "5 seviye ile quadratic model kurulur. Gerçek optimumu yakalayabilir.",
        "dezavantaj": "α noktaları faktör sınırlarının dışına çıkabilir.",
        "oneri":    "Tarama sonrası anlamlı 2-3 faktörle kullan. En güçlü optimizasyon tasarımı.",
        "run_formula": lambda k: 2**k + 2*k + 4,
        "seviye": 5,
    },
    "Box-Behnken (BBD)": {
        "aciklama": "Faktör sınırları içinde kalan, köşe noktası olmayan 3-seviyeli tasarım.",
        "ne_zaman": "3-4 faktörün optimumu, faktör sınırlarının dışına çıkmak istemiyorsan.",
        "avantaj":  "CCD'ye göre az run, köşe noktaları yok — aşırı kombinasyonları test etmez.",
        "dezavantaj": "En az 3 faktör gerektirir. 2 faktörle kullanılamaz.",
        "oneri":    "Formülasyon kısıtı olan çalışmalarda CCD'ye tercih edilir.",
        "run_formula": lambda k: 2*k*(k-1) + 3 if k >= 3 else 0,
        "seviye": 3,
    },
    "Plackett-Burman": {
        "aciklama": "Minimum run sayısıyla maksimum faktörü tarayan 2-seviyeli tarama tasarımı.",
        "ne_zaman": "5+ faktörü minimum run ile taramak istediğinde.",
        "avantaj":  "Çok az run (N = 4'ün katı). 11 faktörü 12 run'da tarayabilirsin.",
        "dezavantaj": "Sadece ana etkiler hesaplanır, etkileşimler hesaplanamaz.",
        "oneri":    "İlk tarama çalışması için en verimli tasarım.",
        "run_formula": lambda k: max(8, ((k + 1) // 4 + 1) * 4),
        "seviye": 2,
    },
    "One Factor at a Time (OFAT)": {
        "aciklama": "Her seferinde tek faktörü değiştirir, diğerleri sabit kalır.",
        "ne_zaman": "Referans veya tek faktörü izole etmek istediğinde.",
        "avantaj":  "Anlaşılması ve uygulanması kolay.",
        "dezavantaj": "Faktörler arası etkileşimleri göremez.",
        "oneri":    "Karşılaştırma amaçlı kullan. Optimizasyon için tercih etme.",
        "run_formula": lambda k: 2*k + 1,
        "seviye": 2,
    },
}

STYLE = f"""
QMainWindow, QDialog {{
    background: {BG};
}}
QWidget {{
    background: {BG};
    color: {TXT};
    font-family: 'Segoe UI', 'Arial';
    font-size: 13px;
}}
QLabel {{ color: {TXT}; background: transparent; }}
QLineEdit, QDoubleSpinBox, QSpinBox, QTextEdit {{
    background: {BG3};
    border: 1px solid #2a4060;
    border-radius: 4px;
    padding: 3px 6px;
    color: {TXT};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{ border: 1px solid {GOLD}; }}
QPushButton {{
    background: {BG3};
    border: 1px solid #2a4060;
    border-radius: 5px;
    padding: 5px 12px;
    color: {TXT};
    font-weight: 500;
}}
QPushButton:hover {{ background: #253a5e; border-color: {GOLD}; }}
QPushButton:pressed {{ background: {BG2}; }}
QPushButton:disabled {{ color: #3a5070; border-color: #1a2a40; }}
QTabWidget::pane {{
    border: 1px solid #2a4060;
    background: {BG2};
    border-radius: 4px;
}}
QTabBar::tab {{
    background: {BG3};
    color: {TXT2};
    border: 1px solid #2a4060;
    padding: 7px 18px;
    border-bottom: none;
    margin-right: 2px;
    border-radius: 4px 4px 0 0;
    font-size: 12px;
}}
QTabBar::tab:selected {{
    background: {BG2};
    color: {GOLD};
    border-color: {GOLD};
    font-weight: bold;
}}
QTabBar::tab:hover:!selected {{ background: #253a5e; color: {TXT}; }}
QComboBox {{
    background: {BG3};
    border: 1px solid #2a4060;
    border-radius: 4px;
    padding: 4px 8px;
    color: {TXT};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {BG2};
    color: {TXT};
    selection-background-color: #1F4E79;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {BG2}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: #2a4060; border-radius: 4px; min-height: 20px;
}}
QScrollBar:horizontal {{
    background: {BG2}; height: 8px; border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: #2a4060; border-radius: 4px; min-width: 20px;
}}
QTableWidget {{
    background: {BG2};
    gridline-color: #2a4060;
    color: {TXT};
    border: 1px solid #2a4060;
    border-radius: 4px;
}}
QTableWidget::item {{ padding: 4px; }}
QTableWidget::item:selected {{ background: #1F4E79; }}
QHeaderView::section {{
    background: #1F4E79;
    color: white;
    padding: 5px;
    border: 1px solid #2a4060;
    font-weight: bold;
    font-size: 12px;
}}
QGroupBox {{
    border: 1px solid #2a4060;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    color: {TXT2};
    font-size: 11px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {GOLD};
}}
QCheckBox {{ color: {TXT}; background: transparent; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid #2a4060;
    border-radius: 3px;
    background: {BG3};
}}
QCheckBox::indicator:checked {{
    background: #2E75B6;
    border-color: #4a95d6;
}}
QRadioButton {{ color: {TXT}; background: transparent; spacing: 6px; }}
QRadioButton::indicator {{
    width: 14px; height: 14px;
    border: 1px solid #2a4060;
    border-radius: 7px;
    background: {BG3};
}}
QRadioButton::indicator:checked {{
    background: {GOLD};
    border-color: {GOLD};
}}
QStatusBar {{
    background: {BG2};
    color: {TXT2};
    border-top: 1px solid #2a4060;
    font-size: 11px;
}}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════
def make_btn(text, color=None, height=32):
    b = QPushButton(text)
    b.setFixedHeight(height)
    c = color or "rgba(30,50,90,0.8)"
    b.setStyleSheet(f"""
        QPushButton {{
            background: {c};
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 5px;
            padding: 4px 14px;
            color: {TXT};
            font-weight: 500;
        }}
        QPushButton:hover {{ background: rgba(255,255,255,0.08); border-color: {GOLD}; }}
        QPushButton:pressed {{ background: rgba(0,0,0,0.2); }}
        QPushButton:disabled {{ color: #3a5070; border-color: #1a2a40; background: rgba(20,30,50,0.5); }}
    """)
    return b

def section_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {GOLD}; font-size: 11px; font-weight: bold; background: transparent;")
    return lbl

def card_frame():
    f = QFrame()
    f.setStyleSheet(f"""
        QFrame {{
            background: {BG2};
            border: 1px solid #2a4060;
            border-radius: 8px;
        }}
        QLabel {{ background: transparent; }}
    """)
    return f

def info_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {TXT2}; font-size: 11px; background: transparent;")
    lbl.setWordWrap(True)
    return lbl

def make_help_btn(tooltip_text, parent=None):
    btn = QPushButton("?")
    btn.setFixedSize(18, 18)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: rgba(30,60,100,0.8);
            border: 1px solid #4a7ab0;
            border-radius: 9px;
            color: #90c0f0;
            font-size: 10px;
            font-weight: bold;
            padding: 0px;
        }}
        QPushButton:hover {{
            background: rgba(50,100,160,0.9);
            border-color: {GOLD};
            color: {GOLD};
        }}
    """)
    btn.clicked.connect(lambda: QMessageBox.information(parent, "Açıklama", tooltip_text))
    return btn

def tr2ascii(text):
    replacements = {
        'ç':'c','ğ':'g','ı':'i','ö':'o','ş':'s','ü':'u',
        'Ç':'C','Ğ':'G','İ':'I','Ö':'O','Ş':'S','Ü':'U',
    }
    result = str(text)
    for tr_char, ascii_char in replacements.items():
        result = result.replace(tr_char, ascii_char)
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# VERİ MODELİ — Tüm proje verisi burada tutulur
# ═══════════════════════════════════════════════════════════════════════════════
class OptimizerProject:
    """Tüm sekmelerin paylaştığı merkezi veri modeli."""
    def __init__(self):
        self.reset()

    def reset(self):
        # Faktörler: [{"name": str, "unit": str, "low": float, "mid": float, "high": float, "type": "continuous"}]
        self.factors        = []
        # Seçili yanıtlar: ["mmad", "fpf_5um", ...]
        self.responses      = []
        # Tasarım tipi
        self.design_type    = "Central Composite (CCD/RSM)"
        # Tasarım matrisi (DataFrame) — faktör değerleri
        self.design_matrix  = None
        # Run sonuçları: {run_idx: {"mmad": float, "fpf_5um": float, ...}}
        self.run_results    = {}
        # Model sonuçları: {resp_key: statsmodels_model}
        self.model_results  = {}
        self.model_errors   = {}
        # Optimizasyon sonucu
        self.opt_solutions  = []   # Top-5 liste
        # Spesifikasyon sınırları: {"mmad": {"lsl": float, "usl": float, "goal": str, "weight": float}}
        self.spec_limits    = {}
        # Doğrulama sonuçları
        self.validation_results = {}

    def get_safe_names(self):
        """Faktör isimlerini istatistiksel model için güvenli hale getirir."""
        safe = []
        for i, f in enumerate(self.factors):
            s = f["name"].replace(" ", "_").replace("-", "_")
            s = "".join(c if c.isalnum() or c == "_" else "_" for c in s)
            if not s or s[0].isdigit():
                s = f"F{i+1}_{s}"
            safe.append(s)
        return safe

    def get_coded_df(self):
        """Tasarım matrisini kodlanmış değerlerle DataFrame olarak döner."""
        if self.design_matrix is None:
            return None, None
        safe_names = self.get_safe_names()
        df = pd.DataFrame()
        for i, f in enumerate(self.factors):
            col_name = f["name"]
            if col_name not in self.design_matrix.columns:
                continue
            vals = self.design_matrix[col_name].values
            lo, hi = f["low"], f["high"]
            mid = f.get("mid", (lo + hi) / 2)
            coded = []
            for v in vals:
                if hi == lo:
                    coded.append(0.0)
                elif v <= mid and (mid - lo) != 0:
                    coded.append((v - lo) / (mid - lo) - 1)
                elif (hi - mid) != 0:
                    coded.append((v - mid) / (hi - mid))
                else:
                    coded.append(0.0)
            df[safe_names[i]] = coded
        return df, safe_names

    def build_response_table(self):
        """Run sonuçlarını DataFrame olarak döner."""
        if self.design_matrix is None or not self.run_results:
            return None
        rows = []
        for i in range(len(self.design_matrix)):
            row = dict(self.run_results.get(i, {}))
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_design_matrix(self):
        """Seçilen tasarım tipine göre matris oluşturur."""
        k = len(self.factors)
        if k == 0:
            return None, "En az 1 faktör ekleyin."

        design_type = self.design_type
        info = DESIGN_INFO.get(design_type, {})

        try:
            if design_type == "Full Factorial (2k)":
                coded = pyDOE3.ff2n(k)

            elif design_type == "Fractional Factorial (2k-p)":
                coded = pyDOE3.fracfact(" ".join([chr(97+i) for i in range(k)]))

            elif design_type == "Central Composite (CCD/RSM)":
                coded = pyDOE3.ccdesign(k, center=(4, 4), face="circumscribed")

            elif design_type == "Box-Behnken (BBD)":
                if k < 3:
                    return None, "Box-Behnken en az 3 faktör gerektirir."
                coded = pyDOE3.bbdesign(k, center=3)

            elif design_type == "Plackett-Burman":
                n_runs = max(8, ((k + 1) // 4 + 1) * 4)
                coded = pyDOE3.pbdesign(n_runs)
                coded = coded[:, :k]

            elif design_type == "One Factor at a Time (OFAT)":
                rows = []
                center = np.zeros(k)
                rows.append(center.copy())
                for i in range(k):
                    low_row = center.copy(); low_row[i] = -1; rows.append(low_row)
                    high_row = center.copy(); high_row[i] = 1; rows.append(high_row)
                coded = np.array(rows)

            else:
                return None, f"Bilinmeyen tasarım tipi: {design_type}"

        except Exception as e:
            return None, f"Matris oluşturma hatası: {e}"

        # Kodlanmış → gerçek değerlere çevir
        df = pd.DataFrame()
        for i, f in enumerate(self.factors):
            lo, hi = f["low"], f["high"]
            mid = f.get("mid", (lo + hi) / 2)
            col = coded[:, i]
            real = []
            for v in col:
                if v < 0:
                    real.append(lo + (v + 1) * (mid - lo))
                elif v > 0:
                    real.append(mid + v * (hi - mid))
                else:
                    real.append(mid)
            df[f["name"]] = [round(x, 6) for x in real]

        self.design_matrix = df
        self.run_results = {}
        self.model_results = {}
        return df, None

# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 1 — DENEY TASARIMI
# ═══════════════════════════════════════════════════════════════════════════════
class FactorRow(QWidget):
    """Tek bir faktör satırı — isim, birim, düşük, orta, yüksek."""
    deleted = pyqtSignal(object)

    def __init__(self, idx, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(6)

        # Sıra numarası
        num = QLabel(f"{idx+1}.")
        num.setFixedWidth(22)
        num.setStyleSheet(f"color: {TXT2}; font-size: 11px; background: transparent;")
        lay.addWidget(num)

        # İsim
        self.name_edit = QLineEdit(f"Faktör {idx+1}")
        self.name_edit.setFixedWidth(140)
        self.name_edit.setPlaceholderText("Faktör adı")
        lay.addWidget(self.name_edit)

        # Birim
        self.unit_edit = QLineEdit("")
        self.unit_edit.setFixedWidth(70)
        self.unit_edit.setPlaceholderText("Birim")
        lay.addWidget(self.unit_edit)

        # Düşük
        lbl_lo = QLabel("Alt:")
        lbl_lo.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        lbl_lo.setFixedWidth(26)
        lay.addWidget(lbl_lo)
        self.low_sp = QDoubleSpinBox()
        self.low_sp.setRange(-1e9, 1e9); self.low_sp.setValue(0); self.low_sp.setDecimals(4)
        self.low_sp.setFixedWidth(90)
        lay.addWidget(self.low_sp)

        # Orta
        lbl_mid = QLabel("Orta:")
        lbl_mid.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        lbl_mid.setFixedWidth(36)
        lay.addWidget(lbl_mid)
        self.mid_sp = QDoubleSpinBox()
        self.mid_sp.setRange(-1e9, 1e9); self.mid_sp.setValue(0.5); self.mid_sp.setDecimals(4)
        self.mid_sp.setFixedWidth(90)
        lay.addWidget(self.mid_sp)

        # Yüksek
        lbl_hi = QLabel("Üst:")
        lbl_hi.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        lbl_hi.setFixedWidth(30)
        lay.addWidget(lbl_hi)
        self.high_sp = QDoubleSpinBox()
        self.high_sp.setRange(-1e9, 1e9); self.high_sp.setValue(1); self.high_sp.setDecimals(4)
        self.high_sp.setFixedWidth(90)
        lay.addWidget(self.high_sp)

        lay.addStretch()

        # Sil butonu
        btn_del = QPushButton("✕")
        btn_del.setFixedSize(24, 24)
        btn_del.setStyleSheet(f"""
            QPushButton {{
                background: rgba(160,30,30,0.5);
                border: 1px solid #5a1010;
                border-radius: 12px;
                color: #ff8080;
                font-size: 11px;
            }}
            QPushButton:hover {{ background: rgba(200,40,40,0.8); color: white; }}
        """)
        btn_del.clicked.connect(lambda: self.deleted.emit(self))
        lay.addWidget(btn_del)

    def get_factor(self):
        return {
            "name": self.name_edit.text().strip() or f"Faktör {self.idx+1}",
            "unit": self.unit_edit.text().strip(),
            "low":  self.low_sp.value(),
            "mid":  self.mid_sp.value(),
            "high": self.high_sp.value(),
            "type": "continuous",
        }


class Tab1_Design(QWidget):
    """Sekme 1 — Deney Tasarımı: faktör girişi + tasarım seçimi + matris önizleme."""
    design_generated = pyqtSignal()   # Diğer sekmeleri uyarmak için

    def __init__(self, project: OptimizerProject, app_ref, parent=None):
        super().__init__(parent)
        self.project  = project
        self.app      = app_ref
        self._factor_rows = []
        self._build()

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # ── Sol panel: Faktörler + Tasarım tipi ──────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(10)

        # — Faktörler kartı —
        fcard = card_frame()
        fl = QVBoxLayout(fcard)
        fl.setContentsMargins(14, 12, 14, 12)
        fl.setSpacing(8)

        fhdr = QHBoxLayout()
        fhdr.addWidget(section_label("📋  Faktörler"))
        fhdr.addWidget(make_help_btn(
            "Her faktör için:\n"
            "• İsim: Faktörün adı (örn. Oleik Asit)\n"
            "• Birim: Ölçü birimi (örn. %)\n"
            "• Alt: Minimum test seviyesi\n"
            "• Orta: Merkez nokta (CCD/BBD için önemli)\n"
            "• Üst: Maksimum test seviyesi\n\n"
            "En az 2 faktör girilmesi önerilir.", self))
        fhdr.addStretch()
        btn_add = make_btn("+ Faktör Ekle", "rgba(20,70,20,0.8)", 28)
        btn_add.clicked.connect(self._add_factor)
        fhdr.addWidget(btn_add)
        fl.addLayout(fhdr)

        # Sütun başlıkları
        hdr_w = QWidget(); hdr_w.setStyleSheet("background:transparent;")
        hdr_l = QHBoxLayout(hdr_w)
        hdr_l.setContentsMargins(28, 0, 32, 0); hdr_l.setSpacing(6)
        for txt, w in [("İsim", 140), ("Birim", 70), ("", 26),
                       ("Alt Seviye", 90), ("", 36), ("Orta Seviye", 90),
                       ("", 30), ("Üst Seviye", 90)]:
            l = QLabel(txt)
            l.setFixedWidth(w)
            l.setStyleSheet(f"color:{GOLD}; font-size:10px; font-weight:bold; background:transparent;")
            hdr_l.addWidget(l)
        hdr_l.addStretch()
        fl.addWidget(hdr_w)

        # Faktör satırları scroll alanı
        self.factor_scroll = QScrollArea()
        self.factor_scroll.setWidgetResizable(True)
        self.factor_scroll.setFixedHeight(220)
        self.factor_scroll.setStyleSheet("border: none; background: transparent;")
        self.factor_container = QWidget()
        self.factor_container.setStyleSheet("background: transparent;")
        self.factor_layout = QVBoxLayout(self.factor_container)
        self.factor_layout.setSpacing(2)
        self.factor_layout.setContentsMargins(0, 0, 0, 0)
        self.factor_layout.addStretch()
        self.factor_scroll.setWidget(self.factor_container)
        fl.addWidget(self.factor_scroll)

        # Başlangıç faktörleri (2 tane)
        self._add_factor(name="Oleik Asit", unit="%", low=0.5, mid=1.0, high=2.0)
        self._add_factor(name="Tween 80",   unit="%", low=0.05, mid=0.1, high=0.2)

        left.addWidget(fcard)

        # — Tasarım tipi kartı —
        dcard = card_frame()
        dl = QVBoxLayout(dcard)
        dl.setContentsMargins(14, 12, 14, 12)
        dl.setSpacing(10)
        dl.addWidget(section_label("🔬  Tasarım Tipi"))

        dcontent = QHBoxLayout()
        dcontent.setSpacing(16)

        # Sol: seçim + run bilgisi
        dsel = QVBoxLayout()
        dsel.setSpacing(8)
        self.design_combo = QComboBox()
        self.design_combo.addItems(DESIGN_TYPES)
        self.design_combo.setCurrentText("Central Composite (CCD/RSM)")
        self.design_combo.setFixedHeight(32)
        self.design_combo.currentTextChanged.connect(self._on_design_changed)
        dsel.addWidget(self.design_combo)

        self.run_info_lbl = QLabel("")
        self.run_info_lbl.setStyleSheet(f"color:{GOLD}; font-size:13px; font-weight:bold; background:transparent;")
        dsel.addWidget(self.run_info_lbl)

        self.seviye_lbl = QLabel("")
        self.seviye_lbl.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        dsel.addWidget(self.seviye_lbl)

        dsel.addStretch()
        dcontent.addLayout(dsel)

        # Sağ: tasarım açıklaması
        self.design_info_box = QFrame()
        self.design_info_box.setStyleSheet(f"""
            QFrame {{
                background: {BG3};
                border: 1px solid #1a3050;
                border-radius: 6px;
            }}
            QLabel {{ background: transparent; }}
        """)
        dib_l = QVBoxLayout(self.design_info_box)
        dib_l.setContentsMargins(10, 8, 10, 8)
        dib_l.setSpacing(4)
        self.lbl_ne_zaman = info_label("")
        self.lbl_avantaj  = info_label("")
        self.lbl_dezavantaj = info_label("")
        self.lbl_oneri    = info_label("")
        for w in [self.lbl_ne_zaman, self.lbl_avantaj, self.lbl_dezavantaj, self.lbl_oneri]:
            dib_l.addWidget(w)
        dcontent.addWidget(self.design_info_box, 1)

        dl.addLayout(dcontent)
        left.addWidget(dcard)

        # — Yanıt seçimi kartı —
        rcard = card_frame()
        rl = QVBoxLayout(rcard)
        rl.setContentsMargins(14, 10, 14, 10)
        rl.setSpacing(8)
        rl.addWidget(section_label("🎯  Yanıt Değişkenleri (CQA)"))

        rgrid = QGridLayout()
        rgrid.setSpacing(6)
        self.resp_checks = {}
        items = list(RESPONSE_LABELS.items())
        for i, (key, lbl) in enumerate(items):
            cb = QCheckBox(lbl)
            cb.setChecked(key in ("mmad", "fpf_5um"))
            self.resp_checks[key] = cb
            rgrid.addWidget(cb, i // 3, i % 3)
        rl.addLayout(rgrid)
        left.addWidget(rcard)

        left.addStretch()

        # — Oluştur butonu —
        btn_gen = make_btn("▶  Deney Matrisi Oluştur", "rgba(20,70,120,0.9)", 42)
        btn_gen.setStyleSheet(btn_gen.styleSheet() + f"font-size: 14px; font-weight: bold;")
        btn_gen.clicked.connect(self._generate)
        left.addWidget(btn_gen)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(520)

        # ── Sağ panel: Matris önizleme + özet ───────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(10)

        # Özet bilgi çubuğu
        self.summary_frame = card_frame()
        sf_l = QHBoxLayout(self.summary_frame)
        sf_l.setContentsMargins(14, 8, 14, 8)
        sf_l.setSpacing(30)
        self.lbl_s_design  = QLabel("Tasarım: —")
        self.lbl_s_runs    = QLabel("Run: —")
        self.lbl_s_factors = QLabel("Faktör: —")
        self.lbl_s_resps   = QLabel("Yanıt: —")
        for l in [self.lbl_s_design, self.lbl_s_runs, self.lbl_s_factors, self.lbl_s_resps]:
            l.setStyleSheet(f"color:{TXT}; font-size:12px; font-weight:bold; background:transparent;")
            sf_l.addWidget(l)
        sf_l.addStretch()
        right.addWidget(self.summary_frame)

        # Tablo başlığı + export butonu
        tbl_hdr = QHBoxLayout()
        tbl_hdr.addWidget(section_label("📊  Deney Matrisi Önizleme"))
        tbl_hdr.addStretch()
        self.btn_export = make_btn("⬇  Excel'e Aktar", "rgba(20,80,20,0.8)", 28)
        self.btn_export.clicked.connect(self._export_excel)
        self.btn_export.setEnabled(False)
        tbl_hdr.addWidget(self.btn_export)
        right.addLayout(tbl_hdr)

        self.matrix_table = QTableWidget()
        self.matrix_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.matrix_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.matrix_table.setMinimumHeight(300)
        right.addWidget(self.matrix_table, 1)

        # Uyarı / bilgi alanı
        self.info_txt = QTextEdit()
        self.info_txt.setReadOnly(True)
        self.info_txt.setFixedHeight(80)
        self.info_txt.setStyleSheet(f"""
            QTextEdit {{
                background: {BG3};
                border: 1px solid #1a3050;
                border-radius: 4px;
                color: {TXT2};
                font-size: 11px;
            }}
        """)
        self.info_txt.setPlaceholderText("Deney matrisi oluşturulduğunda bilgiler burada görünür...")
        right.addWidget(self.info_txt)

        right_widget = QWidget()
        right_widget.setLayout(right)

        outer.addWidget(left_widget)
        outer.addWidget(right_widget, 1)

        # İlk güncelleme
        self._on_design_changed(self.design_combo.currentText())

    # ── İç metodlar ────────────────────────────────────────────────────────────
    def _add_factor(self, name=None, unit="", low=0.0, mid=0.5, high=1.0):
        idx = len(self._factor_rows)
        row = FactorRow(idx, self)
        if name:
            row.name_edit.setText(name)
            row.unit_edit.setText(unit)
            row.low_sp.setValue(low)
            row.mid_sp.setValue(mid)
            row.high_sp.setValue(high)
        row.deleted.connect(self._remove_factor)
        self._factor_rows.append(row)
        self.factor_layout.insertWidget(self.factor_layout.count() - 1, row)
        self._update_run_estimate()

    def _remove_factor(self, row_widget):
        if len(self._factor_rows) <= 1:
            QMessageBox.warning(self, "", "En az 1 faktör olmalıdır.")
            return
        self._factor_rows.remove(row_widget)
        row_widget.setParent(None)
        # İndeks numaralarını güncelle
        for i, r in enumerate(self._factor_rows):
            r.idx = i
            r.findChild(QLabel).setText(f"{i+1}.")
        self._update_run_estimate()

    def _on_design_changed(self, design_type):
        self.project.design_type = design_type
        info = DESIGN_INFO.get(design_type, {})
        k = len(self._factor_rows)

        # Run tahmini
        try:
            n_runs = info["run_formula"](k) if info else 0
        except Exception:
            n_runs = 0
        self.run_info_lbl.setText(f"Tahmini Run Sayısı: {n_runs}")
        self.seviye_lbl.setText(f"Faktör Seviyesi: {info.get('seviye','?')}")

        # Açıklamalar
        self.lbl_ne_zaman.setText(f"🕐 Ne zaman: {info.get('ne_zaman','')}")
        self.lbl_avantaj.setText(f"✅ Avantaj: {info.get('avantaj','')}")
        self.lbl_dezavantaj.setText(f"⚠️  Dezavantaj: {info.get('dezavantaj','')}")
        self.lbl_oneri.setText(f"💡 Öneri: {info.get('oneri','')}")

    def _update_run_estimate(self):
        design_type = self.design_combo.currentText()
        info = DESIGN_INFO.get(design_type, {})
        k = len(self._factor_rows)
        try:
            n_runs = info["run_formula"](k) if info else 0
        except Exception:
            n_runs = 0
        self.run_info_lbl.setText(f"Tahmini Run Sayısı: {n_runs}")

    def _collect_factors(self):
        factors = []
        for row in self._factor_rows:
            f = row.get_factor()
            if f["low"] >= f["high"]:
                QMessageBox.warning(self, "Hata",
                    f"'{f['name']}': Alt seviye üst seviyeden küçük olmalıdır.")
                return None
            if not (f["low"] <= f["mid"] <= f["high"]):
                f["mid"] = (f["low"] + f["high"]) / 2
            factors.append(f)
        return factors

    def _generate(self):
        # Faktörleri topla
        factors = self._collect_factors()
        if factors is None:
            return

        # Yanıtları topla
        responses = [k for k, cb in self.resp_checks.items() if cb.isChecked()]
        if not responses:
            QMessageBox.warning(self, "", "En az 1 yanıt değişkeni seçin.")
            return

        # Projeye kaydet
        self.project.factors   = factors
        self.project.responses = responses
        self.project.design_type = self.design_combo.currentText()

        # Matris oluştur
        df, err = self.project.generate_design_matrix()
        if err:
            QMessageBox.critical(self, "Hata", err)
            return

        # Tabloyu doldur
        self._fill_table(df)

        # Özet güncelle
        k = len(factors)
        n = len(df)
        self.lbl_s_design.setText(f"Tasarım: {self.project.design_type}")
        self.lbl_s_runs.setText(f"Run: {n}")
        self.lbl_s_factors.setText(f"Faktör: {k}")
        self.lbl_s_resps.setText(f"Yanıt: {len(responses)}")

        # Bilgi mesajı
        info = DESIGN_INFO.get(self.project.design_type, {})
        resp_names = [RESPONSE_LABELS.get(r, r) for r in responses]
        msg = (
            f"✅ Deney matrisi oluşturuldu.\n"
            f"   {n} run  |  {k} faktör  |  {len(responses)} yanıt\n"
            f"   Yanıtlar: {', '.join(resp_names)}\n"
            f"   Sonraki adım: Veri Girişi sekmesinden NGI sonuçlarını girin."
        )
        self.info_txt.setText(msg)
        self.btn_export.setEnabled(True)

        # Diğer sekmeleri bilgilendir
        self.design_generated.emit()
        self.app.status_bar.showMessage(
            f"Deney matrisi oluşturuldu — {n} run, {k} faktör", 5000)

    def _fill_table(self, df):
        self.matrix_table.clear()
        if df is None or df.empty:
            return
        cols = ["Run"] + list(df.columns)
        self.matrix_table.setColumnCount(len(cols))
        self.matrix_table.setRowCount(len(df))
        self.matrix_table.setHorizontalHeaderLabels(cols)

        # Birim satırını başlığa ekle
        for ci, col in enumerate(df.columns):
            f = next((x for x in self.project.factors if x["name"] == col), None)
            unit = f["unit"] if f and f.get("unit") else ""
            hdr_text = f"{col}\n({unit})" if unit else col
            self.matrix_table.horizontalHeaderItem(ci + 1).setText(hdr_text)

        for ri, row in df.iterrows():
            # Run no
            item = QTableWidgetItem(str(ri + 1))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setBackground(QColor("#1F4E79"))
            self.matrix_table.setItem(ri, 0, item)
            # Faktör değerleri
            for ci, col in enumerate(df.columns):
                val = row[col]
                item = QTableWidgetItem(f"{val:.4f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                # Merkez nokta farklı renk
                f = next((x for x in self.project.factors if x["name"] == col), None)
                if f:
                    mid = f.get("mid", (f["low"] + f["high"]) / 2)
                    if abs(val - mid) < 1e-6:
                        item.setBackground(QColor("#1a3a1a"))
                self.matrix_table.setItem(ri, ci + 1, item)

        self.matrix_table.resizeColumnsToContents()

    def _export_excel(self):
        if self.project.design_matrix is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Excel'e Aktar",
            f"DoE_Matrisi_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
            "Excel (*.xlsx)")
        if not path:
            return
        try:
            df = self.project.design_matrix.copy()
            df.insert(0, "Run No", range(1, len(df) + 1))
            # Yanıt sütunlarını boş olarak ekle
            for resp in self.project.responses:
                df[RESPONSE_LABELS.get(resp, resp)] = ""
            df.to_excel(path, index=False, engine="openpyxl")
            QMessageBox.information(self, "Başarılı",
                f"Excel dosyası kaydedildi:\n{path}\n\n"
                f"NGI ölçümlerinizi sarı sütunlara girin.")
            self.app.status_bar.showMessage(f"Excel kaydedildi: {path}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Excel kaydedilemedi:\n{e}")

    def refresh(self):
        """Diğer sekmelerden geri gelindiğinde güncelle."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER SEKMELER (Adım 2-7 için)
# ═══════════════════════════════════════════════════════════════════════════════
class PlaceholderTab(QWidget):
    """Henüz geliştirilmemiş sekmeler için yer tutucu."""
    def __init__(self, title, description, icon="🔧", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"font-size: 48px; background: transparent;")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {GOLD}; font-size: 18px; font-weight: bold; background: transparent;")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title_lbl)

        desc_lbl = QLabel(description)
        desc_lbl.setStyleSheet(f"color: {TXT2}; font-size: 13px; background: transparent;")
        desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_lbl.setWordWrap(True)
        desc_lbl.setMaximumWidth(500)
        lay.addWidget(desc_lbl)

        dev_lbl = QLabel("— Geliştirme aşamasında —")
        dev_lbl.setStyleSheet(f"color: #3a5070; font-size: 11px; margin-top: 20px; background: transparent;")
        dev_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(dev_lbl)

    def refresh(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# ANA PENCERE
# ═══════════════════════════════════════════════════════════════════════════════
class FormulasyonOptimizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = OptimizerProject()
        self.setWindowTitle("Formulasyon-Optimizer  v1")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 820)

        ico_path = resource_path("opt_icon.ico")
        if os.path.exists(ico_path):
            self.setWindowIcon(QIcon(ico_path))

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # ── Başlık çubuğu ────────────────────────────────────────────────────
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a1628,
                    stop:0.5 #0f2040,
                    stop:1 #0a1628
                );
                border-bottom: 2px solid {GOLD};
            }}
        """)
        hdr_lay = QHBoxLayout(header)
        hdr_lay.setContentsMargins(20, 0, 20, 0)

        # Sol: başlık
        title1 = QLabel("Formulasyon-Optimizer")
        title1.setStyleSheet(f"color: {GOLD}; font-size: 18px; font-weight: bold; background: transparent;")
        hdr_lay.addWidget(title1)

        sep = QLabel("│")
        sep.setStyleSheet(f"color: #2a4060; font-size: 20px; background: transparent;")
        hdr_lay.addWidget(sep)

        title2 = QLabel("Farmasötik İnhaler Formülasyon Geliştirme & Optimizasyon")
        title2.setStyleSheet(f"color: {TXT2}; font-size: 12px; background: transparent;")
        hdr_lay.addWidget(title2)
        hdr_lay.addStretch()

        # Sağ: versiyon + yeni proje butonu
        ver_lbl = QLabel("v1.0  |  Ph.Eur 2.9.18 / USP <601>")
        ver_lbl.setStyleSheet(f"color: #3a5070; font-size: 10px; background: transparent;")
        hdr_lay.addWidget(ver_lbl)

        hdr_lay.addSpacing(16)
        btn_new = make_btn("🗋  Yeni Proje", "rgba(160,100,0,0.4)", 30)
        btn_new.clicked.connect(self._new_project)
        hdr_lay.addWidget(btn_new)

        main_lay.addWidget(header)

        # ── Sekmeler ─────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        main_lay.addWidget(self.tabs, 1)

        # Sekme 1 — Deney Tasarımı
        self.tab1 = Tab1_Design(self.project, self)
        self.tab1.design_generated.connect(self._on_design_generated)
        self.tabs.addTab(self.tab1, "1 · Deney Tasarımı")

        # Sekme 2 — Veri Girişi (placeholder)
        self.tab2 = PlaceholderTab(
            "Veri Girişi",
            "Deney matrisi oluşturulduktan sonra her run için\n"
            "NGI ölçüm sonuçlarını (MMAD, FPF vb.) bu sekmeden gireceksiniz.",
            "📝")
        self.tabs.addTab(self.tab2, "2 · Veri Girişi")

        # Sekme 3 — Model
        self.tab3 = PlaceholderTab(
            "Model",
            "Veri girişi tamamlandıktan sonra OLS regresyon modeli kurulur.\n"
            "R², ANOVA tablosu ve katsayı analizi burada görünecek.",
            "📈")
        self.tabs.addTab(self.tab3, "3 · Model")

        # Sekme 4 — Response Surface
        self.tab4 = PlaceholderTab(
            "Response Surface",
            "Model kurulduktan sonra 3D yüzey ve kontur haritaları\n"
            "bu sekmede interaktif olarak incelenebilecek.",
            "🗺")
        self.tabs.addTab(self.tab4, "4 · Response Surface")

        # Sekme 5 — Optimizasyon
        self.tab5 = PlaceholderTab(
            "Optimizasyon",
            "Derringer-Suich desirability fonksiyonu ile\n"
            "USL/LSL kısıtlarına göre Top-5 optimum formülasyon önerisi.",
            "🎯")
        self.tabs.addTab(self.tab5, "5 · Optimizasyon")

        # Sekme 6 — Design Space
        self.tab6 = PlaceholderTab(
            "Design Space",
            "ICH Q8 uyumlu tasarım uzayı haritası.\n"
            "Tüm CQA'ların spec içinde kaldığı faktör bölgesi görselleştirilir.",
            "🗂")
        self.tabs.addTab(self.tab6, "6 · Design Space")

        # Sekme 7 — Doğrulama
        self.tab7 = PlaceholderTab(
            "Doğrulama",
            "Optimum formülasyonun tahmin değerleri ile\n"
            "gerçek NGI ölçümlerinin karşılaştırması.",
            "✅")
        self.tabs.addTab(self.tab7, "7 · Doğrulama")

        # ── Durum çubuğu ─────────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır — Deney Tasarımı sekmesinden başlayın.")

    def _on_tab_changed(self, idx):
        tab = self.tabs.widget(idx)
        if hasattr(tab, "refresh"):
            tab.refresh()

    def _on_design_generated(self):
        """Tasarım matrisi oluşturulduğunda diğer sekmeleri bilgilendir."""
        n = len(self.project.design_matrix) if self.project.design_matrix is not None else 0
        k = len(self.project.factors)
        self.status_bar.showMessage(
            f"Proje hazır — {n} run, {k} faktör, {len(self.project.responses)} yanıt  "
            f"|  Sonraki: Veri Girişi sekmesi", 8000)

    def _new_project(self):
        reply = QMessageBox.question(
            self, "Yeni Proje",
            "Mevcut proje verisi silinecek. Devam edilsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.project.reset()
            self.tab1.refresh()
            self.tabs.setCurrentIndex(0)
            self.status_bar.showMessage("Yeni proje oluşturuldu.")


# ═══════════════════════════════════════════════════════════════════════════════
# GİRİŞ NOKTASI
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = FormulasyonOptimizerApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
