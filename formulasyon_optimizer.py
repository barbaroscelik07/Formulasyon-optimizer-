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
        """Run sonuçlarını DataFrame olarak döner — eksik değerler NaN."""
        if self.design_matrix is None:
            return None
        rows = []
        for i in range(len(self.design_matrix)):
            row = {}
            for resp in self.responses:
                val = self.run_results.get(i, {}).get(resp, None)
                try:
                    row[resp] = float(val) if val is not None else np.nan
                except (TypeError, ValueError):
                    row[resp] = np.nan
            rows.append(row)
        return pd.DataFrame(rows, columns=self.responses)

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
        # design_combo henüz oluşturulmamış olabilir (_add_factor __init__'te çağrılır)
        if not hasattr(self, "design_combo") or not hasattr(self, "run_info_lbl"):
            return
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
# SEKME 2 — VERİ GİRİŞİ
# ═══════════════════════════════════════════════════════════════════════════════
class Tab2_DataEntry(QWidget):
    """Her run için NGI ölçüm sonuçlarını giriş tablosu."""

    def __init__(self, project: OptimizerProject, app_ref, parent=None):
        super().__init__(parent)
        self.project = project
        self.app     = app_ref
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # ── Üst bilgi çubuğu ────────────────────────────────────────────────
        info_card = card_frame()
        info_l = QHBoxLayout(info_card)
        info_l.setContentsMargins(14, 8, 14, 8)
        info_l.setSpacing(30)
        self.lbl_runs    = QLabel("Run: —")
        self.lbl_factors = QLabel("Faktör: —")
        self.lbl_resps   = QLabel("Yanıt: —")
        self.lbl_filled  = QLabel("Dolu: —")
        for l in [self.lbl_runs, self.lbl_factors, self.lbl_resps, self.lbl_filled]:
            l.setStyleSheet(f"color:{TXT}; font-size:12px; font-weight:bold; background:transparent;")
            info_l.addWidget(l)
        info_l.addStretch()

        self.btn_clear = make_btn("🗑  Temizle", "rgba(120,20,20,0.6)", 28)
        self.btn_clear.clicked.connect(self._clear_responses)
        info_l.addWidget(self.btn_clear)

        self.btn_import = make_btn("📂  Excel'den İçe Aktar", "rgba(20,60,100,0.8)", 28)
        self.btn_import.clicked.connect(self._import_excel)
        info_l.addWidget(self.btn_import)

        lay.addWidget(info_card)

        # ── Açıklama ────────────────────────────────────────────────────────
        hint = QLabel(
            "Her run için NGI ölçüm sonuçlarını girin.  "
            "Mavi sütunlar = faktör değerleri (salt okunur).  "
            "Yeşil sütunlar = NGI yanıtları (düzenlenebilir).  "
            "Boş bırakılan run'lar model kurulumuna dahil edilmez.")
        hint.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        # ── Ana tablo ───────────────────────────────────────────────────────
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(self.table.styleSheet() + """
            QTableWidget { alternate-background-color: #111620; }
        """)
        self.table.cellChanged.connect(self._on_cell_changed)
        lay.addWidget(self.table, 1)

        # ── Alt butonlar ────────────────────────────────────────────────────
        bot = QHBoxLayout()
        bot.setSpacing(10)

        self.lbl_validation = QLabel("")
        self.lbl_validation.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        bot.addWidget(self.lbl_validation)
        bot.addStretch()

        self.btn_next = make_btn("Model Kur  ▶", "rgba(20,80,20,0.8)", 36)
        self.btn_next.setStyleSheet(self.btn_next.styleSheet() +
                                    "font-size:13px; font-weight:bold;")
        self.btn_next.clicked.connect(self._go_to_model)
        bot.addWidget(self.btn_next)
        lay.addLayout(bot)

    # ── Yardımcı ──────────────────────────────────────────────────────────────
    def _n_factor_cols(self):
        return len(self.project.factors)

    def _resp_col_indices(self):
        """Yanıt sütunlarının tablo indekslerini döner."""
        nf = self._n_factor_cols()
        return list(range(1 + nf, 1 + nf + len(self.project.responses)))

    # ── Public ────────────────────────────────────────────────────────────────
    def refresh(self):
        """Sekme 1'den gelindiğinde tabloyu yeniden oluştur."""
        if self.project.design_matrix is None:
            self._show_empty()
            return
        self._build_table()
        self._update_info()

    def _show_empty(self):
        self.table.clear()
        self.table.setRowCount(1)
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels([""])
        item = QTableWidgetItem(
            "Önce 'Deney Tasarımı' sekmesinden matris oluşturun.")
        item.setForeground(QColor(TXT2))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.table.setItem(0, 0, item)
        self.lbl_runs.setText("Run: —")
        self.lbl_factors.setText("Faktör: —")
        self.lbl_resps.setText("Yanıt: —")
        self.lbl_filled.setText("Dolu: —")

    def _build_table(self):
        """Design matrix + boş yanıt sütunlarıyla tabloyu oluştur."""
        self.table.blockSignals(True)
        df      = self.project.design_matrix
        factors = self.project.factors
        resps   = self.project.responses
        n_runs  = len(df)

        # Sütunlar: Run No | faktörler... | yanıtlar...
        headers = ["Run"] + \
                  [f"{f['name']}" + (f"\n({f['unit']})" if f.get("unit") else "")
                   for f in factors] + \
                  [RESPONSE_LABELS.get(r, r) for r in resps]

        self.table.setColumnCount(len(headers))
        self.table.setRowCount(n_runs)
        self.table.setHorizontalHeaderLabels(headers)

        nf = len(factors)

        for ri in range(n_runs):
            # Run no
            item = QTableWidgetItem(str(ri + 1))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setBackground(QColor("#1F4E79"))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(ri, 0, item)

            # Faktör değerleri — salt okunur, mavi
            for ci, f in enumerate(factors):
                val = df.iloc[ri][f["name"]]
                item = QTableWidgetItem(f"{val:.4f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setBackground(QColor("#0d1e36"))
                item.setForeground(QColor("#90b8e0"))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.table.setItem(ri, 1 + ci, item)

            # Yanıt sütunları — düzenlenebilir, yeşil tonu
            for ci, resp in enumerate(resps):
                col_idx = 1 + nf + ci
                existing = self.project.run_results.get(ri, {}).get(resp, "")
                val_str = f"{existing:.6g}" if isinstance(existing, float) else ""
                item = QTableWidgetItem(val_str)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setBackground(QColor("#0d2010"))
                item.setForeground(QColor(TXT))
                self.table.setItem(ri, col_idx, item)

        self.table.blockSignals(False)
        self._update_info()

    def _on_cell_changed(self, row, col):
        """Yanıt hücresi değiştiğinde project.run_results güncelle."""
        nf = self._n_factor_cols()
        resp_start = 1 + nf
        if col < resp_start:
            return
        resp_idx = col - resp_start
        if resp_idx >= len(self.project.responses):
            return
        resp_key = self.project.responses[resp_idx]
        item = self.table.item(row, col)
        if item is None:
            return
        txt = item.text().strip().replace(",", ".")
        if txt == "":
            # Boş — sil
            if row in self.project.run_results:
                self.project.run_results[row].pop(resp_key, None)
        else:
            try:
                val = float(txt)
                if row not in self.project.run_results:
                    self.project.run_results[row] = {}
                self.project.run_results[row][resp_key] = val
                # Hücre rengini dolu olarak işaretle
                item.setBackground(QColor("#0a2e14"))
                item.setForeground(QColor("#80e890"))
            except ValueError:
                item.setBackground(QColor("#3a0a0a"))
                item.setForeground(QColor("#ff6060"))
        self._update_info()

    def _update_info(self):
        df    = self.project.design_matrix
        resps = self.project.responses
        if df is None:
            return
        n_runs  = len(df)
        n_resps = len(resps)
        # Kaç run'da en az 1 yanıt dolu?
        filled = sum(1 for i in range(n_runs)
                     if any(isinstance(self.project.run_results.get(i, {}).get(r), float)
                            for r in resps))
        self.lbl_runs.setText(f"Run: {n_runs}")
        self.lbl_factors.setText(f"Faktör: {len(self.project.factors)}")
        self.lbl_resps.setText(f"Yanıt: {n_resps}")
        color = GREEN if filled == n_runs else GOLD if filled > 0 else TXT2
        self.lbl_filled.setStyleSheet(
            f"color:{color}; font-size:12px; font-weight:bold; background:transparent;")
        self.lbl_filled.setText(f"Dolu: {filled}/{n_runs}")

        # Doğrulama mesajı
        if filled == 0:
            msg = "Henüz veri girilmedi."
            self.lbl_validation.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        elif filled < n_runs:
            eksik = n_runs - filled
            msg = f"⚠  {eksik} run eksik — model kurulabilir ama dikkatli değerlendirin."
            self.lbl_validation.setStyleSheet(f"color:{GOLD}; font-size:11px; background:transparent;")
        else:
            msg = f"✅  Tüm {n_runs} run tamamlandı — Model Kur'a geçebilirsiniz."
            self.lbl_validation.setStyleSheet(f"color:{GREEN}; font-size:11px; background:transparent;")
        self.lbl_validation.setText(msg)

    def _clear_responses(self):
        reply = QMessageBox.question(
            self, "Temizle",
            "Tüm yanıt değerleri silinecek. Devam edilsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.project.run_results = {}
            self.project.model_results = {}
            self._build_table()

    def _import_excel(self):
        """Excel dosyasından yanıt değerlerini içe aktar."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Excel Dosyası Seç", "", "Excel (*.xlsx *.xls)")
        if not path:
            return
        try:
            df_imp = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Dosya okunamadı:\n{e}")
            return

        # Yanıt sütunlarını eşleştir
        imported = 0
        resps = self.project.responses
        resp_labels = {RESPONSE_LABELS.get(r, r): r for r in resps}

        for ri in range(min(len(df_imp), len(self.project.design_matrix))):
            for col_name, resp_key in resp_labels.items():
                # Hem tam isim hem kısmi eşleşme dene
                matched_col = None
                for c in df_imp.columns:
                    if col_name.lower() in str(c).lower() or str(c).lower() in col_name.lower():
                        matched_col = c
                        break
                if matched_col is None:
                    continue
                val = df_imp.iloc[ri].get(matched_col, None)
                if pd.notna(val):
                    try:
                        fval = float(val)
                        if ri not in self.project.run_results:
                            self.project.run_results[ri] = {}
                        self.project.run_results[ri][resp_key] = fval
                        imported += 1
                    except (ValueError, TypeError):
                        pass

        self._build_table()
        QMessageBox.information(self, "İçe Aktarma",
            f"{imported} değer başarıyla içe aktarıldı.")
        self.app.status_bar.showMessage(f"{imported} değer içe aktarıldı.", 4000)

    def _go_to_model(self):
        """Sekme 3'e geç."""
        filled = sum(1 for i in range(len(self.project.design_matrix or []))
                     if self.project.run_results.get(i))
        if filled == 0:
            QMessageBox.warning(self, "",
                "Model kurmak için en az birkaç run verisi gereklidir.")
            return
        self.app.tabs.setCurrentIndex(2)


# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 3 — MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class ModelWorker(QThread):
    """OLS model fit işlemini arka planda çalıştırır."""
    finished = pyqtSignal(dict, dict)   # results, errors
    progress = pyqtSignal(str)

    def __init__(self, project):
        super().__init__()
        self.project = project

    def run(self):
        try:
            self._run_safe()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            write_log(f"ModelWorker crash:\n{tb}")
            self.finished.emit({}, {"_global": f"Model crash: {e}\n\n{tb}"})

    def _run_safe(self):
        import statsmodels.api as sm
        from statsmodels.stats.anova import anova_lm
        import statsmodels.formula.api as smf

        results = {}
        errors  = {}
        p       = self.project
        safe    = p.get_safe_names()

        # Yanıt verisini topla
        resp_df = p.build_response_table()
        if resp_df is None or resp_df.empty:
            errors["_global"] = "Yanıt verisi bulunamadı."
            self.finished.emit(results, errors)
            return

        # Kodlanmış faktör matrisi
        coded_df, _ = p.get_coded_df()
        if coded_df is None:
            errors["_global"] = "Tasarım matrisi bulunamadı."
            self.finished.emit(results, errors)
            return

        # Her yanıt için model kur
        for resp_key in p.responses:
            self.progress.emit(f"Model kuruluyor: {RESPONSE_LABELS.get(resp_key, resp_key)}")
            try:
                # Sütun yoksa atla
                if resp_key not in resp_df.columns:
                    errors[resp_key] = "Yanıt sütunu bulunamadı."
                    continue

                y = resp_df[resp_key].values.astype(float)

                # Eksik satırları çıkar (NaN olanlar)
                mask = ~np.isnan(y)
                min_obs = len(safe) + 2
                if mask.sum() < min_obs:
                    errors[resp_key] = (
                        f"Yeterli veri yok: {int(mask.sum())} run dolu, "
                        f"en az {min_obs} gerekli.")
                    continue

                X_df    = coded_df[mask].copy().reset_index(drop=True)
                y_clean = y[mask]
                y_s     = pd.Series(y_clean, name=resp_key)
                n_factors = len(safe)

                # Karesel terimler
                for name in safe:
                    X_df[f"{name}_sq"] = X_df[name] ** 2

                # İkili etkileşimler
                for i in range(n_factors):
                    for j in range(i+1, n_factors):
                        col = f"{safe[i]}_x_{safe[j]}"
                        X_df[col] = X_df[safe[i]] * X_df[safe[j]]

                X_mat = sm.add_constant(X_df, has_constant="add")
                model = sm.OLS(y_s, X_mat).fit()

                # ANOVA
                try:
                    anova = sm.stats.anova_lm(model, typ=2)
                except Exception:
                    anova = None

                # Artık normallik testi (Shapiro-Wilk 3-5000 arası çalışır)
                sw_p = None
                try:
                    n_res = len(model.resid)
                    if 3 <= n_res <= 5000:
                        _, sw_p = stats.shapiro(model.resid)
                except Exception:
                    pass

                results[resp_key] = {
                    "model":      model,
                    "anova":      anova,
                    "X_df":       X_df,
                    "y":          y_clean,
                    "sw_p":       sw_p,
                    "r2":         model.rsquared,
                    "r2_adj":     model.rsquared_adj,
                    "rmse":       float(np.sqrt(model.mse_resid)),
                    "f_pvalue":   float(model.f_pvalue),
                    "safe_names": safe,
                    "n_obs":      int(mask.sum()),
                }

            except Exception as e:
                import traceback
                errors[resp_key] = f"{e}\n{traceback.format_exc()}"

        self.finished.emit(results, errors)


class Tab3_Model(QWidget):
    """Sekme 3 — OLS model fit, R², ANOVA, katsayı tablosu, artık grafikleri."""

    model_ready = pyqtSignal()

    def __init__(self, project: OptimizerProject, app_ref, parent=None):
        super().__init__(parent)
        self.project = project
        self.app     = app_ref
        self._worker = None
        self._build()

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # ── Sol: kontroller + özet tablo ──────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(10)

        # Başlık + buton
        top = QHBoxLayout()
        top.addWidget(section_label("📈  Regresyon Modeli"))
        top.addStretch()
        self.btn_fit = make_btn("▶  Model Kur", "rgba(20,80,20,0.9)", 34)
        self.btn_fit.setStyleSheet(self.btn_fit.styleSheet() +
                                   "font-size:13px; font-weight:bold;")
        self.btn_fit.clicked.connect(self._fit_models)
        top.addWidget(self.btn_fit)
        left.addLayout(top)

        # Model tipi açıklaması
        info = info_label(
            "Quadratic (ikinci dereceden) model: ana etkiler + ikili etkileşimler + "
            "karesel terimler. Seçilen yanıt değişkenlerine ayrı ayrı OLS modeli kurulur.")
        left.addWidget(info)

        # Yanıt seçim combo
        resp_row = QHBoxLayout()
        resp_row.addWidget(QLabel("Yanıt:"))
        self.resp_combo = QComboBox()
        self.resp_combo.setFixedHeight(28)
        self.resp_combo.currentTextChanged.connect(self._on_resp_changed)
        resp_row.addWidget(self.resp_combo)
        resp_row.addStretch()
        left.addLayout(resp_row)

        # ── Özet kart ─────────────────────────────────────────────────────────
        sum_card = card_frame()
        sl = QGridLayout(sum_card)
        sl.setContentsMargins(14, 10, 14, 10)
        sl.setSpacing(8)

        def stat_pair(label, attr):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
            val = QLabel("—")
            val.setStyleSheet(f"color:{GOLD}; font-size:14px; font-weight:bold; background:transparent;")
            setattr(self, attr, val)
            return lbl, val

        for row, (lbl_txt, attr) in enumerate([
            ("R²",           "lbl_r2"),
            ("Adj R²",       "lbl_r2adj"),
            ("RMSE",         "lbl_rmse"),
            ("Model p",      "lbl_fp"),
            ("Shapiro-Wilk p", "lbl_swp"),
            ("Gözlem (n)",   "lbl_nobs"),
        ]):
            lbl, val = stat_pair(lbl_txt, attr)
            sl.addWidget(lbl, row, 0)
            sl.addWidget(val, row, 1)

        left.addWidget(sum_card)

        # ── Katsayı tablosu ───────────────────────────────────────────────────
        left.addWidget(section_label("🔢  Katsayılar & Anlamlılık"))
        self.coef_table = QTableWidget()
        self.coef_table.setColumnCount(5)
        self.coef_table.setHorizontalHeaderLabels(
            ["Terim", "Katsayı", "Std Hata", "t", "p"])
        self.coef_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.coef_table.horizontalHeader().setStretchLastSection(True)
        self.coef_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.coef_table.setMinimumHeight(200)
        left.addWidget(self.coef_table, 1)

        # ── ANOVA tablosu ─────────────────────────────────────────────────────
        left.addWidget(section_label("📊  ANOVA Tablosu"))
        self.anova_table = QTableWidget()
        self.anova_table.setColumnCount(5)
        self.anova_table.setHorizontalHeaderLabels(
            ["Kaynak", "SS", "df", "MS", "p"])
        self.anova_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.anova_table.horizontalHeader().setStretchLastSection(True)
        self.anova_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.anova_table.setFixedHeight(180)
        left.addWidget(self.anova_table)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(480)
        left_w.setMaximumWidth(560)

        # ── Sağ: Artık grafikleri ─────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(section_label("🔬  Artık (Residual) Analizi"))

        # 4 grafik: Predicted vs Actual, Residuals vs Fitted,
        #           Normal Q-Q, Residuals vs Order
        self.fig = Figure(figsize=(9, 7), facecolor=BG2)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background: {BG2};")
        right.addWidget(self.canvas, 1)

        # Durum mesajı
        self.status_lbl = QLabel("Model henüz kurulmadı.")
        self.status_lbl.setStyleSheet(
            f"color:{TXT2}; font-size:11px; background:transparent;")
        right.addWidget(self.status_lbl)

        # İleri butonu
        bot = QHBoxLayout()
        bot.addStretch()
        self.btn_next = make_btn("Response Surface  ▶", "rgba(20,80,20,0.8)", 36)
        self.btn_next.setStyleSheet(self.btn_next.styleSheet() +
                                    "font-size:13px; font-weight:bold;")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(lambda: self.app.tabs.setCurrentIndex(3))
        bot.addWidget(self.btn_next)
        right.addLayout(bot)

        right_w = QWidget()
        right_w.setLayout(right)

        outer.addWidget(left_w)
        outer.addWidget(right_w, 1)

    # ── Public ────────────────────────────────────────────────────────────────
    def refresh(self):
        if self.project.design_matrix is None:
            return
        # Yanıt combo'yu güncelle
        self.resp_combo.blockSignals(True)
        self.resp_combo.clear()
        for r in self.project.responses:
            self.resp_combo.addItem(RESPONSE_LABELS.get(r, r), r)
        self.resp_combo.blockSignals(False)
        # Daha önce model kurulmuşsa sonuçları göster
        if self.project.model_results:
            self._show_results_for_current()

    # ── Model fit ─────────────────────────────────────────────────────────────
    def _fit_models(self):
        if self.project.design_matrix is None:
            QMessageBox.warning(self, "", "Önce Deney Tasarımı sekmesinden matris oluşturun.")
            return
        filled = sum(1 for i in range(len(self.project.design_matrix))
                     if self.project.run_results.get(i))
        if filled == 0:
            QMessageBox.warning(self, "", "Önce Veri Girişi sekmesinden NGI sonuçlarını girin.")
            return

        self.btn_fit.setEnabled(False)
        self.btn_fit.setText("⏳ Hesaplanıyor...")
        self.status_lbl.setText("Model kuruluyor...")

        self._worker = ModelWorker(self.project)
        self._worker.finished.connect(self._on_model_done)
        self._worker.progress.connect(
            lambda msg: self.app.status_bar.showMessage(msg))
        self._worker.start()

    def _on_model_done(self, results, errors):
        self.btn_fit.setEnabled(True)
        self.btn_fit.setText("▶  Model Kur")

        if "_global" in errors:
            QMessageBox.critical(self, "Hata", errors["_global"])
            return

        self.project.model_results = results
        self.project.model_errors  = errors

        # Combo güncelle
        self.resp_combo.blockSignals(True)
        self.resp_combo.clear()
        for r in self.project.responses:
            label = RESPONSE_LABELS.get(r, r)
            if r in errors:
                label += "  ⚠"
            self.resp_combo.addItem(label, r)
        self.resp_combo.blockSignals(False)

        self._show_results_for_current()
        self.btn_next.setEnabled(bool(results))
        self.model_ready.emit()
        self.app.status_bar.showMessage(
            f"Model hazır — {len(results)} yanıt başarıyla modellendi.", 6000)

        if errors:
            err_list = "\n".join(f"  • {RESPONSE_LABELS.get(k,k)}: {v}"
                                 for k, v in errors.items() if k != "_global")
            QMessageBox.warning(self, "Uyarı",
                f"Bazı yanıtlar modellenemedi:\n{err_list}")

    def _on_resp_changed(self, _):
        self._show_results_for_current()

    def _show_results_for_current(self):
        resp_key = self.resp_combo.currentData()
        if resp_key is None:
            return
        res = self.project.model_results.get(resp_key)
        if res is None:
            err = self.project.model_errors.get(resp_key, "Model kurulmadı.")
            self.status_lbl.setText(f"⚠  {err}")
            self._clear_display()
            return

        model = res["model"]

        # ── Özet istatistikler ────────────────────────────────────────────────
        def fmt_p(p):
            if p is None: return "—"
            if p < 0.001: return "<0.001 ✅"
            if p < 0.05:  return f"{p:.4f} ✅"
            return f"{p:.4f} ⚠"

        def color_r2(v):
            c = GREEN if v >= 0.9 else GOLD if v >= 0.7 else RED
            return f'<span style="color:{c}">{v:.4f}</span>'

        self.lbl_r2.setText(f"{res['r2']:.4f}")
        self.lbl_r2.setStyleSheet(
            f"color:{'#00B050' if res['r2']>=0.9 else '#FFC600' if res['r2']>=0.7 else '#C00000'};"
            f"font-size:14px;font-weight:bold;background:transparent;")
        self.lbl_r2adj.setText(f"{res['r2_adj']:.4f}")
        self.lbl_rmse.setText(f"{res['rmse']:.4f}")
        self.lbl_fp.setText(fmt_p(res['f_pvalue']))
        sw_p = res.get("sw_p")
        self.lbl_swp.setText(
            f"{sw_p:.4f} {'✅ Normal' if sw_p and sw_p>0.05 else '⚠ Normal değil'}"
            if sw_p else "—")
        self.lbl_nobs.setText(str(res["n_obs"]))

        # ── Katsayı tablosu ───────────────────────────────────────────────────
        params  = model.params
        bse     = model.bse
        tvalues = model.tvalues
        pvalues = model.pvalues

        self.coef_table.setRowCount(len(params))
        for i, (name, coef) in enumerate(params.items()):
            p_val = pvalues[name]
            significant = p_val < 0.05

            name_item = QTableWidgetItem(name)
            if significant:
                name_item.setForeground(QColor(GREEN))
                name_item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            else:
                name_item.setForeground(QColor(TXT2))

            items = [
                name_item,
                QTableWidgetItem(f"{coef:.4f}"),
                QTableWidgetItem(f"{bse[name]:.4f}"),
                QTableWidgetItem(f"{tvalues[name]:.3f}"),
                QTableWidgetItem(f"{p_val:.4f}"),
            ]
            for j, item in enumerate(items):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if j > 0:
                    if significant:
                        item.setBackground(QColor("#0a2a0a"))
                    else:
                        item.setBackground(QColor(BG3))
                self.coef_table.setItem(i, j, item)

            # p-değeri renklendirme
            p_item = self.coef_table.item(i, 4)
            if p_val < 0.001:
                p_item.setForeground(QColor(GREEN))
            elif p_val < 0.05:
                p_item.setForeground(QColor("#90e890"))
            elif p_val < 0.1:
                p_item.setForeground(QColor(GOLD))
            else:
                p_item.setForeground(QColor(TXT2))

        # ── ANOVA tablosu ─────────────────────────────────────────────────────
        anova = res.get("anova")
        if anova is not None:
            self.anova_table.setRowCount(len(anova))
            for i, (src, row) in enumerate(anova.iterrows()):
                p_v = row.get("PR(>F)", np.nan)
                items = [
                    QTableWidgetItem(str(src)),
                    QTableWidgetItem(f"{row.get('sum_sq', 0):.4f}"),
                    QTableWidgetItem(f"{int(row.get('df', 0))}"),
                    QTableWidgetItem(f"{row.get('sum_sq', 0)/max(row.get('df',1),1):.4f}"),
                    QTableWidgetItem(f"{p_v:.4f}" if not np.isnan(p_v) else "—"),
                ]
                for j, item in enumerate(items):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setBackground(QColor(BG3 if i % 2 == 0 else BG2))
                    if j == 4 and not np.isnan(p_v):
                        item.setForeground(
                            QColor(GREEN if p_v < 0.05 else TXT2))
                    self.anova_table.setItem(i, j, item)

        # ── Artık grafikleri ──────────────────────────────────────────────────
        self._plot_residuals(res)

        self.status_lbl.setText(
            f"✅  Model hazır  |  R²={res['r2']:.4f}  |  Adj R²={res['r2_adj']:.4f}  "
            f"|  n={res['n_obs']}")

    def _clear_display(self):
        self.coef_table.setRowCount(0)
        self.anova_table.setRowCount(0)
        self.fig.clear()
        self.canvas.draw()
        for attr in ["lbl_r2","lbl_r2adj","lbl_rmse","lbl_fp","lbl_swp","lbl_nobs"]:
            getattr(self, attr).setText("—")

    def _plot_residuals(self, res):
        model = res["model"]
        fitted   = model.fittedvalues.values
        residuals = model.resid.values
        y_actual  = res["y"]

        self.fig.clear()
        self.fig.patch.set_facecolor(BG2)

        axes = self.fig.subplots(2, 2)
        plot_cfg = dict(facecolor=BG3, labelcolor=TXT, titlecolor=GOLD)

        for ax in axes.flat:
            ax.set_facecolor(BG3)
            ax.tick_params(colors=TXT2, labelsize=8)
            ax.xaxis.label.set_color(TXT2)
            ax.yaxis.label.set_color(TXT2)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a4060")

        # 1. Predicted vs Actual
        ax1 = axes[0, 0]
        mn = min(min(y_actual), min(fitted))
        mx = max(max(y_actual), max(fitted))
        ax1.scatter(y_actual, fitted, color=CP[0], s=40, zorder=3, alpha=0.85)
        ax1.plot([mn, mx], [mn, mx], '--', color=GOLD, lw=1, label="İdeal")
        ax1.set_xlabel("Gerçek"); ax1.set_ylabel("Tahmin")
        ax1.set_title("Tahmin vs Gerçek", color=GOLD, fontsize=9)
        ax1.legend(fontsize=7, labelcolor=TXT2,
                   facecolor=BG2, edgecolor="#2a4060")

        # 2. Residuals vs Fitted
        ax2 = axes[0, 1]
        ax2.scatter(fitted, residuals, color=CP[1], s=40, zorder=3, alpha=0.85)
        ax2.axhline(0, color=GOLD, lw=1, linestyle="--")
        ax2.set_xlabel("Tahmin"); ax2.set_ylabel("Artık")
        ax2.set_title("Artık vs Tahmin", color=GOLD, fontsize=9)

        # 3. Normal Q-Q
        ax3 = axes[1, 0]
        (osm, osr), (slope, intercept, _) = stats.probplot(residuals)
        ax3.scatter(osm, osr, color=CP[2], s=30, zorder=3, alpha=0.85)
        line_x = np.array([osm[0], osm[-1]])
        ax3.plot(line_x, slope * line_x + intercept,
                 '--', color=GOLD, lw=1)
        ax3.set_xlabel("Teorik Kantil"); ax3.set_ylabel("Örnek Kantil")
        ax3.set_title("Normal Q-Q", color=GOLD, fontsize=9)

        # 4. Residuals vs Order
        ax4 = axes[1, 1]
        ax4.plot(range(1, len(residuals)+1), residuals,
                 'o-', color=CP[3], markersize=4, lw=1, alpha=0.85)
        ax4.axhline(0, color=GOLD, lw=1, linestyle="--")
        ax4.set_xlabel("Gözlem Sırası"); ax4.set_ylabel("Artık")
        ax4.set_title("Artık vs Sıra", color=GOLD, fontsize=9)

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 4 — RESPONSE SURFACE
# ═══════════════════════════════════════════════════════════════════════════════
class Tab4_ResponseSurface(QWidget):
    """3D yüzey + kontur haritası + sabit faktör slider'ları."""

    def __init__(self, project: OptimizerProject, app_ref, parent=None):
        super().__init__(parent)
        self.project = project
        self.app     = app_ref
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # ── Kontrol çubuğu ────────────────────────────────────────────────────
        ctrl = card_frame()
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(14, 8, 14, 8)
        cl.setSpacing(16)

        cl.addWidget(QLabel("Yanıt:"))
        self.resp_combo = QComboBox(); self.resp_combo.setFixedHeight(28)
        cl.addWidget(self.resp_combo)

        cl.addWidget(QLabel("X Ekseni:"))
        self.x_combo = QComboBox(); self.x_combo.setFixedHeight(28)
        cl.addWidget(self.x_combo)

        cl.addWidget(QLabel("Y Ekseni:"))
        self.y_combo = QComboBox(); self.y_combo.setFixedHeight(28)
        cl.addWidget(self.y_combo)

        cl.addWidget(QLabel("Çözünürlük:"))
        self.res_spin = QSpinBox()
        self.res_spin.setRange(20, 100); self.res_spin.setValue(40)
        self.res_spin.setFixedWidth(60); self.res_spin.setFixedHeight(28)
        cl.addWidget(self.res_spin)

        cl.addStretch()

        self.btn_plot = make_btn("▶  Grafik Çiz", "rgba(20,80,20,0.9)", 32)
        self.btn_plot.setStyleSheet(self.btn_plot.styleSheet() +
                                    "font-size:12px; font-weight:bold;")
        self.btn_plot.clicked.connect(self._plot)
        cl.addWidget(self.btn_plot)

        outer.addWidget(ctrl)

        # ── Sabit faktör slider'ları ───────────────────────────────────────────
        self.slider_card = card_frame()
        self.slider_layout = QHBoxLayout(self.slider_card)
        self.slider_layout.setContentsMargins(14, 8, 14, 8)
        self.slider_layout.setSpacing(20)
        self.slider_card.setFixedHeight(70)
        self._slider_widgets = {}   # {factor_name: (label, spinbox)}
        outer.addWidget(self.slider_card)

        # ── Ana grafik alanı ─────────────────────────────────────────────────
        self.fig = Figure(figsize=(13, 5), facecolor=BG2)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background: {BG2};")
        outer.addWidget(self.canvas, 1)

        # ── Alt çubuk ────────────────────────────────────────────────────────
        bot = QHBoxLayout()
        self.status_lbl = QLabel("Model kurulduktan sonra grafik çizebilirsiniz.")
        self.status_lbl.setStyleSheet(
            f"color:{TXT2}; font-size:11px; background:transparent;")
        bot.addWidget(self.status_lbl)
        bot.addStretch()
        self.btn_next = make_btn("Optimizasyon  ▶", "rgba(20,80,20,0.8)", 34)
        self.btn_next.setStyleSheet(self.btn_next.styleSheet() +
                                    "font-size:12px; font-weight:bold;")
        self.btn_next.clicked.connect(lambda: self.app.tabs.setCurrentIndex(4))
        bot.addWidget(self.btn_next)
        outer.addLayout(bot)

        # Sinyal bağlantıları
        self.resp_combo.currentTextChanged.connect(self._on_selection_changed)
        self.x_combo.currentTextChanged.connect(self._on_selection_changed)
        self.y_combo.currentTextChanged.connect(self._on_selection_changed)

    # ── Public ───────────────────────────────────────────────────────────────
    def refresh(self):
        if not self.project.model_results:
            return
        self._populate_combos()

    # ── İç metodlar ──────────────────────────────────────────────────────────
    def _populate_combos(self):
        """Combo kutularını model sonuçlarına göre doldur."""
        self.resp_combo.blockSignals(True)
        self.x_combo.blockSignals(True)
        self.y_combo.blockSignals(True)

        self.resp_combo.clear()
        for r in self.project.responses:
            if r in self.project.model_results:
                self.resp_combo.addItem(RESPONSE_LABELS.get(r, r), r)

        self.x_combo.clear()
        self.y_combo.clear()
        safe = self.project.get_safe_names()
        factors = self.project.factors
        for i, f in enumerate(factors):
            label = f"{f['name']}" + (f" ({f['unit']})" if f.get("unit") else "")
            self.x_combo.addItem(label, i)
            self.y_combo.addItem(label, i)

        # Varsayılan: X=0, Y=1
        if self.x_combo.count() >= 2:
            self.y_combo.setCurrentIndex(1)

        self.resp_combo.blockSignals(False)
        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)

        self._build_sliders()

    def _build_sliders(self):
        """Sabit faktörler için slider (SpinBox) oluştur."""
        # Mevcut widget'ları temizle
        while self.slider_layout.count():
            item = self.slider_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._slider_widgets = {}

        xi = self.x_combo.currentData() or 0
        yi = self.y_combo.currentData() or 1

        for i, f in enumerate(self.project.factors):
            if i in (xi, yi):
                continue
            mid = f.get("mid", (f["low"] + f["high"]) / 2)
            grp = QGroupBox(f"{f['name']}")
            grp.setFixedWidth(150)
            gl = QVBoxLayout(grp)
            gl.setContentsMargins(6, 4, 6, 4)
            sp = QDoubleSpinBox()
            sp.setRange(f["low"], f["high"])
            sp.setValue(mid)
            sp.setDecimals(3)
            sp.setSingleStep((f["high"] - f["low"]) / 20)
            sp.setFixedHeight(26)
            if f.get("unit"):
                sp.setSuffix(f"  {f['unit']}")
            gl.addWidget(sp)
            self.slider_layout.addWidget(grp)
            self._slider_widgets[i] = sp

        self.slider_layout.addStretch()

    def _on_selection_changed(self, _=None):
        self._build_sliders()

    def _get_fixed_values(self):
        """Sabit faktörlerin şu anki değerlerini döner: {factor_idx: value}"""
        return {i: sp.value() for i, sp in self._slider_widgets.items()}

    def _predict_grid(self, resp_key, xi, yi, n):
        """X-Y ızgara üzerinde tahmin yüzeyi hesapla."""
        res   = self.project.model_results[resp_key]
        model = res["model"]
        safe  = self.project.get_safe_names()
        factors = self.project.factors

        fx = factors[xi]; fy = factors[yi]
        x_real = np.linspace(fx["low"], fx["high"], n)
        y_real = np.linspace(fy["low"], fy["high"], n)
        XX, YY = np.meshgrid(x_real, y_real)

        fixed = self._get_fixed_values()

        # Kodlanmış değerlere çevir
        def encode(val, f):
            lo, hi = f["low"], f["high"]
            mid = f.get("mid", (lo + hi) / 2)
            if hi == lo: return 0.0
            if val <= mid and (mid - lo) != 0:
                return (val - lo) / (mid - lo) - 1
            elif (hi - mid) != 0:
                return (val - mid) / (hi - mid)
            return 0.0

        ZZ = np.zeros_like(XX)
        n_factors = len(factors)

        import statsmodels.api as _sm
        param_idx = model.params.index

        for r in range(n):
            for c in range(n):
                row_vals = {}
                for k, f in enumerate(factors):
                    if k == xi:
                        row_vals[safe[k]] = encode(XX[r, c], f)
                    elif k == yi:
                        row_vals[safe[k]] = encode(YY[r, c], f)
                    else:
                        row_vals[safe[k]] = encode(
                            fixed.get(k, f.get("mid", (f["low"]+f["high"])/2)), f)

                # Karesel ve etkileşim terimleri
                for name in safe:
                    row_vals[f"{name}_sq"] = row_vals[name] ** 2
                for a in range(n_factors):
                    for b in range(a+1, n_factors):
                        col = f"{safe[a]}_x_{safe[b]}"
                        row_vals[col] = row_vals[safe[a]] * row_vals[safe[b]]

                row_df  = pd.DataFrame([row_vals])
                row_mat = _sm.add_constant(row_df, has_constant="add")
                # Modelin beklediği sütunlarla hizala
                for col in param_idx:
                    if col not in row_mat.columns:
                        row_mat[col] = 0.0
                row_mat = row_mat.reindex(columns=param_idx, fill_value=0.0)
                try:
                    ZZ[r, c] = float(model.predict(row_mat)[0])
                except Exception:
                    ZZ[r, c] = np.nan

        return XX, YY, ZZ, x_real, y_real

    def _plot(self):
        if not self.project.model_results:
            QMessageBox.warning(self, "", "Önce Model sekmesinden model kurun.")
            return

        resp_key = self.resp_combo.currentData()
        xi = self.x_combo.currentData()
        yi = self.y_combo.currentData()

        if resp_key is None or xi is None or yi is None:
            return
        if xi == yi:
            QMessageBox.warning(self, "", "X ve Y ekseni farklı faktörler olmalıdır.")
            return
        if resp_key not in self.project.model_results:
            QMessageBox.warning(self, "", "Bu yanıt için model bulunamadı.")
            return

        self.btn_plot.setEnabled(False)
        self.btn_plot.setText("⏳ Hesaplanıyor...")
        self.app.status_bar.showMessage("Yüzey hesaplanıyor...")
        QApplication.processEvents()

        try:
            n   = self.res_spin.value()
            XX, YY, ZZ, x_real, y_real = self._predict_grid(resp_key, xi, yi, n)
            self._draw(resp_key, xi, yi, XX, YY, ZZ)
            self.status_lbl.setText(
                f"✅  {RESPONSE_LABELS.get(resp_key,resp_key)}  |  "
                f"X: {self.project.factors[xi]['name']}  |  "
                f"Y: {self.project.factors[yi]['name']}")
            self.app.status_bar.showMessage("Grafik hazır.", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Grafik çizilemedi:\n{e}")
        finally:
            self.btn_plot.setEnabled(True)
            self.btn_plot.setText("▶  Grafik Çiz")

    def _draw(self, resp_key, xi, yi, XX, YY, ZZ):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.cm as cm

        factors = self.project.factors
        fx = factors[xi]; fy = factors[yi]
        resp_label = RESPONSE_LABELS.get(resp_key, resp_key)

        self.fig.clear()
        self.fig.patch.set_facecolor(BG2)

        # Sol: 3D yüzey
        ax3d = self.fig.add_subplot(1, 2, 1, projection='3d')
        ax3d.set_facecolor(BG3)
        ax3d.tick_params(colors=TXT2, labelsize=7)
        ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor("#2a4060")
        ax3d.yaxis.pane.set_edgecolor("#2a4060")
        ax3d.zaxis.pane.set_edgecolor("#2a4060")

        surf = ax3d.plot_surface(XX, YY, ZZ,
                                  cmap="viridis", alpha=0.85,
                                  linewidth=0, antialiased=True)
        ax3d.set_xlabel(f"{fx['name']}" + (f"\n({fx['unit']})" if fx.get("unit") else ""),
                        color=TXT2, fontsize=8, labelpad=8)
        ax3d.set_ylabel(f"{fy['name']}" + (f"\n({fy['unit']})" if fy.get("unit") else ""),
                        color=TXT2, fontsize=8, labelpad=8)
        ax3d.set_zlabel(resp_label, color=TXT2, fontsize=8, labelpad=8)
        ax3d.set_title("3D Yüzey", color=GOLD, fontsize=10, pad=10)

        cb3d = self.fig.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1)
        cb3d.ax.tick_params(colors=TXT2, labelsize=7)

        # Sağ: kontur haritası
        ax2d = self.fig.add_subplot(1, 2, 2)
        ax2d.set_facecolor(BG3)
        ax2d.tick_params(colors=TXT2, labelsize=8)
        for spine in ax2d.spines.values():
            spine.set_edgecolor("#2a4060")

        levels = 20
        cf = ax2d.contourf(XX, YY, ZZ, levels=levels, cmap="viridis", alpha=0.9)
        cs = ax2d.contour(XX, YY, ZZ, levels=levels, colors="white",
                           alpha=0.25, linewidths=0.5)
        ax2d.clabel(cs, inline=True, fontsize=6, colors="white", fmt="%.2f")

        cb2d = self.fig.colorbar(cf, ax=ax2d)
        cb2d.ax.tick_params(colors=TXT2, labelsize=7)
        cb2d.set_label(resp_label, color=TXT2, fontsize=8)

        # Optimum noktayı işaretle
        if self.project.opt_solutions:
            best = self.project.opt_solutions[0]
            ox = best.get("factors", {}).get(self.project.factors[xi]["name"])
            oy = best.get("factors", {}).get(self.project.factors[yi]["name"])
            if ox is not None and oy is not None:
                ax2d.plot(ox, oy, "*", color=GOLD, markersize=14,
                          zorder=5, label="Optimum")
                ax2d.legend(fontsize=8, labelcolor=TXT2,
                             facecolor=BG2, edgecolor="#2a4060")

        # Gerçek ölçüm noktaları
        res = self.project.model_results.get(resp_key, {})
        y_data = res.get("y")
        if y_data is not None and self.project.design_matrix is not None:
            df = self.project.design_matrix
            xs_real, ys_real = [], []
            for ri in range(len(df)):
                if self.project.run_results.get(ri, {}).get(resp_key) is not None:
                    xs_real.append(df.iloc[ri][fx["name"]])
                    ys_real.append(df.iloc[ri][fy["name"]])
            if xs_real:
                ax2d.scatter(xs_real, ys_real, c="white", s=30,
                             zorder=4, edgecolors="#2a4060", linewidths=0.5,
                             alpha=0.8, label="Ölçüm")

        ax2d.set_xlabel(f"{fx['name']}" + (f" ({fx['unit']})" if fx.get("unit") else ""),
                        color=TXT2, fontsize=9)
        ax2d.set_ylabel(f"{fy['name']}" + (f" ({fy['unit']})" if fy.get("unit") else ""),
                        color=TXT2, fontsize=9)
        ax2d.set_title("Kontur Haritası", color=GOLD, fontsize=10)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER SEKMELER (Adım 5-7 için)
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

        # Sekme 2 — Veri Girişi
        self.tab2 = Tab2_DataEntry(self.project, self)
        self.tabs.addTab(self.tab2, "2 · Veri Girişi")

        # Sekme 3 — Model
        self.tab3 = Tab3_Model(self.project, self)
        self.tab3.model_ready.connect(lambda: None)
        self.tabs.addTab(self.tab3, "3 · Model")

        # Sekme 4 — Response Surface
        self.tab4 = Tab4_ResponseSurface(self.project, self)
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
def write_log(msg):
    """Hata logunu masaüstüne yazar."""
    import traceback, datetime
    try:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        log_path = os.path.join(desktop, "optimizer_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"{datetime.datetime.now()}\n")
            f.write(msg + "\n")
    except Exception:
        pass

def exception_hook(exc_type, exc_value, exc_tb):
    import traceback
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    write_log(msg)
    try:
        QMessageBox.critical(None, "Hata",
            f"Beklenmedik hata oluştu:\n\n{exc_value}\n\n"
            f"Detaylar masaüstündeki optimizer_log.txt dosyasına kaydedildi.")
    except Exception:
        pass

def main():
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)

    # Qt exception handler
    def qt_msg_handler(mode, context, message):
        if "error" in message.lower() or "critical" in message.lower():
            write_log(f"Qt mesajı: {message}")
    from PyQt6.QtCore import qInstallMessageHandler
    qInstallMessageHandler(qt_msg_handler)

    window = FormulasyonOptimizerApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
