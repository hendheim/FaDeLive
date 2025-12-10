# file: src/tools_visualisations/gui_tag_topic_explorer.py
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt

# -----------------------------
# Projektpfade & Struktur
# -----------------------------
def project_root() -> Path:
    try:
        here = Path(__file__).resolve()
        candidate = here.parents[2]
    except NameError:
        candidate = Path.cwd()
    if (candidate / "output").exists() and (candidate / "resources").exists():
        return candidate
    if (candidate.parent / "output").exists() and (candidate.parent / "resources").exists():
        return candidate.parent
    return candidate

PROJECT_ROOT = project_root()
OUTPUT_DIR = PROJECT_ROOT / "output"
RESOURCES_DIR = PROJECT_ROOT / "resources"
EXPLORATION_DIR = OUTPUT_DIR / "exploration"
EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# File Discovery System
# -----------------------------
class FileDiscovery:
    """
    Auto-Discovery für Projektdateien basierend auf Ordnerstruktur und Pattern-Matching.
    
    Unterstützt flexible Ordnerstrukturen:
    - termsets/: Dateiende fix (z.B. "_2.3.csv"), Anfang variabel
    - topic-models/: Dateiname fix, Unterordner variabel
    - processed_termset/: Dateiende fix (z.B. "_2.3.csv"), Anfang variabel
    """
    
    def __init__(self, base_dir: Path, termset_suffix: str = "_2.3.csv"):
        self.base_dir = base_dir
        self.termset_suffix = termset_suffix
        self.found_files: Dict[str, Optional[Path]] = {}
    
    def scan_project(self) -> Dict[str, Optional[Path]]:
        """Scannt Projekt und findet alle relevanten Dateien"""
        self.found_files = {
            # Termsets (Dateiende fix: *{suffix})
            'termset': self._find_in_termsets(),
            'topic_words': self._find_topic_words(),
            'tfidf': self._find_tfidf(),
            
            # Processed termset (Dateiende fix: *{suffix})
            'ranks': self._find_in_processed_termset('rank', 'tag', 'topic'),
            'relevance': self._find_in_processed_termset('relevance'),
            'counts_per_year': self._find_in_processed_termset('counts', 'year'),
            'top10_year_value': self._find_in_processed_termset('top10', 'year', 'value'),
            'top10_value_per_text': self._find_in_processed_termset('top10', 'value', 'text'),
            
            # Topic models (Dateiname fix)
            'topics_dist': self._find_topics_dist(),
            
            # Statistics & processed topics
            'tokens_year': self._find_tokens_year(),
            'global_topdocs': self._find_global_topdocs(),
            
            # Metadata
            'metadata': self._find_metadata(),
        }
        return self.found_files
    
    def _find_in_termsets(self) -> Optional[Path]:
        """Findet Termset-Datei in resources/termsets/ (Ende: {suffix})"""
        search_dir = self.base_dir / "resources" / "termsets"
        if not search_dir.exists():
            return None
        
        # Suche *{suffix} (z.B. *_2.3.csv)
        for path in search_dir.glob(f"*{self.termset_suffix}"):
            return path
        return None
    
    def _find_topic_words(self) -> Optional[Path]:
        """
        Findet topic_words in topic-models/.
        
        Struktur: resources/topic-models/topics*/*words*tag*.csv
        Beispiele: topics_100_words_tag.csv, topic_words_tag.csv
        """
        search_dir = self.base_dir / "resources" / "topic-models"
        if not search_dir.exists():
            return None
        
        # Suche in topics* Unterordnern
        for topics_dir in search_dir.glob("topics*"):
            if not topics_dir.is_dir():
                continue
            
            # Suche *words*tag*.csv (flexibler Pattern)
            # Matched: topics_100_words_tag.csv, topic_words_tag.csv, etc.
            for path in topics_dir.glob("*words*tag*.csv"):
                return path
        
        return None
    
    def _find_tfidf(self) -> Optional[Path]:
        """Findet TF-IDF Datei in output/dtm_tfidf*/"""
        search_dir = self.base_dir / "output"
        if not search_dir.exists():
            return None
        
        # Suche in dtm_tfidf* Ordnern
        for tfidf_dir in search_dir.glob("dtm_tfidf*"):
            if tfidf_dir.is_dir():
                for path in tfidf_dir.glob("tfidf-*.csv"):
                    return path
        return None
    
    def _find_in_processed_termset(self, *keywords: str) -> Optional[Path]:
        """
        Findet Datei in processed_termset/ basierend auf Keywords.
        
        Struktur: output/processed_termset/Termset_{suffix}/Termset_{suffix}_keywords.csv
        
        Beispiele mit suffix="_2.3":
          _find_in_processed_termset('rank') 
            → sucht Termset_*_2.3/*rank*.csv
          
          _find_in_processed_termset('counts', 'year')
            → sucht Termset_*_2.3/*counts*year*.csv
        
        WICHTIG: Das Suffix ist Teil des ORDNERNAMEN, nicht des Dateiende!
        """
        search_dir = self.base_dir / "output" / "processed_termset"
        if not search_dir.exists():
            return None
        
        # Suffix ohne führenden Unterstrich für Ordnersuche
        # z.B. "_2.3.csv" → "2.3" oder "_2.3" → "2.3"
        suffix_clean = self.termset_suffix.lstrip('_').replace('.csv', '')
        
        # Suche in Termset_*{suffix} Unterordnern
        # Beispiel: Termset_Begriffe_2.3, Termset_Final_3.1
        pattern_dir = f"Termset*{suffix_clean}"
        
        for termset_dir in search_dir.glob(pattern_dir):
            if not termset_dir.is_dir():
                continue
            
            # Pattern: *keyword1*keyword2*.csv (OHNE Suffix am Ende!)
            pattern = "*" + "*".join(keywords) + "*.csv"
            
            for path in termset_dir.glob(pattern):
                return path
        
        return None
    
    def _find_topics_dist(self) -> Optional[Path]:
        """
        Findet document-topics-distribution_tag.csv in topic-models/ (fester Name).
        
        Struktur: resources/topic-models/topics*/document-topics-distribution_tag.csv
        """
        search_dir = self.base_dir / "resources" / "topic-models"
        if not search_dir.exists():
            return None
        
        # Suche in topics* Unterordnern
        for topics_dir in search_dir.glob("topics*"):
            if not topics_dir.is_dir():
                continue
            
            # Exakter Dateiname
            target = topics_dir / "document-topics-distribution_tag.csv"
            if target.exists():
                return target
        
        return None
    
    def _find_tokens_year(self) -> Optional[Path]:
        """Findet tokens-per-year in statistics/"""
        search_dir = self.base_dir / "output" / "statistics"
        if not search_dir.exists():
            return None
        
        for path in search_dir.glob("*tokens*.csv"):
            if 'year' in path.name.lower():
                return path
        return None
    
    def _find_global_topdocs(self) -> Optional[Path]:
        """Findet global topdocs in processed_topics/"""
        search_dir = self.base_dir / "output" / "processed_topics"
        if not search_dir.exists():
            return None
        
        for path in search_dir.glob("*topdocs*year*.csv"):
            return path
        return None
    
    def _find_metadata(self) -> Optional[Path]:
        """Findet metadata.csv (Standard-Orte)"""
        candidates = [
            self.base_dir / "data" / "raw" / "metadata.csv",
            self.base_dir / "data" / "metadata.csv",
            self.base_dir / "metadata.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None
    
    def get_status_report(self) -> str:
        """Erstellt Status-Report für UI"""
        report = []
        report.append(f"📁 Arbeitsordner: {self.base_dir}\n")
        report.append(f"🔖 Termset-Suffix: {self.termset_suffix}\n\n")
        
        categories = {
            "Termsets": ['termset', 'topic_words', 'tfidf'],
            "Processed Termset": ['ranks', 'relevance', 'counts_per_year', 'top10_year_value', 'top10_value_per_text'],
            "Topic Models": ['topics_dist'],
            "Statistics": ['tokens_year', 'global_topdocs'],
            "Metadata": ['metadata'],
        }
        
        for category, keys in categories.items():
            report.append(f"{category}:\n")
            for key in keys:
                path = self.found_files.get(key)
                if path:
                    rel_path = path.relative_to(self.base_dir) if path else "?"
                    report.append(f"  ✅ {key}: {rel_path}\n")
                else:
                    report.append(f"  ❌ {key}: nicht gefunden\n")
            report.append("\n")
        
        return "".join(report)
    
    def get_missing_files(self) -> List[str]:
        """Liste der fehlenden Dateien"""
        return [key for key, path in self.found_files.items() if path is None]

# Metadatenbereich
METADATA_COLS: int = 24
META_NAME_BLACKLIST = {
    "_id","id","doc_id","filename",
    "author","author_surname","author_surname_norm",
    "title","title_norm","source","journal","magazine",
    "year","year_first","year_final","Jahr_final",
    "textclass","address","address_author","lang","language"
}

# -----------------------------
# Matplotlib Style (kompakt)
# -----------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "regular",
    "axes.labelsize": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.figsize": (12, 6),
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

def _apply_layout(fig: plt.Figure) -> None:
    """Verhindert abgeschnittene Labels."""
    try:
        if hasattr(fig, "set_constrained_layout"):
            fig.set_constrained_layout(True)
            fig.canvas.draw_idle()
        else:
            fig.tight_layout()
    except Exception:
        fig.tight_layout()

# -----------------------------
# Utils
# -----------------------------
_nonword = re.compile(r"[^\w\-.,+]+")
def _norm_ctx(s: str) -> str:
    return _nonword.sub("_", s).strip("_") or "export"

def coalesce_years(df: pd.DataFrame, col_year_first="year_first", col_year="year", out="year_final") -> pd.DataFrame:
    to_num = lambda s: pd.to_numeric(s, errors="coerce")
    yf = to_num(df[col_year_first]) if col_year_first in df.columns else pd.Series(index=df.index, dtype="float64")
    y  = to_num(df[col_year]) if col_year in df.columns else pd.Series(index=df.index, dtype="float64")
    df[out] = yf.where(~yf.isna(), y)
    return df

def term_columns(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    cand = cols[METADATA_COLS:] if len(cols) > METADATA_COLS else []
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    def is_term(c: str) -> bool:
        cl = str(c).strip()
        return cl not in META_NAME_BLACKLIST
    return [c for c in dict.fromkeys(list(cand) + numeric) if is_term(c)]

def ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def parse_year_range(default_min: int, default_max: int, raw: str) -> Tuple[int,int]:
    raw = (raw or "").strip()
    if "-" in raw:
        a, b = raw.split("-", 1)
        try:
            lo, hi = int(a.strip()), int(b.strip())
            if lo <= hi:
                return lo, hi
        except Exception:
            pass
    return default_min, default_max

def ask_save_df(df: Optional[pd.DataFrame], tab: str, ctx: str, parent: tk.Tk) -> None:
    if df is None or df.empty:
        messagebox.showinfo("Info", "Keine Daten zum Speichern.", parent=parent); return
    try:
        initdir = EXPLORATION_DIR / tab; initdir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=parent, title="Als CSV speichern",
            initialdir=str(initdir),
            initialfile=f"{_norm_ctx(ctx)}.csv",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("Alle Dateien","*.*")]
        )
        if not path: return
        df.to_csv(path, index=False)
        messagebox.showinfo("Gespeichert", path, parent=parent)
    except Exception as e:
        messagebox.showerror("Fehler beim Speichern", str(e), parent=parent)

def ask_save_current_figure(tab: str, ctx: str, parent: tk.Tk, fig: Optional[plt.Figure] = None, dpi: int = 300) -> None:
    try:
        initdir = EXPLORATION_DIR / tab; initdir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=parent, title="Als PNG speichern",
            initialdir=str(initdir),
            initialfile=f"{_norm_ctx(ctx)}.png",
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("Alle Dateien","*.*")]
        )
        if not path: return
        (fig or plt.gcf()).savefig(path, dpi=dpi, bbox_inches="tight")
        messagebox.showinfo("Gespeichert", path, parent=parent)
    except Exception as e:
        messagebox.showerror("Fehler beim Speichern", str(e), parent=parent)

# -----------------------------
# Tk-Utils
# -----------------------------
def safe_exit_tk(root: tk.Tk) -> None:
    try:
        for w in list(root.winfo_children()):
            try: w.destroy()
            except Exception: pass
        root.quit()
    except Exception:
        pass
    try: root.destroy()
    except Exception: pass

def install_safe_exit(root: tk.Tk) -> None:
    root.protocol("WM_DELETE_WINDOW", lambda: safe_exit_tk(root))
    root.bind("<Escape>", lambda _e: safe_exit_tk(root), add="+")

def bring_front(win: tk.Toplevel | tk.Tk) -> None:
    win.update_idletasks()
    try: win.attributes("-topmost", False)
    except Exception: pass
    try: win.lift()
    except Exception: pass
    try: win.focus_force()
    except Exception: pass

def install_focus_minimize(root: tk.Tk, enable: bool = True) -> None:
    if not enable:
        return
    BUTTON_MASK = 0x100 | 0x200 | 0x400
    def _on_focus_out(event):
        try:
            focus_inside = (root.focus_displayof() is not None)
            st = getattr(event, "state", 0)
            mouse_down = bool(st & BUTTON_MASK)
            if (not focus_inside) and mouse_down:
                root.iconify()
        except Exception:
            pass
    root.bind("<FocusOut>", _on_focus_out, add="+")


def _mk_entry(parent, **kwargs):
    e = ttk.Entry(parent, **kwargs)
    try: e.configure(state="normal", takefocus=True)
    except Exception: pass
    return e

def _shrink_axes(ax):
    ax.tick_params(axis='x', labelrotation=45, labelsize=8)
    ax.margins(x=0.02)
    for label in ax.get_xticklabels():
        label.set_ha('right')

# -----------------------------
# DataManager (Lazy; Termset lowercase)
# -----------------------------
class DataManager:
    def __init__(self) -> None:
        # Termset↔Topic-Abbildungen (Defaults)
        self.path_termset: Path = RESOURCES_DIR / "termsets" / "Termset_Begriffe_2.3.csv"
        self.path_topic_words: Path = RESOURCES_DIR / "topic-models" / "topics_v3" / "fadelive_mallet_stop_topic_words_100_words_tag.csv"
        self.path_tfidf: Path = OUTPUT_DIR / "dtm_tfidf_stop" / "tfidf-2000.csv"
        self.path_ranks: Path = OUTPUT_DIR / "processed_termset" / "Termset_Begriffe_2.3" / "Termset_Begriffe_2.3_tag_topic_rank.csv"
        self.path_relevance: Path = OUTPUT_DIR / "processed_termset" / "Termset_Begriffe_2.3" / "Termset_Begriffe_2.3_tag_topic_relevance.csv"
        self.path_counts_per_year: Path = OUTPUT_DIR / "processed_termset" / "Termset_Begriffe_2.3" / "Termset_Begriffe_2.3_dtti_topdocs_topic_counts_per_year.csv"
        self.path_top10_year_value: Path = OUTPUT_DIR / "processed_termset" / "Termset_Begriffe_2.3" / "Termset_Begriffe_2.3_dtti_topdocs_top10_year_value.csv"
        self.path_top10_value_per_text_topic: Path = OUTPUT_DIR / "processed_termset" / "Termset_Begriffe_2.3" / "Termset_Begriffe_2.3_dtti_topdocs_top10_value_per_text_topic.csv"
        # Topic-only Visuals + Verläufe
        self.path_tokens_year: Path = OUTPUT_DIR / "statistics" / "year_count_tokens.csv"
        self.path_global_topdocs_year: Path = OUTPUT_DIR / "processed_topics" / "document-topics-distribution_tag_topdocs_year_value.csv"
        self.path_topics: Path = RESOURCES_DIR / "topic-models" / "topics_v3" / "document-topics-distribution_tag.csv"
        self.path_metadata: Path = PROJECT_ROOT / "data" / "raw" / "metadata.csv"

        self._df_tags: Optional[pd.DataFrame] = None
        self._df_topics: Optional[pd.DataFrame] = None
        self._df_tfidf: Optional[pd.DataFrame] = None
        self._df_ranks: Optional[pd.DataFrame] = None
        self._df_relevance: Optional[pd.DataFrame] = None
        self._df_counts_per_year: Optional[pd.DataFrame] = None
        self._df_top10_year_value: Optional[pd.DataFrame] = None
        self._df_top10_value_per_text_topic: Optional[pd.DataFrame] = None
        self._df_tokens_year: Optional[pd.DataFrame] = None
        self._df_global_topdocs_year: Optional[pd.DataFrame] = None
        self._df_topics_dist: Optional[pd.DataFrame] = None
        self._df_metadata: Optional[pd.DataFrame] = None
        
        # File Discovery
        self.current_base_dir: Path = PROJECT_ROOT
        self.current_termset_suffix: str = "_2.3.csv"

    def auto_discover_files(self, base_dir: Path, termset_suffix: str) -> Dict[str, Optional[Path]]:
        """
        Automatisches Finden aller Dateien basierend auf Ordnerstruktur.
        
        Args:
            base_dir: Basis-Projektordner
            termset_suffix: Suffix für Termset-Dateien (z.B. "_2.3.csv")
        
        Returns:
            Dictionary mit gefundenen Pfaden
        """
        self.current_base_dir = base_dir
        self.current_termset_suffix = termset_suffix
        
        discovery = FileDiscovery(base_dir, termset_suffix)
        found = discovery.scan_project()
        
        # Setze gefundene Pfade (invalidiert Caches)
        if found.get('termset'):
            self.set_termset(found['termset'])
        if found.get('topic_words'):
            self.set_topic_words(found['topic_words'])
        if found.get('tfidf'):
            self.set_tfidf(found['tfidf'])
        if found.get('ranks'):
            self.set_ranks(found['ranks'])
        if found.get('relevance'):
            self.set_relevance(found['relevance'])
        if found.get('counts_per_year'):
            self.set_counts_per_year(found['counts_per_year'])
        if found.get('top10_year_value'):
            self.set_top10_year_value(found['top10_year_value'])
        if found.get('top10_value_per_text'):
            self.set_top10_value_per_text_topic(found['top10_value_per_text'])
        if found.get('tokens_year'):
            self.set_tokens_year(found['tokens_year'])
        if found.get('global_topdocs'):
            self.set_global_topdocs_year(found['global_topdocs'])
        if found.get('topics_dist'):
            self.set_topics(found['topics_dist'])
        if found.get('metadata'):
            self.set_metadata(found['metadata'])
        
        return found
    
    def get_discovery_report(self) -> str:
        """Status-Report für UI"""
        discovery = FileDiscovery(self.current_base_dir, self.current_termset_suffix)
        discovery.found_files = {
            'termset': self.path_termset if self.path_termset.exists() else None,
            'topic_words': self.path_topic_words if self.path_topic_words.exists() else None,
            'tfidf': self.path_tfidf if self.path_tfidf.exists() else None,
            'ranks': self.path_ranks if self.path_ranks.exists() else None,
            'relevance': self.path_relevance if self.path_relevance.exists() else None,
            'counts_per_year': self.path_counts_per_year if self.path_counts_per_year.exists() else None,
            'top10_year_value': self.path_top10_year_value if self.path_top10_year_value.exists() else None,
            'top10_value_per_text': self.path_top10_value_per_text_topic if self.path_top10_value_per_text_topic.exists() else None,
            'topics_dist': self.path_topics if self.path_topics.exists() else None,
            'tokens_year': self.path_tokens_year if self.path_tokens_year.exists() else None,
            'global_topdocs': self.path_global_topdocs_year if self.path_global_topdocs_year.exists() else None,
            'metadata': self.path_metadata if self.path_metadata.exists() else None,
        }
        return discovery.get_status_report()

    # Setter (Cache invalidieren)
    def set_termset(self, p: Path) -> None: self.path_termset = p; self._df_tags = None
    def set_topic_words(self, p: Path) -> None: self.path_topic_words = p; self._df_topics = None
    def set_tfidf(self, p: Path) -> None: self.path_tfidf = p; self._df_tfidf = None
    def set_ranks(self, p: Path) -> None: self.path_ranks = p; self._df_ranks = None
    def set_relevance(self, p: Path) -> None: self.path_relevance = p; self._df_relevance = None
    def set_counts_per_year(self, p: Path) -> None: self.path_counts_per_year = p; self._df_counts_per_year = None
    def set_top10_year_value(self, p: Path) -> None: self.path_top10_year_value = p; self._df_top10_year_value = None
    def set_top10_value_per_text_topic(self, p: Path) -> None: self.path_top10_value_per_text_topic = p; self._df_top10_value_per_text_topic = None
    def set_tokens_year(self, p: Path) -> None: self.path_tokens_year = p; self._df_tokens_year = None
    def set_global_topdocs_year(self, p: Path) -> None: self.path_global_topdocs_year = p; self._df_global_topdocs_year = None
    def set_topics(self, p: Path) -> None: self.path_topics = p; self._df_topics_dist = None
    def set_metadata(self, p: Path) -> None: self.path_metadata = p; self._df_metadata = None

    def _require(self, p: Path) -> None:
        if not p.exists():
            raise FileNotFoundError(f"Datei fehlt: {p}")

    def suggest_termset_folder(self) -> Path:
        base_root = OUTPUT_DIR / "processed_termset"
        if not base_root.exists():
            return OUTPUT_DIR
        preferred = base_root / "Termset_Begriffe_2.3"
        if preferred.exists() and preferred.is_dir():
            return preferred
        candidates = [p for p in base_root.glob("Termset_Begriffe_*") if p.is_dir()]
        if not candidates:
            return base_root
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    # Loader
    def load_termset(self) -> pd.DataFrame:
        if self._df_tags is None:
            self._require(self.path_termset)
            df = pd.read_csv(self.path_termset)
            def _lower_cell(x):
                if pd.isna(x): return x
                return str(x).strip().lower()
            self._df_tags = df.applymap(_lower_cell)
        return self._df_tags

    def load_topic_words(self) -> pd.DataFrame:
        if self._df_topics is None:
            self._require(self.path_topic_words)
            df = pd.read_csv(self.path_topic_words, index_col=0)
            self._df_topics = df.applymap(lambda x: str(x).strip().lower() if pd.notna(x) else x)
        return self._df_topics

    def load_tfidf(self) -> pd.DataFrame:
        if self._df_tfidf is None:
            self._require(self.path_tfidf)
            self._df_tfidf = pd.read_csv(self.path_tfidf)
        return self._df_tfidf

    def load_ranks(self) -> pd.DataFrame:
        if self._df_ranks is None:
            self._require(self.path_ranks)
            self._df_ranks = pd.read_csv(self.path_ranks)
        return self._df_ranks

    def load_relevance(self) -> pd.DataFrame:
        if self._df_relevance is None:
            self._require(self.path_relevance)
            self._df_relevance = pd.read_csv(self.path_relevance)
        return self._df_relevance

    def load_counts_per_year(self) -> pd.DataFrame:
        if self._df_counts_per_year is None:
            self._require(self.path_counts_per_year)
            self._df_counts_per_year = pd.read_csv(self.path_counts_per_year, index_col=0)
        return self._df_counts_per_year

    def load_top10_year_value(self) -> pd.DataFrame:
        if self._df_top10_year_value is None:
            self._require(self.path_top10_year_value)
            self._df_top10_year_value = pd.read_csv(self.path_top10_year_value)
        return self._df_top10_year_value

    def load_top10_value_per_text_topic(self) -> pd.DataFrame:
        if self._df_top10_value_per_text_topic is None:
            self._require(self.path_top10_value_per_text_topic)
            self._df_top10_value_per_text_topic = pd.read_csv(self.path_top10_value_per_text_topic)
        return self._df_top10_value_per_text_topic

    def load_tokens_year(self) -> pd.DataFrame:
        if self._df_tokens_year is None:
            self._require(self.path_tokens_year)
            self._df_tokens_year = pd.read_csv(self.path_tokens_year)
        return self._df_tokens_year

    def load_global_topdocs_year(self) -> pd.DataFrame:
        if self._df_global_topdocs_year is None:
            self._require(self.path_global_topdocs_year)
            self._df_global_topdocs_year = pd.read_csv(self.path_global_topdocs_year)
        return self._df_global_topdocs_year

    def load_topics_dist(self) -> pd.DataFrame:
        if self._df_topics_dist is None:
            self._require(self.path_topics)
            self._df_topics_dist = pd.read_csv(self.path_topics, index_col=0)
            self._df_topics_dist.index = self._df_topics_dist.index.astype(str).str.replace(".txt", "", regex=False)
        return self._df_topics_dist

    def load_metadata(self) -> pd.DataFrame:
        if self._df_metadata is None:
            self._require(self.path_metadata)
            df = pd.read_csv(self.path_metadata, sep=";")
            df["_id"] = df["_id"].astype(str)
            df = coalesce_years(df, "year_first", "year", "Jahr_final")
            self._df_metadata = df
        return self._df_metadata

DATA = DataManager()

# -----------------------------
# Feature-Helper
# -----------------------------
def tag_dict_from_df(df_tags: pd.DataFrame) -> Dict[str, List[str]]:
    return {col: df_tags[col].dropna().astype(str).str.strip().tolist() for col in df_tags.columns}

def topic_word_map_from_df(df_topics: pd.DataFrame) -> Dict[str, List[str]]:
    return {str(topic): df_topics.loc[topic].dropna().astype(str).str.strip().tolist() for topic in df_topics.index}

def tfidf_series_sum(DATA: DataManager) -> pd.Series:
    df = DATA.load_tfidf()
    cols = term_columns(df)
    if not cols:
        raise ValueError("Keine Ausdrucksspalten (TF-IDF) erkannt.")
    s = df[cols].sum(numeric_only=True)
    s.index = s.index.astype(str).str.strip().str.lower()
    return s

def topic_labels_ordered_from_rank(df_ranks: pd.DataFrame, topics: Iterable[str]) -> List[str]:
    r = df_ranks.copy()
    if "Topic" not in r.columns or "TFIDF-Positions-Rang" not in r.columns:
        raise ValueError("Ranking-CSV benötigt Spalten: Topic, TFIDF-Positions-Rang.")
    r = r.assign(Topic=r["Topic"].astype(str),
                 rank=pd.to_numeric(r["TFIDF-Positions-Rang"], errors="coerce")).dropna(subset=["rank"])
    r = r[r["Topic"].isin(list(topics))].sort_values("rank")
    return r["Topic"].astype(str).str.split("(", n=1).str[0].str.strip().tolist()

# -----------------------------
# Tag-Topic-Exploration
# -----------------------------
def build_tab_bubbles_rank_topn(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="TT-Relevanz (Bubbles)")
    row = 0
    ttk.Label(frame, text="Top-N Topics (Rang):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_topn = _mk_entry(frame, width=8); ent_topn.insert(0, "10"); ent_topn.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="Bubble-Skalierung:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_scale = _mk_entry(frame, width=8); ent_scale.insert(0, "3"); ent_scale.grid(row=row, column=3, sticky="w", padx=6, pady=4)

    row += 1
    btn_csv = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled")
    btn_csv.grid(row=row, column=2, sticky="e", padx=6, pady=6)
    btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)

    out_df: Optional[pd.DataFrame] = None
    ctx = {"v": ""}

    def run():
        nonlocal out_df
        try:
            df_tags = DATA.load_termset()
            df_topics = DATA.load_topic_words()
            df_ranks = DATA.load_ranks()
            tfidf_sum = tfidf_series_sum(DATA)
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return

        r = (df_ranks.assign(Topic=df_ranks["Topic"].astype(str),
                             rank=pd.to_numeric(df_ranks["TFIDF-Positions-Rang"], errors="coerce"))
                      .dropna(subset=["rank"]).sort_values("rank"))

        try:
            n = max(1, int((ent_topn.get() or "10").strip()))
            scale = float((ent_scale.get() or "3").strip())
        except Exception:
            messagebox.showerror("Fehler", "Ungültige Parameter (Top-N / Skalierung).", parent=root); return

        topics_available = set(df_topics.index.astype(str))
        top_topics_ordered = [t for t in r["Topic"].tolist() if t in topics_available][:n]
        if not top_topics_ordered:
            messagebox.showerror("Fehler", "Keine passenden Topics in Rangdatei.", parent=root); return

        tag_dict = tag_dict_from_df(df_tags)
        topic_map = topic_word_map_from_df(df_topics)
        tfidf_dict = tfidf_sum.to_dict()

        rows = []
        for topic in top_topics_ordered:
            topic_words = set(topic_map[topic])
            for tag, expr in tag_dict.items():
                common = topic_words.intersection(expr)
                val = float(sum(tfidf_dict.get(w, 0.0) for w in common)) if common else 0.0
                rows.append({"Topic": topic, "Tag": tag, "tfidf_sum": val})

        df_top = pd.DataFrame(rows)
        df_top["Topic_Label"] = df_top["Topic"].astype(str).str.split("(", n=1).str[0].str.strip()
        df_top["rank"] = df_top["Topic"].map(dict(zip(r["Topic"], r["rank"])))
        df_top = df_top.sort_values(["rank", "Tag"])
        ordered_labels = topic_labels_ordered_from_rank(r, top_topics_ordered)
        df_top["Topic_Label"] = pd.Categorical(df_top["Topic_Label"], categories=ordered_labels, ordered=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        base = df_top["tfidf_sum"].clip(lower=0).to_numpy()
        sizes = (base * scale) + 10.0
        ax.scatter(df_top["Topic_Label"], df_top["Tag"], s=sizes, color="sienna", alpha=0.6, edgecolors="black")
        ax.set_xlabel("Topic (TFIDF-Positions-Rang)")
        ax.set_ylabel("Tag")
        ax.set_title(f"Top {len(ordered_labels)} Topics – TT-Relevanz (TF-IDF-Overlap)")
        _shrink_axes(ax); _apply_layout(fig); plt.show()

        out_df = df_top.rename(columns={"tfidf_sum": "value"})
        ctx["v"] = f"tt_relevanz_bubbles_top{len(ordered_labels)}"
        btn_csv.configure(state="normal", command=lambda: ask_save_df(out_df, "TT_Relevanz_Bubbles", ctx["v"], root))
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("TT_Relevanz_Bubbles", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Berechnen", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)


def build_tab_tag_relevance(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="Tag-Topic-Relevancescore")
    row = 0
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled"); btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    ctx = {"v": "tag_topic_relevancescore"}

    def run():
        try:
            df = DATA.load_relevance()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return
        needed = {"Tag","Relevanzscore_Tag_Topic","Relevanzscore_Tag_TFIDF"}
        if not needed.issubset(df.columns):
            messagebox.showerror("Fehler", f"Spalten fehlen: {needed}", parent=root); return

        d = df.sort_values("Relevanzscore_Tag_TFIDF", ascending=True).reset_index(drop=True)
        fig, ax1 = plt.subplots(figsize=(12, 5.5))
        ax1.bar(d["Tag"], d["Relevanzscore_Tag_Topic"], color="sienna", alpha=0.9)
        ax1.set_ylabel("Relevanzscore_Tag_Topic")
        ax1.set_xticks(range(len(d))); ax1.set_xticklabels(d["Tag"], rotation=45, ha='right', fontsize=8)
        ax1.tick_params(axis='y', labelsize=8)

        ax2 = ax1.twinx()
        ax2.plot(range(len(d)), d["Relevanzscore_Tag_TFIDF"], color="black", marker="o", markersize=3, linewidth=1)
        ax2.set_ylabel("Relevanzscore_Tag_TFIDF")
        ax2.tick_params(axis='y', labelsize=8)
        plt.title("Tag-Topic-Relevancescore vs. TF-IDF-Sum", fontsize=12)
        _apply_layout(fig); plt.show()

        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("Tag_Topic_Relevancescore", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)


def _ranked_topics_by_r(df_counts: pd.DataFrame, df_ranks: pd.DataFrame, topn: int) -> List[str]:
    """
    Findet Top-N Topics aus df_ranks die in df_counts vorkommen.
    Verwendet flexible Matching-Strategien für verschiedene Namensformate.
    """
    topics_in_df = set(df_counts.columns.astype(str))
    
    r = (df_ranks.assign(Topic=df_ranks["Topic"].astype(str),
                         rank=pd.to_numeric(df_ranks["TFIDF-Positions-Rang"], errors="coerce"))
                  .dropna(subset=["rank"]).sort_values("rank"))
    
    ranked = []
    for topic in r["Topic"].tolist():
        if len(ranked) >= topn:
            break
        
        topic_str = str(topic).strip()
        
        # Strategie 1: Exakte Übereinstimmung
        if topic_str in topics_in_df:
            ranked.append(topic_str)
            continue
        
        # Strategie 2: Mit "Topic " Präfix
        topic_with_prefix = f"Topic {topic_str}"
        if topic_with_prefix in topics_in_df:
            ranked.append(topic_with_prefix)
            continue
        
        # Strategie 3: Ohne "Topic " Präfix (falls topic_str="Topic 1")
        if topic_str.startswith("Topic "):
            topic_without_prefix = topic_str.replace("Topic ", "", 1)
            if topic_without_prefix in topics_in_df:
                ranked.append(topic_without_prefix)
                continue
        
        # Strategie 4: Nur Nummer extrahieren und Varianten probieren
        import re
        match = re.search(r'\d+', topic_str)
        if match:
            num = match.group()
            for variant in [f"Topic {num}", num, f"topic_{num}", f"topic{num}"]:
                if variant in topics_in_df:
                    ranked.append(variant)
                    break
    
    # Fallback: Wenn keine Topics gefunden wurden, nehme Top-N nach Summe
    if not ranked:
        counts = df_counts.sum(axis=0).sort_values(ascending=False)
        ranked = counts.head(topn).index.tolist()
    
    return ranked


def _normalize_id(id_str: str) -> str:
    """
    Normalisiert verschiedene ID-Formate zu einer Zahl.
    
    Beispiele:
      "doc_001.txt" → "1"
      "text_123" → "123"
      "456" → "456"
    """
    import re
    # Entferne Dateiendungen
    id_str = re.sub(r'\.(txt|csv|pdf|xml|html)$', '', str(id_str), flags=re.IGNORECASE)
    # Entferne Präfixe wie "doc_", "text_", etc.
    id_str = re.sub(r'^(doc|text|document|file)[-_\s]*', '', id_str, flags=re.IGNORECASE)
    # Entferne führende Nullen: "001" → "1"
    id_str = id_str.lstrip('0') or '0'
    # Extrahiere erste Zahl falls noch Text dabei
    match = re.search(r'\d+', id_str)
    return match.group() if match else id_str


def _find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Findet Spalte in DataFrame anhand möglicher Namen (case-insensitive).
    
    Args:
        df: DataFrame
        possible_names: Liste möglicher Spaltennamen
    
    Returns:
        Gefundener Spaltenname oder None
    """
    df_cols_lower = {col.lower(): col for col in df.columns}
    for name in possible_names:
        name_lower = name.lower()
        if name_lower in df_cols_lower:
            return df_cols_lower[name_lower]
    return None


def build_tab_topics_year_stacked(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="TT-Texts/Jahr (Stacked)")
    row=0
    ttk.Label(frame, text="Jahr von–bis:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_years = _mk_entry(frame, width=12); ent_years.insert(0,"1780-1900"); ent_years.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="Top-N (Rang):").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_topn = _mk_entry(frame, width=8); ent_topn.insert(0,"10"); ent_topn.grid(row=row, column=3, sticky="w", padx=6, pady=4)

    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled"); btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    ctx = {"v": ""}

    def run():
        try:
            df_counts = DATA.load_counts_per_year()
            df_ranks = DATA.load_ranks()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return

        df = df_counts.drop(columns=["Anzahl Topics"], errors="ignore")
        df.index = df.index.astype(int)

        ymin, ymax = parse_year_range(int(df.index.min()), int(df.index.max()), ent_years.get())
        rng = pd.Index(range(ymin, ymax+1), name="Jahr")
        df = df.reindex(rng, fill_value=0)

        try:
            topn = max(1, int((ent_topn.get() or "10").strip()))
        except Exception:
            topn = 10

        ranked_topics = _ranked_topics_by_r(df, df_ranks, topn)
        if not ranked_topics:
            messagebox.showerror("Fehler", "Keine Topics aus Rangdatei in den Counts gefunden.", parent=root); return

        dfp = df[ranked_topics]

        fig, ax = plt.subplots(figsize=(12, 5.8))
        colors = plt.cm.tab20(np.linspace(0, 1, len(dfp.columns)))
        dfp.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.86)
        rolling = dfp.sum(axis=1).rolling(window=5, center=True, min_periods=1).mean()
        ax.plot(range(len(dfp)), rolling.values, color="black", linestyle="--", linewidth=1, alpha=0.75, label="Gleitender Mittelwert")
        ticks = [i for i, y in enumerate(dfp.index) if int(y) % 10 == 0]
        ax.set_xticks(ticks); ax.set_xticklabels([dfp.index[i] for i in ticks], rotation=45, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel("Jahr"); ax.set_ylabel("Anzahl TT-Texts (Top-50) pro Topic")
        ax.set_title("TT-Texts/Jahr (Stacked) – Top-N Topics nach TFIDF-Positions-Rang", fontsize=12)

        # Legende in Rangreihenfolge
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=[lab.replace("(", "\n(") for lab in labels],
                  bbox_to_anchor=(1.02, 0.5), loc="center left", title="Topics", fontsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"tt_texts_year_stacked_{ymin}_{ymax}_top{topn}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("TT_Texts_Year_Stacked", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)
    
def build_tab_topics_year_poly(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="TT-Texts/Jahr (Poly)")
    row=0
    ttk.Label(frame, text="Polynom-Grad:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_deg = _mk_entry(frame, width=8); ent_deg.insert(0,"6"); ent_deg.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="Top-N (Rang):").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_topn = _mk_entry(frame, width=8); ent_topn.insert(0,"10"); ent_topn.grid(row=row, column=3, sticky="w", padx=6, pady=4)

    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled"); btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    ctx = {"v": ""}

    def run():
        try:
            df_counts = DATA.load_counts_per_year()
            df_ranks = DATA.load_ranks()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return

        df = df_counts.drop(columns=["Anzahl Topics"], errors="ignore")
        df.index = df.index.astype(int)
        df = df.reindex(range(df.index.min(), df.index.max()+1), fill_value=0)

        try:
            n = max(1, int((ent_topn.get() or "10").strip()))
        except Exception:
            n = 10

        ranked_topics = _ranked_topics_by_r(df, df_ranks, n)
        if not ranked_topics:
            messagebox.showerror("Fehler", "Keine Topics aus Rangdatei in den Counts gefunden.", parent=root); return

        x = df.index.values.astype(float)
        try:
            deg = int((ent_deg.get() or "6").strip())
            deg = max(1, min(deg, len(x)-1))
        except Exception:
            deg = 6

        colors = plt.cm.tab10(np.linspace(0, 1, len(ranked_topics)))
        fig, ax = plt.subplots(figsize=(12, 5.8))
        for i, topic in enumerate(ranked_topics):
            y = df[topic].values.astype(float)
            if len(np.unique(y)) < 2:
                continue
            coeffs = np.polyfit(x, y, deg)
            y_poly = np.polyval(coeffs, x)
            ax.plot(x, y_poly, label=topic.replace("(", "\n("), color=colors[i], linestyle="-", linewidth=1)

        ax.set_xticks([int(j) for j in x.astype(int) if int(j) % 10 == 0])
        ax.set_xlabel("Jahr"); ax.set_ylabel("Anzahl TT-Texts (Top-50) pro Topic")
        ax.set_title(f"TT-Texts/Jahr (Poly, Grad {deg}) – Top-N nach TFIDF-Positions-Rang", fontsize=12)
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", title="Topics", fontsize=8)
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"tt_texts_year_poly_deg{deg}_top{n}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("TT_Texts_Year_Poly", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

def build_tab_series_top10(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="T-Top10-T-Texts/Jahr (Global)")
    row=0
    ttk.Label(frame, text="Jahr von–bis:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_years = _mk_entry(frame, width=12); ent_years.insert(0,"1780-1900"); ent_years.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="Polynom-Grad:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_deg = _mk_entry(frame, width=8); ent_deg.insert(0,"6"); ent_deg.grid(row=row, column=3, sticky="w", padx=6, pady=4)

    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled"); btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    ctx = {"v": ""}

    def run():
        try:
            df = DATA.load_top10_year_value()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return

        ymin, ymax = parse_year_range(int(df["Jahr"].min()), int(df["Jahr"].max()), ent_years.get())
        df = df[(df["Jahr"] >= ymin) & (df["Jahr"] <= ymax)].sort_values("Jahr")

        y = ensure_numeric(df["Wert"])
        if y.notna().sum() == 0:
            messagebox.showinfo("Info", "Keine numerischen Werte.", parent=root); return
        y = y.fillna(0.0)
        y_norm = (y - y.min()) / (y.max() - y.min()) if (y.max() > y.min()) else y * 0
        df = df.assign(Wert=y_norm)
        df["Gleitmittel"] = df["Wert"].rolling(window=5, center=True, min_periods=1).mean()

        x = df["Jahr"].astype(float).values
        try:
            deg = int((ent_deg.get() or "6").strip()); deg = max(1, min(deg, len(x)-1))
        except Exception:
            deg = 6
        coeffs = np.polyfit(x, df["Wert"].values, deg)
        df["Polynom"] = np.polyval(coeffs, x)

        fig, ax = plt.subplots(figsize=(12, 5.2))
        ax.plot(df["Jahr"], df["Wert"], label="Rohfrequenz (norm.)", linestyle='-', marker='o', markersize=3, alpha=0.6)
        ax.plot(df["Jahr"], df["Gleitmittel"], label="Gleitender Mittelwert (5J)", linewidth=1.8)
        ax.plot(df["Jahr"], df["Polynom"], label=f"Polynom (Grad {deg})", linewidth=1.5, linestyle='--')
        ax.set_title("T-Top10-T-Texts/Jahr (Global, normiert)", fontsize=12)
        ax.set_xlabel("Jahr"); ax.set_ylabel("Normalisierter Wert")
        ax.set_xticks(np.arange(ymin, ymax+1, 10))
        ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"t_top10_t_texts_year_{ymin}_{ymax}_deg{deg}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("T_Top10_T_Texts_Year", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

def build_tab_compare_tokens_topics(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="Global: Tokens, Topics, TT-Texts")
    row=0
    ttk.Label(frame, text="Jahr von–bis:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_years = _mk_entry(frame, width=12); ent_years.insert(0,"1780-1900"); ent_years.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled"); btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    ctx = {"v": ""}

    def _minmax(s: pd.Series) -> pd.Series:
        s = ensure_numeric(s).fillna(0.0)
        return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

    def run():
        try:
            tokens_df = DATA.load_tokens_year()
            global_df = DATA.load_global_topdocs_year()
            begriffe_df = DATA.load_top10_year_value()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return

        if "year" not in tokens_df.columns and "Jahr" in tokens_df.columns:
            tokens_df = tokens_df.rename(columns={"Jahr":"year"})
        if "anzahl_tokens" not in tokens_df.columns:
            for c in ["tokens","count_tokens","anzahl"]:
                if c in tokens_df.columns:
                    tokens_df = tokens_df.rename(columns={c:"anzahl_tokens"}); break
        if "Wert" not in global_df.columns or "Wert" not in begriffe_df.columns:
            messagebox.showerror("Fehler", "Spalte 'Wert' fehlt.", parent=root); return

        tokens_df = tokens_df.sort_values("year")
        global_df = global_df.sort_values("Jahr")
        begriffe_df = begriffe_df.sort_values("Jahr")

        tokens_df["mw"] = ensure_numeric(tokens_df["anzahl_tokens"]).rolling(window=5, center=True, min_periods=1).mean()
        global_df["mw"] = ensure_numeric(global_df["Wert"]).rolling(window=5, center=True, min_periods=1).mean()
        begriffe_df["mw"] = ensure_numeric(begriffe_df["Wert"]).rolling(window=5, center=True, min_periods=1).mean()

        tokens_df["norm"] = _minmax(tokens_df["mw"])
        global_df["norm"] = _minmax(global_df["mw"])
        begriffe_df["norm"] = _minmax(begriffe_df["mw"])

        ymin = min(int(tokens_df["year"].min()), int(global_df["Jahr"].min()), int(begriffe_df["Jahr"].min()))
        ymax = max(int(tokens_df["year"].max()), int(global_df["Jahr"].max()), int(begriffe_df["Jahr"].max()))
        ymin, ymax = parse_year_range(ymin, ymax, ent_years.get())

        fig, ax = plt.subplots(figsize=(12, 5.2))
        ax.plot(begriffe_df["Jahr"], begriffe_df["norm"], label="TT-Texts (Termset, norm.)", linestyle="-", linewidth=1.5)
        ax.plot(tokens_df["year"], tokens_df["norm"], label="Tokens (norm.)", linestyle="--", linewidth=1.2)
        ax.plot(global_df["Jahr"], global_df["norm"], label="TopDocs-Distribution (global, norm.)", linestyle="--", linewidth=1.2)
        ax.set_title("TopDocs-Distribution vs. Tokens vs. TT-Texts (normiert)", fontsize=12)
        ax.set_xlabel("Jahr"); ax.set_ylabel("Normalisierter Wert (0–1)")
        ax.set_xticks(np.arange(ymin, ymax+1, 10))
        ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"global_tokens_topics_tttexts_{ymin}_{ymax}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("Global_Tokens_Topics_TTTexts", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

def build_tab_top10_value_per_text_topic(nb: ttk.Notebook, root: tk.Tk) -> None:
    """Anzeige und Rang von TT-Texts; erzwungen: Top30 Texte pro Topic wenn möglich."""
    frame = ttk.Frame(nb); nb.add(frame, text="TT-Texts-Rang")
    row = 0

    cols = ("text", "rank")
    tree = ttk.Treeview(frame, columns=cols, show="headings", height=18)
    for c, w in zip(cols, [520, 100]):
        tree.heading(c, text=c); tree.column(c, width=w, anchor="w")
    tree.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=scroll.set)
    scroll.grid(row=row, column=3, sticky="ns")

    row += 1
    btn_csv = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_csv.grid(row=row, column=2, sticky="e", padx=6, pady=6)

    out_df: Optional[pd.DataFrame] = None

    def run():
        nonlocal out_df
        try:
            df = DATA.load_top10_value_per_text_topic()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return

        if df.shape[1] < 2:
            messagebox.showerror("Fehler", "Datei benötigt mind. zwei Spalten (Text-ID/Name, Wert).", parent=root); return

        # Spalten identifizieren
        text_col = df.columns[0]
        value_col = df.columns[1]
        topic_col = None
        for cand in ["Topic","topic","topic_label"]:
            if cand in df.columns: topic_col = cand; break

        work = df.copy()
        work[value_col] = ensure_numeric(work[value_col]).fillna(0.0)

        # Erzwinge Top30 pro Topic, wenn Topic-Spalte vorhanden; sonst global Top30
        if topic_col:
            work = (work.sort_values([topic_col, value_col], ascending=[True, False])
                        .groupby(topic_col, group_keys=False)
                        .head(30))
        else:
            work = work.sort_values(value_col, ascending=False).head(30)

        # Rang 1 = höchster Wert (innerhalb Topic, falls vorhanden; sonst global)
        if topic_col:
            work["rank"] = work.groupby(topic_col)[value_col].rank(method="min", ascending=False).astype(int)
        else:
            work["rank"] = work[value_col].rank(method="min", ascending=False).astype(int)

        df2 = work.rename(columns={text_col: "text"})[["text","rank"]].sort_values(["rank","text"], ascending=[True, True])

        tree.delete(*tree.get_children())
        for _, r in df2.iterrows():
            tree.insert("", "end", values=(str(r["text"]), int(r["rank"])))

        out_df = df2
        btn_csv.configure(state="normal", command=lambda: ask_save_df(out_df, "TT_Texts_Rang", "text_rank_top30", root))

    ttk.Button(frame, text="Laden/anzeigen", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

# -----------------------------
# Topic-Exploration (global)
# -----------------------------
def build_tab_global_series(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="TopDocs-Distributions/Year")
    row=0
    ttk.Label(frame, text="Jahr von–bis:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_years = _mk_entry(frame, width=12); ent_years.insert(0,"1780-1900"); ent_years.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="Polynom-Grad:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_deg = _mk_entry(frame, width=8); ent_deg.insert(0,"6"); ent_deg.grid(row=row, column=3, sticky="w", padx=6, pady=4)

    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled"); btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    ctx = {"v": ""}

    def run():
        try:
            df = DATA.load_global_topdocs_year()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return

        # Auto-Detect Spaltennamen
        jahr_col = _find_column(df, ["Jahr", "year", "Year", "YEAR", "jahre"])
        if jahr_col is None:
            messagebox.showerror(
                "Fehler",
                f"Keine Jahr-Spalte gefunden.\nVerfügbare Spalten: {', '.join(df.columns)}",
                parent=root
            )
            return
        
        wert_col = _find_column(df, ["Wert", "value", "Value", "count", "Count", "anzahl"])
        if wert_col is None:
            # Nehme erste numerische Spalte außer Jahr
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != jahr_col]
            if numeric_cols:
                wert_col = numeric_cols[0]
            else:
                messagebox.showerror("Fehler", "Keine Wert-Spalte gefunden.", parent=root)
                return
        
        # Normalisiere Spaltennamen
        df = df.rename(columns={jahr_col: "Jahr", wert_col: "Wert"})

        df = df.sort_values("Jahr")
        ymin, ymax = parse_year_range(int(df["Jahr"].min()), int(df["Jahr"].max()), ent_years.get())
        df = df[(df["Jahr"] >= ymin) & (df["Jahr"] <= ymax)]

        y = ensure_numeric(df["Wert"]).fillna(0.0)
        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y * 0
        df = df.assign(Wert=y_norm)
        df["Gleitmittel"] = df["Wert"].rolling(window=5, center=True, min_periods=1).mean()

        x = df["Jahr"].astype(float).values
        try:
            deg = int((ent_deg.get() or "6").strip()); deg = max(1, min(deg, len(x)-1))
        except Exception:
            deg = 6
        coeffs = np.polyfit(x, df["Wert"].values, deg)
        df["Polynom"] = np.polyval(coeffs, x)

        fig, ax = plt.subplots(figsize=(12, 5.2))
        ax.plot(df["Jahr"], df["Wert"], label="Rohfrequenz (norm.)", linestyle='-', marker='o', markersize=3, alpha=0.6)
        ax.plot(df["Jahr"], df["Gleitmittel"], label="Gleitender Mittelwert (5J)", linewidth=1.8)
        ax.plot(df["Jahr"], df["Polynom"], label=f"Polynom (Grad {deg})", linewidth=1.5, linestyle='--')
        ax.set_title("TopDocs-Distributions/Year (normiert)", fontsize=12)
        ax.set_xlabel("Jahr"); ax.set_ylabel("Normalisierter Wert")
        ax.set_xticks(np.arange(ymin, ymax+1, 10))
        ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"topdocs_distributions_year_{ymin}_{ymax}_deg{deg}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("TopDocs_Distributions_Year", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

def build_tab_global_compare_tokens(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="TopDocs-Distribution vs. Tokens")
    row=0
    ttk.Label(frame, text="Jahr von–bis:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_years = _mk_entry(frame, width=12); ent_years.insert(0,"1780-1900"); ent_years.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled"); btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    ctx = {"v": ""}

    def _minmax(s: pd.Series) -> pd.Series:
        s = ensure_numeric(s).fillna(0.0)
        return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

    def run():
        try:
            tokens_df = DATA.load_tokens_year()
            global_df = DATA.load_global_topdocs_year()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return

        # Auto-Detect Spaltennamen für tokens_df
        jahr_col_tokens = _find_column(tokens_df, ["year", "Jahr", "Year", "YEAR"])
        if jahr_col_tokens is None:
            messagebox.showerror("Fehler", f"Keine Jahr-Spalte in Tokens-Datei gefunden.\nVerfügbare: {', '.join(tokens_df.columns)}", parent=root)
            return
        
        tokens_col = _find_column(tokens_df, ["anzahl_tokens", "tokens", "count_tokens", "anzahl", "token_count"])
        if tokens_col is None:
            numeric_cols = tokens_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != jahr_col_tokens]
            if numeric_cols:
                tokens_col = numeric_cols[0]
            else:
                messagebox.showerror("Fehler", "Keine Token-Spalte gefunden.", parent=root)
                return
        
        # Auto-Detect Spaltennamen für global_df
        jahr_col_global = _find_column(global_df, ["Jahr", "year", "Year", "YEAR"])
        if jahr_col_global is None:
            messagebox.showerror("Fehler", f"Keine Jahr-Spalte in Global-Datei gefunden.\nVerfügbare: {', '.join(global_df.columns)}", parent=root)
            return
        
        wert_col_global = _find_column(global_df, ["Wert", "value", "Value", "count"])
        if wert_col_global is None:
            numeric_cols = global_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != jahr_col_global]
            if numeric_cols:
                wert_col_global = numeric_cols[0]
            else:
                messagebox.showerror("Fehler", "Keine Wert-Spalte in Global-Datei gefunden.", parent=root)
                return
        
        # Normalisiere Spaltennamen
        tokens_df = tokens_df.rename(columns={jahr_col_tokens: "year", tokens_col: "anzahl_tokens"})
        global_df = global_df.rename(columns={jahr_col_global: "Jahr", wert_col_global: "Wert"})

        tokens_df = tokens_df.sort_values("year")
        global_df = global_df.sort_values("Jahr")

        tokens_df["mw"] = ensure_numeric(tokens_df["anzahl_tokens"]).rolling(window=5, center=True, min_periods=1).mean()
        global_df["mw"] = ensure_numeric(global_df["Wert"]).rolling(window=5, center=True, min_periods=1).mean()

        tokens_df["norm"] = _minmax(tokens_df["mw"])
        global_df["norm"] = _minmax(global_df["mw"])

        ymin = min(int(tokens_df["year"].min()), int(global_df["Jahr"].min()))
        ymax = max(int(tokens_df["year"].max()), int(global_df["Jahr"].max()))
        ymin, ymax = parse_year_range(ymin, ymax, ent_years.get())

        fig, ax = plt.subplots(figsize=(12, 5.2))
        ax.plot(tokens_df["year"], tokens_df["norm"], label="Tokens (norm.)", linestyle="--", linewidth=1.2)
        ax.plot(global_df["Jahr"], global_df["norm"], label="TopDocs-Distribution (norm.)", linestyle="-", linewidth=1.5)
        ax.set_title("TopDocs-Distribution vs. Tokens (normiert)", fontsize=12)
        ax.set_xlabel("Jahr"); ax.set_ylabel("Normalisierter Wert (0–1)")
        ax.set_xticks(np.arange(ymin, ymax+1, 10))
        ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"topdocs_distribution_vs_tokens_{ymin}_{ymax}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("TopDocs_Distribution_vs_Tokens", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

def build_tab_topic_trends_from_distribution(nb: ttk.Notebook, root: tk.Tk) -> None:
    """Integriert die Verlaufsfunktion (Topics+Metadata)."""
    frame=ttk.Frame(nb); nb.add(frame, text="Topic-Verläufe (Cosinus/Schwelle)")
    row=0
    ttk.Label(frame, text="Schwelle (Cosinus):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_thr = _mk_entry(frame, width=8); ent_thr.insert(0,"0.2"); ent_thr.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="MA-Fenster:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_ma = _mk_entry(frame, width=8); ent_ma.insert(0,"3"); ent_ma.grid(row=row, column=3, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="Poly-Grad:").grid(row=row, column=4, sticky="w", padx=6, pady=4)
    ent_deg = _mk_entry(frame, width=8); ent_deg.insert(0,"3"); ent_deg.grid(row=row, column=5, sticky="w", padx=6, pady=4)

    row+=1
    abs_var = tk.BooleanVar(value=False)
    smooth_var = tk.BooleanVar(value=True)
    poly_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Absolut", variable=abs_var).grid(row=row, column=0, sticky="w", padx=6, pady=2)
    ttk.Checkbutton(frame, text="Geglättet", variable=smooth_var).grid(row=row, column=1, sticky="w", padx=6, pady=2)
    ttk.Checkbutton(frame, text="Polynom", variable=poly_var).grid(row=row, column=2, sticky="w", padx=6, pady=2)

    row+=1
    ttk.Label(frame, text="Topics auswählen:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, width=60, height=12, exportselection=False)
    listbox.grid(row=row, column=1, columnspan=5, sticky="nsew", padx=6, pady=6); frame.rowconfigure(row, weight=1)

    def load_topics_to_listbox():
        listbox.delete(0, tk.END)
        try:
            df = DATA.load_topics_dist()
            for c in sorted(df.columns.tolist()):
                listbox.insert(tk.END, c)
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)

    load_topics_to_listbox()

    def compute():
        try:
            df_topics = DATA.load_topics_dist()
            mapping_df = DATA.load_metadata()
        except Exception as e:
            messagebox.showerror("Fehler", f"Datenfehler: {e}", parent=root); return
        try:
            thr=float(ent_thr.get().strip()); ma=max(1,int(ent_ma.get().strip())); deg=max(1,int(ent_deg.get().strip()))
        except Exception:
            messagebox.showerror("Fehler","Parameter prüfen.", parent=root); return

        mapping_df = mapping_df.copy()
        mapping_df['_id'] = mapping_df['_id'].astype(str)
        mapping_df['Jahr_final'] = pd.to_numeric(mapping_df['Jahr_final'], errors="coerce")

        # Flexible ID-Normalisierung für besseres Matching
        df = df_topics.copy()
        idx_original = df.index.astype(str)
        idx_normalized = idx_original.map(_normalize_id)
        mapping_ids_normalized = mapping_df['_id'].map(_normalize_id)
        
        jahr_mapping = dict(zip(mapping_ids_normalized, mapping_df['Jahr_final']))
        df['Jahr'] = idx_normalized.map(jahr_mapping)
        
        # Warnung bei vielen fehlgeschlagenen Matches
        missing_count = df['Jahr'].isna().sum()
        total_count = len(df)
        if missing_count > 0:
            missing_ratio = missing_count / total_count
            if missing_ratio > 0.5:
                messagebox.showwarning(
                    "ID-Matching Warnung",
                    f"{missing_ratio*100:.1f}% der Dokument-IDs konnten nicht mit Metadaten gemapped werden.\n"
                    f"({missing_count} von {total_count} Dokumenten)\n\n"
                    "Mögliche Ursachen:\n"
                    "- Verschiedene ID-Formate in den Dateien\n"
                    "- Dateien stammen aus unterschiedlichen Quellen\n\n"
                    "Das Plotting wird mit den verfügbaren Daten fortgesetzt.",
                    parent=root
                )
        
        df = df.dropna(subset=['Jahr'])
        if df.empty:
            messagebox.showerror("Fehler", "Keine Dokumente nach ID-Matching übrig. Dateien passen nicht zusammen.", parent=root)
            return
        
        df['Jahr'] = df['Jahr'].astype(int)
        df = df[df['Jahr'] >= 1840]

        selected_indices = listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Fehler","Keine Topics ausgewählt.", parent=root); return
        selected_topics = [listbox.get(i) for i in selected_indices]

        df_grouped = df.groupby('Jahr').mean().fillna(0.0)

        if abs_var.get():
            fig, ax = plt.subplots(figsize=(12, 6))
            for topic_label in selected_topics:
                years = df_grouped.index.values
                values = df_grouped[topic_label].values
                ax.plot(years, values, label=topic_label)
            ax.set_xlabel('Jahr'); ax.set_ylabel('Durchschnittliche Cosinus-Ähnlichkeit')
            ax.set_title('Absolute Topic-Verläufe (Jahresmittel)')
            ax.legend(title='Topics', bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8)
            _shrink_axes(ax); _apply_layout(fig); plt.show()

        if smooth_var.get():
            fig, ax = plt.subplots(figsize=(12, 6))
            for topic_label in selected_topics:
                years = df_grouped.index.values
                values = df_grouped[topic_label].values
                values_ma = pd.Series(values).rolling(window=ma, min_periods=1, center=True).mean()
                ax.plot(years, values_ma, label=topic_label)
            ax.set_xlabel('Jahr'); ax.set_ylabel('Durchschnittliche Cosinus-Ähnlichkeit')
            ax.set_title('Gleitender Mittelwert der ausgewählten Topics')
            ax.legend(title='Topics', bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8)
            _shrink_axes(ax); _apply_layout(fig); plt.show()

        relevant_counts = pd.DataFrame(index=sorted(df['Jahr'].unique()))
        for topic_label in selected_topics:
            counts_per_year = df.groupby('Jahr')[topic_label].apply(lambda x: (x >= thr).sum())
            relevant_counts[topic_label] = counts_per_year
        fig, ax = plt.subplots(figsize=(12, 6))
        for topic_label in selected_topics:
            ax.plot(relevant_counts.index, relevant_counts[topic_label], label=topic_label)
        ax.set_xlabel('Jahr'); ax.set_ylabel(f'Anzahl Texte mit Cosinus ≥ {thr}')
        ax.set_title(f'Anzahl relevanter Dokumente pro Jahr (Schwelle {thr})')
        ax.legend(title='Topics', bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8)
        _shrink_axes(ax); _apply_layout(fig); plt.show()

        if poly_var.get():
            fig, ax = plt.subplots(figsize=(12, 6))
            for topic_label in selected_topics:
                years = df_grouped.index.values
                values = df_grouped[topic_label].values
                mask = ~np.isnan(values)
                years_clean = years[mask]; values_clean = values[mask]
                if len(years_clean) < deg + 1:
                    continue
                z = np.polyfit(years_clean, values_clean, deg)
                p = np.poly1d(z); values_poly = p(years_clean)
                ax.plot(years_clean, values_poly, label=topic_label)
            if plt.gca().has_data():
                ax.legend(title='Topics', bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8)
            ax.set_xlabel('Jahr'); ax.set_ylabel('Durchschnittliche Cosinus-Ähnlichkeit')
            ax.set_title(f'Polynomiale Regression (Grad {deg}) der ausgewählten Topics')
            _shrink_axes(ax); _apply_layout(fig); plt.show()

    ttk.Button(frame, text="Berechnen", command=compute).grid(row=row+1, column=0, padx=6, pady=8, sticky="w")

# =============================
# Workspace Management Tab
# =============================
def build_tab_workspace(root_nb: ttk.Notebook, root: tk.Tk) -> None:
    """Tab für Arbeitsordner-Verwaltung mit Auto-Discovery"""
    frame = ttk.Frame(root_nb)
    root_nb.add(frame, text="📁 Arbeitsordner")
    
    row = 0
    
    # Header
    header = ttk.Label(frame, text="Arbeitsordner-Verwaltung", font=("TkDefaultFont", 12, "bold"))
    header.grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=(6, 10))
    
    row += 1
    
    # Aktueller Arbeitsordner
    ttk.Label(frame, text="Aktueller Arbeitsordner:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    workspace_var = tk.StringVar(value=str(DATA.current_base_dir))
    ent_workspace = _mk_entry(frame, width=70, textvariable=workspace_var)
    ent_workspace.grid(row=row, column=1, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    def browse_workspace():
        path = filedialog.askdirectory(
            parent=root,
            title="Arbeitsordner wählen",
            initialdir=str(DATA.current_base_dir)
        )
        if path:
            workspace_var.set(path)
    
    ttk.Button(frame, text="📁 Wählen...", command=browse_workspace).grid(row=row, column=2, sticky="w", padx=4, pady=4)
    
    row += 1
    
    # Termset-Suffix
    ttk.Label(frame, text="Termset-Suffix (Dateiende):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    suffix_var = tk.StringVar(value=DATA.current_termset_suffix)
    ent_suffix = _mk_entry(frame, width=20, textvariable=suffix_var)
    ent_suffix.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    ttk.Label(frame, text="z.B. _2.3.csv oder _v3.1.csv", foreground="gray").grid(row=row, column=2, sticky="w", padx=4, pady=4)
    
    row += 1
    
    # Info-Text
    info_frame = ttk.LabelFrame(frame, text="ℹ️ Info", padding=8)
    info_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=8)
    
    info_label = ttk.Label(
        info_frame,
        text="Das Tool sucht automatisch nach Dateien basierend auf:\n"
             "• termsets/: Dateien mit dem angegebenen Suffix\n"
             "• topic-models/: Feste Dateinamen (z.B. document-topics-distribution_tag.csv)\n"
             "• processed_termset/: Dateien mit dem angegebenen Suffix",
        justify="left"
    )
    info_label.pack(anchor="w")
    
    row += 1
    
    # Status-Report (Textfeld)
    ttk.Label(frame, text="Gefundene Dateien:").grid(row=row, column=0, sticky="nw", padx=6, pady=4)
    
    status_text = tk.Text(frame, height=20, width=100, wrap="word", font=("Courier", 9))
    status_text.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=6, pady=4)
    frame.rowconfigure(row, weight=1)
    
    status_scroll = ttk.Scrollbar(frame, orient="vertical", command=status_text.yview)
    status_text.configure(yscrollcommand=status_scroll.set)
    status_scroll.grid(row=row, column=3, sticky="ns")
    
    row += 1
    
    # Buttons
    def scan_files():
        """Scannt Dateien und zeigt Status"""
        status_text.delete(1.0, tk.END)
        status_text.insert(tk.END, "🔄 Scanne Dateien...\n\n")
        root.update_idletasks()
        
        try:
            base_dir = Path(workspace_var.get())
            suffix = suffix_var.get().strip()
            
            if not base_dir.exists():
                messagebox.showerror("Fehler", f"Ordner existiert nicht:\n{base_dir}", parent=root)
                return
            
            # Auto-Discovery
            found = DATA.auto_discover_files(base_dir, suffix)
            
            # Status-Report
            status_text.delete(1.0, tk.END)
            report = DATA.get_discovery_report()
            status_text.insert(tk.END, report)
            
            # Warnungen bei fehlenden Dateien
            discovery = FileDiscovery(base_dir, suffix)
            discovery.found_files = found
            missing = discovery.get_missing_files()
            
            if missing:
                status_text.insert(tk.END, f"\n⚠️ Warnung: {len(missing)} Datei(en) nicht gefunden:\n")
                for key in missing:
                    status_text.insert(tk.END, f"  • {key}\n")
                status_text.insert(tk.END, "\nDas Tool funktioniert trotzdem mit den verfügbaren Dateien.\n")
            else:
                status_text.insert(tk.END, "\n✅ Alle Dateien gefunden!\n")
            
            messagebox.showinfo(
                "Scan abgeschlossen",
                f"Arbeitsordner: {base_dir.name}\n"
                f"Gefunden: {len(found) - len(missing)}/{len(found)} Dateien\n\n"
                f"Die Pfade wurden aktualisiert.",
                parent=root
            )
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Scan fehlgeschlagen:\n{e}", parent=root)
            status_text.insert(tk.END, f"\n❌ Fehler: {e}\n")
    
    def show_current_status():
        """Zeigt aktuellen Status ohne neuen Scan"""
        status_text.delete(1.0, tk.END)
        report = DATA.get_discovery_report()
        status_text.insert(tk.END, report)
    
    ttk.Button(frame, text="🔄 Dateien scannen", command=scan_files).grid(row=row, column=0, sticky="w", padx=6, pady=8)
    ttk.Button(frame, text="📊 Aktuellen Status anzeigen", command=show_current_status).grid(row=row, column=1, sticky="w", padx=6, pady=8)
    
    # Initial-Status anzeigen
    show_current_status()

# -----------------------------
# Daten-Tab (Pfade setzen/prüfen)
# -----------------------------
def build_tab_data(root_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(root_nb); root_nb.add(frame, text="Daten")
    row=0

    def row_pick(label: str, getter: Callable[[], Path], setter: Callable[[Path], None]):
        nonlocal row
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        var = tk.StringVar(value=str(getter()))
        _mk_entry(frame, width=90, textvariable=var).grid(row=row, column=1, sticky="we", padx=6, pady=4)
        def browse():
            p = filedialog.askopenfilename(parent=root, initialdir=str(Path(var.get()).parent), title=label,
                                           filetypes=[("CSV","*.csv"),("Alle Dateien","*.*")])
            if p:
                var.set(p); setter(Path(p))
        ttk.Button(frame, text="…", width=3, command=browse).grid(row=row, column=2, sticky="w", padx=4)
        frame.columnconfigure(1, weight=1); row += 1

    # Termset-Topic
    row_pick("Termset (Tags):", lambda: DATA.path_termset, DATA.set_termset)
    row_pick("Top-100-Topics-Words:",   lambda: DATA.path_topic_words, DATA.set_topic_words)
    row_pick("TF-IDF:",         lambda: DATA.path_tfidf, DATA.set_tfidf)
    row_pick("Tag-Topic-Ranking:",        lambda: DATA.path_ranks, DATA.set_ranks)
    row_pick("Tag-Topic-Relevancescore:", lambda: DATA.path_relevance, DATA.set_relevance)
    row_pick("Tag-Topic-Counts/Year:",    lambda: DATA.path_counts_per_year, DATA.set_counts_per_year)
    row_pick("Tag-Topic-Value/Year:",     lambda: DATA.path_top10_year_value, DATA.set_top10_year_value)
    row_pick("Tag-Topic-Value/Texts:",    lambda: DATA.path_top10_value_per_text_topic, DATA.set_top10_value_per_text_topic)
    # Topic-only
    row_pick("Tokens/Year:",    lambda: DATA.path_tokens_year, DATA.set_tokens_year)
    row_pick("Topic-Value/Year:", lambda: DATA.path_global_topdocs_year, DATA.set_global_topdocs_year)
    row_pick("Document-Topics-Distribution:", lambda: DATA.path_topics, DATA.set_topics)
    row_pick("Metadata:", lambda: DATA.path_metadata, DATA.set_metadata)

    row += 1
    info = tk.Text(frame, height=12, width=100); info.grid(row=row, column=0, columnspan=4, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)

    def _ok(name: str, fn: Callable[[], pd.DataFrame]) -> str:
        try:
            df = fn(); return f"✅ {name}: {df.shape[0]:,} × {df.shape[1]:,}"
        except Exception as e:
            return f"❌ {name}: {e}"

    def load_check():
        info.delete(1.0, tk.END)
        msgs = [
            _ok("Termset (Tags)", DATA.load_termset),
            _ok("Top-100-Topics-Words", DATA.load_topic_words),
            _ok("TF-IDF", DATA.load_tfidf),
            _ok("Tag-Topic-Ranking", DATA.load_ranks),
            _ok("Tag-Topic-Relevancescore", DATA.load_relevance),
            _ok("Tag-Topic-Counts/Year", DATA.load_counts_per_year),
            _ok("Tag-Topic-Value/Year", DATA.load_top10_year_value),
            _ok("Tag-Topic-Value/Texts", DATA.load_top10_value_per_text_topic),
            _ok("Tokens/Year", DATA.load_tokens_year),
            _ok("Topic-Value/Year", DATA.load_global_topdocs_year),
            _ok("Document-Topics-Distribution", DATA.load_topics_dist),
            _ok("Metadata", DATA.load_metadata),
        ]
        info.insert(tk.END, "\n".join(msgs))

    ttk.Button(frame, text="Laden & Prüfen", command=load_check).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")

# -----------------------------
# Main – zwei Hauptreiter
# -----------------------------
def main() -> None:
    root = tk.Tk()
    root.title("Tag-Topic-Explorer")  # Fenstername angepasst
    install_safe_exit(root)
    bring_front(root)
    install_focus_minimize(root, enable=True)

    nb_root = ttk.Notebook(root)
    nb_root.pack(fill="both", expand=True)

    # Arbeitsordner-Verwaltung (NEU!)
    build_tab_workspace(nb_root, root)

    # Daten
    build_tab_data(nb_root, root)

    # Reiter 1: Topic-Exploration (global)
    nb_topics = ttk.Notebook(nb_root); nb_root.add(nb_topics, text="Topic-Exploration")
    build_tab_global_series(nb_topics, root)              # TopDocs-Distributions/Year
    build_tab_global_compare_tokens(nb_topics, root)      # TopDocs-Distribution vs. Tokens
    build_tab_topic_trends_from_distribution(nb_topics, root)

    # Reiter 2: Tag-Topic-Exploration (Mapping)
    nb_t2 = ttk.Notebook(nb_root); nb_root.add(nb_t2, text="Tag-Topic-Exploration")
    build_tab_bubbles_rank_topn(nb_t2, root)              # TT-Relevanz (Bubbles)
    # build_tab_bubbles_ranked_score(nb_t2, root)          # ENTFERNT: Bubbles (Score nach …)
    build_tab_tag_relevance(nb_t2, root)                  # Tag-Topic-Relevancescore
    build_tab_topics_year_stacked(nb_t2, root)            # TT-Texts/Jahr (Stacked)
    build_tab_topics_year_poly(nb_t2, root)               # TT-Texts/Jahr (Poly)
    build_tab_series_top10(nb_t2, root)                   # T-Top10-T-Texts/Jahr (Global)
    build_tab_compare_tokens_topics(nb_t2, root)          # Global: Tokens, Topics, TT-Texts
    build_tab_top10_value_per_text_topic(nb_t2, root)     # TT-Texts-Rang

    root.mainloop()

if __name__ == "__main__":
    main()