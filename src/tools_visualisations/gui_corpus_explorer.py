#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpus-Explorer
===============

GUI-Anwendung zur explorativen Analyse von Textkorpora.

Funktionen:
- Ausdrücke: Frequenzen, TF-IDF, Konkordanz, Wortverläufe
- Word2Vec: Embeddings, Netzwerke
- Termset: Streudiagramme, Wortwolke, Dendrogramme
- Texte: UMAP-Streudiagramme
- Topics: Topicverläufe

Projektstruktur (erwartet):
---------------------------

fadelive/
│
├── data/
│   └── raw/
│       └── metadata.csv                    # Dokument-Metadaten (Pflicht)
│
├── resources/
│   ├── termsets/                           # Termset-Definitionen
│   │   └── Termset_Begriffe_2.3.csv        # Standard-Termliste
│   │
│   └── topic-models/                       # Topic-Model-Ausgaben
│       └── topics_exp_v1/                  # Topic-Ordner (Beispiel)
│           └── document-topics-distribution_tag.csv
│
├── output/
│   ├── processed_corpus/                   # Vorverarbeiteter Korpus
│   │   └── korpus_stop.csv                 # Korpus mit Stoppwort-Filterung
│   │
│   ├── dtm_tfidf_stop/                     # DTM und TF-IDF Matrizen
│   │   ├── dtm_minfreq6.csv                # Document-Term-Matrix
│   │   └── tfidf-2000.csv                  # TF-IDF Matrix (Top 2000)
│   │
│   ├── cosine/                             # Kosinus-Ähnlichkeiten
│   │   └── cosine_tfidf1000.csv            # Kosinus-Matrix
│   │
│   ├── word2vec_models/                    # Word2Vec-Modelle
│   │   └── korpus_gen.model                # Trainiertes Modell
│   │
│   └── exploration/                        # Export-Ordner (wird erstellt)
│       ├── networks/                       # Netzwerk-Exporte
│       └── dendrograms/                    # Dendrogramm-Exporte
│
└── gui_corpus_explorer.py                  # Hauptskript

Autor: Hendrick Heimböckel

"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple

import tkinter as tk    
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec, KeyedVectors
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import umap
import networkx as nx

# Optionale Importe
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# =============================================================================
# KONFIGURATION
# =============================================================================

def _detect_project_root() -> Path:
    """Ermittelt das Projektstammverzeichnis."""
    try:
        candidate = Path(__file__).resolve().parents[2]
    except NameError:
        candidate = Path.cwd()
    
    for path in [candidate, candidate.parent]:
        if (path / "output").exists() and (path / "resources").exists():
            return path
    return candidate


PROJECT_ROOT = _detect_project_root()
OUTPUT_DIR = PROJECT_ROOT / "output"
RESOURCES_DIR = PROJECT_ROOT / "resources"
MODEL_DIR = OUTPUT_DIR / "word2vec_models"
TERMSET_DIR = RESOURCES_DIR / "termsets"
EXPLORATION_DIR = OUTPUT_DIR / "exploration"

# Verzeichnisse erstellen
for d in [EXPLORATION_DIR, EXPLORATION_DIR / "networks", EXPLORATION_DIR / "dendrograms"]:
    d.mkdir(parents=True, exist_ok=True)

APP_NAME = "Korpus-Explorer"


# =============================================================================
# METADATEN-ERKENNUNG (MAPPING AUS METADATA.CSV)
# =============================================================================

class MetadataDetector:
    """
    Erkennt Metadaten-Spalten durch Mapping mit metadata.csv.
    
    EINFACHE LOGIK:
    - Metadaten = Spalten, die in metadata.csv existieren
    - Terme = Alle anderen (numerischen) Spalten in DTM/TF-IDF
    
    Beispiel:
        metadata.csv hat: _id, author_surname, title, year, textclass
        DTM hat: _id, author_surname, year, adelheid, mitleid, volumen, ...
        
        → Metadaten in DTM: _id, author_surname, year
        → Terme in DTM: adelheid, mitleid, volumen, ...
    """
    
    def __init__(self):
        self._metadata_columns: set = set()
        self._loaded = False
    
    def load_from_metadata_file(self, metadata_df: pd.DataFrame) -> None:
        """
        Lädt die Spaltennamen aus der Metadaten-Datei.
        Diese werden als einzige Metadaten-Spalten verwendet.
        """
        self._metadata_columns = {col.lower().strip() for col in metadata_df.columns}
        self._loaded = True
    
    def load_from_path(self, path: Path) -> bool:
        """Lädt Metadaten-Spalten aus CSV-Datei (nur Header)."""
        try:
            if path.exists():
                df = pd.read_csv(path, nrows=0, sep=None, engine='python')
                self.load_from_metadata_file(df)
                return True
        except Exception:
            pass
        return False
    
    def is_metadata_column(self, col: str) -> bool:
        """Prüft ob eine Spalte in metadata.csv existiert."""
        return col.lower().strip() in self._metadata_columns
    
    def detect(self, df: pd.DataFrame) -> List[str]:
        """Erkennt Metadaten-Spalten (die auch in metadata.csv sind)."""
        return [col for col in df.columns if self.is_metadata_column(col)]
    
    def get_term_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Ermittelt alle Term-Spalten.
        = Alle numerischen Spalten, die NICHT in metadata.csv sind.
        """
        metadata_cols = set(self.detect(df))
        result = []
        
        for col in df.columns:
            if col in metadata_cols:
                continue
            
            series = df[col]
            
            # Numerisch?
            if pd.api.types.is_numeric_dtype(series):
                result.append(col)
                continue
            
            # Konvertierbar zu numerisch?
            numeric = pd.to_numeric(series, errors='coerce')
            numeric_ratio = numeric.notna().sum() / max(len(series), 1)
            if numeric_ratio >= 0.8:
                result.append(col)
        
        return result
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, dict]:
        """Erstellt detaillierte Analyse aller Spalten."""
        analysis = {}
        for col in df.columns:
            series = df[col]
            is_meta = self.is_metadata_column(col)
            
            if pd.api.types.is_numeric_dtype(series):
                numeric_ratio = 1.0
            else:
                numeric = pd.to_numeric(series, errors='coerce')
                numeric_ratio = numeric.notna().sum() / max(len(series), 1)
            
            analysis[col] = {
                'dtype': str(series.dtype),
                'is_metadata': is_meta,
                'unique_count': series.nunique(),
                'null_count': series.isna().sum(),
                'numeric_ratio': round(numeric_ratio, 3),
            }
        return analysis
    
    def is_loaded(self) -> bool:
        """Prüft ob Metadaten-Spalten geladen wurden."""
        return self._loaded
    
    def get_metadata_column_names(self) -> List[str]:
        """Gibt die Namen aller bekannten Metadaten-Spalten zurück."""
        return sorted(list(self._metadata_columns))


# Globale Instanz
METADATA_DETECTOR = MetadataDetector()




# =============================================================================
# MATPLOTLIB-KONFIGURATION
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.figsize": (14, 9),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.5,
})


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def normalize_filename(s: str) -> str:
    return re.sub(r"[^\w\-.,+]+", "_", s).strip("_") or "export"


def coalesce_years(df: pd.DataFrame) -> pd.DataFrame:
    """Kombiniert year_first und year zu year_final."""
    # Prüfen ob Spalten existieren
    has_year_first = "year_first" in df.columns
    has_year = "year" in df.columns
    
    if has_year_first and has_year:
        y1 = pd.to_numeric(df["year_first"], errors="coerce")
        y2 = pd.to_numeric(df["year"], errors="coerce")
        df["year_final"] = y1.where(y1.notna(), y2)
    elif has_year_first:
        df["year_final"] = pd.to_numeric(df["year_first"], errors="coerce")
    elif has_year:
        df["year_final"] = pd.to_numeric(df["year"], errors="coerce")
    # Wenn keine Jahr-Spalte existiert, wird year_final nicht erstellt
    
    return df


def get_term_columns(df: pd.DataFrame) -> List[str]:
    """
    Ermittelt Term-Spalten (numerisch, keine Metadaten).
    
    Verwendet den globalen MetadataDetector für dynamische Erkennung.
    Funktioniert mit beliebigen Korpus-Strukturen.
    """
    return METADATA_DETECTOR.get_term_columns(df)


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Findet erste vorhandene Spalte (case-insensitive, whitespace-tolerant)."""
    # Mapping: normalisierter Name -> Original-Spaltenname
    normalized_map = {str(c).lower().strip(): c for c in df.columns}
    
    for cand in candidates:
        # Exakte Übereinstimmung
        if cand in df.columns:
            return cand
        # Case-insensitive + stripped
        cand_norm = cand.lower().strip()
        if cand_norm in normalized_map:
            return normalized_map[cand_norm]
    return None


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    """Liest CSV mit automatischer Separator-Erkennung (; oder ,)."""
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else ','
    return pd.read_csv(path, sep=sep, **kwargs)


def ensure_doc_id(df: pd.DataFrame) -> pd.DataFrame:
    """Stellt sicher, dass doc_id existiert."""
    for col in ["doc_id", "_id", "id", "filename"]:
        if col in df.columns and df[col].notna().any():
            df["doc_id"] = df[col].astype(str)
            return df
    df["doc_id"] = np.arange(1, len(df) + 1).astype(str)
    return df


# =============================================================================
# SPEICHER-FUNKTIONEN
# =============================================================================

def save_dataframe(df: pd.DataFrame, tab: str, context: str, parent: tk.Tk) -> None:
    if df is None or df.empty:
        messagebox.showinfo("Info", "Keine Daten.", parent=parent)
        return
    try:
        save_dir = EXPLORATION_DIR / tab
        save_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=parent, title="CSV speichern",
            initialdir=str(save_dir),
            initialfile=f"{normalize_filename(context)}.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if path:
            df.to_csv(path, index=False)
            messagebox.showinfo("Gespeichert", path, parent=parent)
    except Exception as e:
        messagebox.showerror("Fehler", str(e), parent=parent)


def save_figure(fig: plt.Figure, tab: str, context: str, parent: tk.Tk) -> None:
    try:
        save_dir = EXPLORATION_DIR / tab
        save_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=parent, title="PNG speichern",
            initialdir=str(save_dir),
            initialfile=f"{normalize_filename(context)}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")]
        )
        if path:
            fig.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Gespeichert", path, parent=parent)
    except Exception as e:
        messagebox.showerror("Fehler", str(e), parent=parent)


# =============================================================================
# TKINTER-HILFSFUNKTIONEN
# =============================================================================

def create_entry(parent, **kwargs) -> ttk.Entry:
    entry = ttk.Entry(parent, **kwargs)
    entry.configure(state="normal", takefocus=True)
    return entry


def setup_window(root: tk.Tk) -> None:
    def safe_exit():
        try:
            root.quit()
            root.destroy()
        except Exception:
            pass
    root.protocol("WM_DELETE_WINDOW", safe_exit)
    root.bind("<Escape>", lambda e: safe_exit())


def enable_treeview_sort(tree: ttk.Treeview) -> None:
    sort_state = {}
    def sort_by(col, desc):
        data = []
        for iid in tree.get_children(""):
            v = tree.set(iid, col)
            try:
                key = float(str(v).replace(",", ".")) if v else float("inf")
            except ValueError:
                key = v
            data.append((key, iid))
        data.sort(reverse=desc)
        for idx, (_, iid) in enumerate(data):
            tree.move(iid, "", idx)
        sort_state[col] = not desc
    for col in tree["columns"]:
        tree.heading(col, command=lambda c=col: sort_by(c, sort_state.get(c, False)))
    
    # Rechtsklick-Kontextmenü zum Kopieren
    def copy_selection(event=None):
        selected = tree.selection()
        if not selected:
            return
        lines = []
        for iid in selected:
            values = tree.item(iid, "values")
            lines.append("\t".join(str(v) for v in values))
        text = "\n".join(lines)
        tree.clipboard_clear()
        tree.clipboard_append(text)
    
    def copy_all(event=None):
        lines = []
        # Header
        lines.append("\t".join(tree["columns"]))
        # Daten
        for iid in tree.get_children(""):
            values = tree.item(iid, "values")
            lines.append("\t".join(str(v) for v in values))
        text = "\n".join(lines)
        tree.clipboard_clear()
        tree.clipboard_append(text)
    
    # Kontextmenü erstellen
    context_menu = tk.Menu(tree, tearoff=0)
    context_menu.add_command(label="Zeile(n) kopieren", command=copy_selection)
    context_menu.add_command(label="Alle kopieren", command=copy_all)
    
    def show_context_menu(event):
        # Zeile unter Cursor auswählen falls keine Auswahl
        iid = tree.identify_row(event.y)
        if iid and iid not in tree.selection():
            tree.selection_set(iid)
        context_menu.post(event.x_root, event.y_root)
    
    tree.bind("<Button-3>", show_context_menu)  # Rechtsklick
    tree.bind("<Control-c>", copy_selection)    # Strg+C


def get_available_metadata_columns() -> List[str]:
    """Gibt verfügbare Metadaten-Spalten zurück (außer ID-Spalten)."""
    try:
        meta = DATA.load_metadata()
        # ID-Spalten ausschließen
        id_cols = {"_id", "id", "doc_id", "document_id", "ID", "Id"}
        # Content-Spalten ausschließen
        content_prefixes = ("content", "text", "clean_text")
        
        result = []
        for col in meta.columns:
            col_lower = col.lower()
            if col in id_cols or col_lower in id_cols:
                continue
            if any(col_lower.startswith(p) for p in content_prefixes):
                continue
            result.append(col)
        return result
    except Exception:
        return ["author_surname", "title", "year_final", "textclass"]


def create_metadata_selector(parent, row: int, label_text: str = "Anzeigespalten:", num_fields: int = 3) -> Tuple[List[tk.StringVar], int]:
    """Erstellt Dropdown-Menüs zur Auswahl von Metadaten-Spalten.
    
    Args:
        parent: Parent-Widget
        row: Aktuelle Zeile im Grid
        label_text: Beschriftung
        num_fields: Anzahl der Dropdown-Felder (2 oder 3)
    
    Returns:
        Tuple von (Liste der StringVars, nächste Zeile)
    """
    available_cols = get_available_metadata_columns()
    available_cols = sorted(available_cols, key=str.lower)

    # Standard-Auswahl
    defaults = ["author_surname", "title", "year_final"][:num_fields]
    
    ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=6, pady=4)
    
    selector_frame = ttk.Frame(parent)
    selector_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=6, pady=4)
    
    vars_list = []
    for i, default in enumerate(defaults):
        var = tk.StringVar(value=default if default in available_cols else (available_cols[i] if i < len(available_cols) else ""))
        combo = ttk.Combobox(selector_frame, textvariable=var, values=[""] + available_cols, width=15, state="readonly")
        combo.pack(side="left", padx=2)
        vars_list.append(var)
    
    return vars_list, row + 1


# =============================================================================
# MODEL-MANAGER
# =============================================================================

class ModelManager:
    """Verwaltet Word2Vec-Modelle und Termlisten."""
    
    def __init__(self):
        self.model_path = MODEL_DIR / "korpus_gen.model"
        self.termlist_path = TERMSET_DIR / "Termset_Begriffe_2.3.csv"
        self._model: Optional[KeyedVectors] = None
    
    def load_model(self) -> KeyedVectors:
        if self._model is not None:
            return self._model
        
        path = self.model_path
        if not path.exists():
            raise FileNotFoundError(f"Modell nicht gefunden: {path}")
        
        if path.suffix in {'.wordvectors', '.kv'}:
            self._model = KeyedVectors.load(str(path))
        elif path.suffix == '.model':
            self._model = Word2Vec.load(str(path)).wv
        else:
            binary = path.suffix.lower() in {".bin", ".gz"}
            self._model = KeyedVectors.load_word2vec_format(str(path), binary=binary)
        return self._model
    
    def choose_model(self, parent: tk.Tk, label: Optional[ttk.Label] = None) -> None:
        path = filedialog.askopenfilename(
            parent=parent, title="Word2Vec-Modell wählen",
            initialdir=str(MODEL_DIR),
            filetypes=[("Modelle", "*.model *.wordvectors *.kv *.bin"), ("Alle", "*.*")]
        )
        if path:
            self.model_path = Path(path)
            self._model = None
            if label:
                label.config(text=str(self.model_path))
    
    def choose_termlist(self, parent: tk.Tk, label: Optional[ttk.Label] = None) -> None:
        path = filedialog.askopenfilename(
            parent=parent, title="Termliste wählen",
            initialdir=str(TERMSET_DIR),
            filetypes=[("CSV", "*.csv")]
        )
        if path:
            self.termlist_path = Path(path)
            if label:
                label.config(text=str(self.termlist_path))


MODEL = ModelManager()


# =============================================================================
# DATENMANAGER
# =============================================================================

class DataManager:
    """Zentrale Datenverwaltung mit Caching."""
    
    def __init__(self):
        self.path_corpus = OUTPUT_DIR / "processed_corpus" / "korpus_stop.csv"
        self.path_dtm = OUTPUT_DIR / "dtm_tfidf_stop" / "dtm_minfreq6.csv"
        self.path_tfidf = OUTPUT_DIR / "dtm_tfidf_stop" / "tfidf-2000.csv"
        self.path_topics = RESOURCES_DIR / "topic-models" / "topics_exp_v1" / "document-topics-distribution_tag.csv"
        self.path_metadata = PROJECT_ROOT / "data" / "raw" / "metadata.csv"
        self.path_cosine = OUTPUT_DIR / "cosine" / "cosine_tfidf1000.csv"
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def invalidate_cache(self):
        self._cache.clear()
    
    def load_corpus(self) -> pd.DataFrame:
        if "corpus" in self._cache:
            return self._cache["corpus"]
        if not self.path_corpus.exists():
            raise FileNotFoundError(f"Korpus nicht gefunden: {self.path_corpus}")
        df = read_csv_auto(self.path_corpus)
        # Flexiblere Content-Spalten-Erkennung
        text_col = find_column(df, ["content_stop", "content_lem", "content_min", "content_gen", "text", "clean_text", "content"])
        if text_col:
            df["text"] = df[text_col].fillna("").astype(str)
        df = ensure_doc_id(df)
        df = coalesce_years(df)
        self._cache["corpus"] = df
        return df
    
    def load_dtm(self) -> pd.DataFrame:
        if "dtm" in self._cache:
            return self._cache["dtm"]
        if not self.path_dtm.exists():
            raise FileNotFoundError(f"DTM nicht gefunden: {self.path_dtm}")
        df = read_csv_auto(self.path_dtm)
        df = coalesce_years(df)
        self._cache["dtm"] = df
        return df
    
    def load_tfidf(self) -> pd.DataFrame:
        if "tfidf" in self._cache:
            return self._cache["tfidf"]
        if not self.path_tfidf.exists():
            raise FileNotFoundError(f"TF-IDF nicht gefunden: {self.path_tfidf}")
        df = read_csv_auto(self.path_tfidf)
        self._cache["tfidf"] = df
        return df
    
    def load_topics(self) -> pd.DataFrame:
        if "topics" in self._cache:
            return self._cache["topics"]
        if not self.path_topics.exists():
            raise FileNotFoundError(f"Topics nicht gefunden: {self.path_topics}")
        df = read_csv_auto(self.path_topics, index_col=0)
        df.index = df.index.astype(str).str.replace(".txt", "", regex=False)
        self._cache["topics"] = df
        return df
    
    def load_metadata(self) -> pd.DataFrame:
        if "metadata" in self._cache:
            return self._cache["metadata"]
        if not self.path_metadata.exists():
            raise FileNotFoundError(f"Metadaten nicht gefunden: {self.path_metadata}")
        df = read_csv_auto(self.path_metadata)
        # Flexible ID-Spalten-Erkennung: _id oder id
        id_col = find_column(df, ["_id", "id", "doc_id", "document_id"])
        if id_col:
            df["_id"] = df[id_col].astype(str)
        df = coalesce_years(df)
        self._cache["metadata"] = df
        
        # Registriere Metadaten-Spalten für DTM/TF-IDF Mapping
        METADATA_DETECTOR.load_from_metadata_file(df)
        
        return df
    
    def load_cosine(self) -> pd.DataFrame:
        if "cosine" in self._cache:
            return self._cache["cosine"]
        if not self.path_cosine.exists():
            raise FileNotFoundError(f"Kosinus nicht gefunden: {self.path_cosine}")
        df = read_csv_auto(self.path_cosine, index_col=0)
        self._cache["cosine"] = df
        return df
    
    def get_tfidf_averages(self) -> pd.DataFrame:
        df = self.load_tfidf()
        term_cols = get_term_columns(df)
        if not term_cols:
            raise ValueError("Keine Term-Spalten gefunden.")
        avg = df[term_cols].mean().sort_values(ascending=False)
        return pd.DataFrame({"term": avg.index, "tfidf_avg": avg.values, "rank": np.arange(1, len(avg)+1)})


DATA = DataManager()


# =============================================================================
# TAB: FREQUENZ
# =============================================================================

def build_tab_frequency(notebook: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Frequenz")
    
    row = 0
    ttk.Label(frame, text="Suche (Komma):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_search = create_entry(frame, width=50)
    ent_search.grid(row=row, column=1, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    ttk.Label(frame, text="Top-N:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_topn = create_entry(frame, width=8)
    ent_topn.insert(0, "500")
    ent_topn.grid(row=row, column=3, sticky="w", padx=6, pady=4)
    
    row += 1
    columns = ("rank", "term", "freq")
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=18)
    for c, w in [("rank", 60), ("term", 200), ("freq", 100)]:
        tree.heading(c, text=c)
        tree.column(c, width=w)
    tree.grid(row=row, column=0, columnspan=4, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.grid(row=row, column=4, sticky="ns")
    enable_treeview_sort(tree)
    
    result = {"df": None}
    
    row += 1
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_save.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            dtm = DATA.load_dtm()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        term_cols = get_term_columns(dtm)
        if not term_cols:
            messagebox.showerror("Fehler", "Keine Term-Spalten.", parent=root)
            return
        
        sums = dtm[term_cols].sum().sort_values(ascending=False)
        df = pd.DataFrame({"term": sums.index, "freq": sums.values, "rank": np.arange(1, len(sums)+1)})
        
        search = [t.strip().lower() for t in ent_search.get().split(",") if t.strip()]
        top_n = int(ent_topn.get() or "500")
        
        df_show = df[df["term"].isin(search)] if search else df.head(top_n)
        df_show = df_show[["rank", "term", "freq"]]
        
        tree.delete(*tree.get_children())
        for _, r in df_show.iterrows():
            tree.insert("", "end", values=(int(r["rank"]), r["term"], int(r["freq"])))
        
        result["df"] = df_show
        btn_save.configure(state="normal")
    
    btn_save.configure(command=lambda: save_dataframe(result["df"], "Frequenz", "freq", root))
    ttk.Button(frame, text="Berechnen", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: TF-IDF-RANG
# =============================================================================

def build_tab_tfidf_rank(notebook: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="TF-IDF-Rang")
    
    row = 0
    ttk.Label(frame, text="Suche (Komma):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_search = create_entry(frame, width=50)
    ent_search.grid(row=row, column=1, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    ttk.Label(frame, text="Top-N:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_topn = create_entry(frame, width=8)
    ent_topn.insert(0, "500")
    ent_topn.grid(row=row, column=3, sticky="w", padx=6, pady=4)
    
    row += 1
    columns = ("rank", "term", "tfidf_avg")
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=18)
    for c, w in [("rank", 60), ("term", 200), ("tfidf_avg", 100)]:
        tree.heading(c, text=c)
        tree.column(c, width=w)
    tree.grid(row=row, column=0, columnspan=4, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.grid(row=row, column=4, sticky="ns")
    enable_treeview_sort(tree)
    
    result = {"df": None}
    
    row += 1
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_save.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            df = DATA.get_tfidf_averages()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        search = [t.strip().lower() for t in ent_search.get().split(",") if t.strip()]
        top_n = int(ent_topn.get() or "500")
        
        df_show = df[df["term"].isin(search)] if search else df.head(top_n)
        
        tree.delete(*tree.get_children())
        for _, r in df_show.iterrows():
            tree.insert("", "end", values=(int(r["rank"]), r["term"], round(r["tfidf_avg"], 6)))
        
        result["df"] = df_show
        btn_save.configure(state="normal")
    
    btn_save.configure(command=lambda: save_dataframe(result["df"], "TF-IDF", "tfidf", root))
    ttk.Button(frame, text="Berechnen", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: DOKUMENT-FREQUENZ
# =============================================================================

def build_tab_docfreq(notebook: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Dokument-Frequenz")
    
    row = 0
    ttk.Label(frame, text="Ausdrücke (Komma):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_terms = create_entry(frame, width=60)
    ent_terms.grid(row=row, column=1, columnspan=2, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    row += 1
    regex_var = tk.BooleanVar(value=False)
    case_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame, text="Regex", variable=regex_var).grid(row=row, column=0, sticky="w", padx=6)
    ttk.Checkbutton(frame, text="Groß/Klein", variable=case_var).grid(row=row, column=1, sticky="w", padx=6)
    
    # Metadaten-Auswahl
    row += 1
    meta_vars, row = create_metadata_selector(frame, row, "Anzeigespalten:")
    
    row += 1
    columns = ("doc_id", "meta1", "meta2", "meta3", "count")
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=16)
    widths = [80, 120, 200, 100, 60]
    for c, w in zip(columns, widths):
        tree.heading(c, text=c)
        tree.column(c, width=w)
    tree.grid(row=row, column=0, columnspan=4, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.grid(row=row, column=4, sticky="ns")
    enable_treeview_sort(tree)
    
    result = {"df": None}
    
    row += 1
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_save.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            corpus = DATA.load_corpus()
            metadata = DATA.load_metadata()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        terms = [t.strip() for t in ent_terms.get().split(",") if t.strip()]
        if not terms:
            messagebox.showerror("Fehler", "Bitte Begriffe eingeben.", parent=root)
            return
        
        # Gewählte Metadaten-Spalten
        selected_cols = [v.get() for v in meta_vars if v.get()]
        
        # Spaltenüberschriften aktualisieren
        tree.heading("doc_id", text="doc_id")
        for i, col in enumerate(selected_cols[:3]):
            tree.heading(f"meta{i+1}", text=col)
        for i in range(len(selected_cols), 3):
            tree.heading(f"meta{i+1}", text="")
        tree.heading("count", text="count")
        
        flags = 0 if case_var.get() else re.IGNORECASE
        counts = []
        for term in terms:
            pattern = re.compile(term if regex_var.get() else re.escape(term), flags)
            counts.append(corpus["text"].fillna("").apply(lambda x: len(pattern.findall(str(x)))))
        
        corpus["count"] = sum(counts)
        
        # Metadaten joinen
        id_col = find_column(metadata, ["_id", "id", "doc_id", "document_id"])
        if id_col:
            metadata["_merge_id"] = metadata[id_col].astype(str)
            corpus["_merge_id"] = corpus["doc_id"].astype(str)
            merged = corpus.merge(metadata[["_merge_id"] + [c for c in selected_cols if c in metadata.columns]], 
                                  on="_merge_id", how="left", suffixes=("", "_meta"))
        else:
            merged = corpus
        
        df = merged[merged["count"] > 0].copy()
        df = df.sort_values("count", ascending=False)
        
        tree.delete(*tree.get_children())
        for _, r in df.head(500).iterrows():
            values = [r["doc_id"]]
            for col in selected_cols[:3]:
                val = r.get(col, "")
                if pd.isna(val):
                    val = ""
                elif col.lower().startswith("year") and not pd.isna(val):
                    try:
                        val = int(float(val))
                    except (ValueError, TypeError):
                        pass
                values.append(str(val)[:40])
            # Auffüllen falls weniger als 3 Spalten gewählt
            while len(values) < 4:
                values.append("")
            values.append(int(r["count"]))
            tree.insert("", "end", values=tuple(values))
        
        # Export DataFrame
        export_cols = ["doc_id"] + [c for c in selected_cols if c in df.columns] + ["count"]
        result["df"] = df[[c for c in export_cols if c in df.columns]]
        btn_save.configure(state="normal")
    
    btn_save.configure(command=lambda: save_dataframe(result["df"], "DocFreq", "docfreq", root))
    ttk.Button(frame, text="Suchen", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: KONKORDANZ
# =============================================================================

def build_tab_concordance(notebook: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Konkordanz")
    
    row = 0
    ttk.Label(frame, text="Suchbegriff:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_term = create_entry(frame, width=30)
    ent_term.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    ttk.Label(frame, text="Kontext:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_ctx = create_entry(frame, width=6)
    ent_ctx.insert(0, "50")
    ent_ctx.grid(row=row, column=3, sticky="w", padx=6, pady=4)
    
    # Metadaten-Auswahl (nur 2 Felder für Konkordanz)
    row += 1
    meta_vars, row = create_metadata_selector(frame, row, "Anzeigespalten:", num_fields=2)
    
    row += 1
    columns = ("doc_id", "meta1", "meta2", "left", "match", "right")
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=18)
    widths = [70, 100, 100, 150, 80, 150]
    for c, w in zip(columns, widths):
        tree.heading(c, text=c)
        tree.column(c, width=w)
    tree.grid(row=row, column=0, columnspan=5, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    frame.columnconfigure(1, weight=1)
    
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.grid(row=row, column=5, sticky="ns")
    enable_treeview_sort(tree)
    
    result = {"df": None}
    
    row += 1
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_save.grid(row=row, column=4, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            corpus = DATA.load_corpus()
            metadata = DATA.load_metadata()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        term = ent_term.get().strip()
        if not term:
            messagebox.showerror("Fehler", "Bitte Begriff eingeben.", parent=root)
            return
        
        # Gewählte Metadaten-Spalten (nur 2 für Konkordanz)
        selected_cols = [v.get() for v in meta_vars[:2] if v.get()]
        
        # Spaltenüberschriften aktualisieren
        tree.heading("doc_id", text="doc_id")
        for i, col in enumerate(selected_cols[:2]):
            tree.heading(f"meta{i+1}", text=col)
        for i in range(len(selected_cols), 2):
            tree.heading(f"meta{i+1}", text="")
        
        # Metadaten-Lookup erstellen
        id_col = find_column(metadata, ["_id", "id", "doc_id", "document_id"])
        meta_lookup = {}
        if id_col:
            for _, m_row in metadata.iterrows():
                doc_id = str(m_row[id_col])
                meta_lookup[doc_id] = {col: m_row.get(col, "") for col in selected_cols}
        
        ctx = int(ent_ctx.get() or "50")
        pattern = re.compile(f"(.{{0,{ctx}}})({re.escape(term)})(.{{0,{ctx}}})", re.IGNORECASE)
        
        results = []
        for _, row_data in corpus.iterrows():
            text = str(row_data.get("text", ""))
            doc_id = str(row_data.get("doc_id", ""))
            meta_vals = meta_lookup.get(doc_id, {})
            
            for m in pattern.finditer(text):
                result_row = {
                    "doc_id": doc_id,
                    "left": m.group(1).replace("\n", " "),
                    "match": m.group(2),
                    "right": m.group(3).replace("\n", " ")
                }
                for col in selected_cols:
                    result_row[col] = meta_vals.get(col, "")
                results.append(result_row)
        
        df = pd.DataFrame(results)
        
        tree.delete(*tree.get_children())
        for _, r in df.head(1000).iterrows():
            values = [r["doc_id"]]
            for col in selected_cols[:2]:
                val = r.get(col, "")
                if pd.isna(val):
                    val = ""
                elif str(col).lower().startswith("year") and not pd.isna(val):
                    try:
                        val = int(float(val))
                    except (ValueError, TypeError):
                        pass
                values.append(str(val)[:25])
            while len(values) < 3:
                values.append("")
            values.extend([r["left"], r["match"], r["right"]])
            tree.insert("", "end", values=tuple(values))
        
        result["df"] = df
        btn_save.configure(state="normal")
        messagebox.showinfo("Info", f"{len(df)} Treffer gefunden.", parent=root)
    
    btn_save.configure(command=lambda: save_dataframe(result["df"], "Konkordanz", f"kwic_{ent_term.get()}", root))
    ttk.Button(frame, text="Suchen", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: WORTVERLÄUFE
# =============================================================================

def build_tab_wordtrends(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """Tab: Wortverläufe mit Glättung und Polynom-Regression."""
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Wortverläufe")
    
    row = 0
    ttk.Label(frame, text="Begriffe (Komma, exakte Spaltennamen):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_terms = create_entry(frame, width=50)
    ent_terms.grid(row=row, column=1, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    row += 1
    ttk.Label(frame, text="Glättung (Fenster):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_win = create_entry(frame, width=6)
    ent_win.insert(0, "5")
    ent_win.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Polynom-Grad:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_deg = create_entry(frame, width=6)
    ent_deg.insert(0, "6")
    ent_deg.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    absolute_var = tk.BooleanVar(value=False)
    smooth_var = tk.BooleanVar(value=True)
    poly_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Absolut", variable=absolute_var).grid(row=row, column=0, sticky="w", padx=6)
    ttk.Checkbutton(frame, text="Geglättet (relativ)", variable=smooth_var).grid(row=row, column=1, sticky="w", padx=6)
    ttk.Checkbutton(frame, text="Polynom (relativ)", variable=poly_var).grid(row=row, column=2, sticky="w", padx=6)
    
    row += 1
    ttk.Label(frame, text="Jahrbereich (z.B. 1780-1900):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_range = create_entry(frame, width=15)
    ent_range.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    result = {"fig": None, "df": None}
    
    row += 1
    btn_csv = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled")
    btn_csv.grid(row=row, column=0, sticky="w", padx=6, pady=6)
    btn_png.grid(row=row, column=1, sticky="w", padx=6, pady=6)
    
    def compute():
        try:
            dtm = DATA.load_dtm()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        terms_input = [t.strip() for t in ent_terms.get().split(",") if t.strip()]
        if not terms_input:
            messagebox.showerror("Fehler", "Bitte Begriffe eingeben.", parent=root)
            return
        
        term_cols = get_term_columns(dtm)
        terms = [t for t in terms_input if t in term_cols]
        missing = [t for t in terms_input if t not in term_cols]
        
        if missing:
            messagebox.showwarning("Hinweis", f"Nicht gefunden: {', '.join(missing)}", parent=root)
        
        if not terms:
            messagebox.showerror("Fehler", "Keine gültigen Begriffe.", parent=root)
            return
        
        try:
            win = max(1, int(ent_win.get()))
            deg = max(1, int(ent_deg.get()))
        except ValueError:
            win, deg = 5, 6
        
        # Jahr-Spalte finden
        year_col = find_column(dtm, ["year_final", "year_first", "year"])
        if not year_col:
            messagebox.showerror("Fehler", "Keine Jahr-Spalte.", parent=root)
            return
        
        dtm["_year"] = pd.to_numeric(dtm[year_col], errors="coerce")
        dtm_clean = dtm.dropna(subset=["_year"])
        
        # Jahrbereich
        range_str = ent_range.get().strip()
        if range_str and "-" in range_str:
            y_min, y_max = map(int, range_str.split("-"))
        else:
            y_min = int(dtm_clean["_year"].min())
            y_max = int(dtm_clean["_year"].max())
        
        dtm_filtered = dtm_clean[(dtm_clean["_year"] >= y_min) & (dtm_clean["_year"] <= y_max)]
        
        # Absolute Frequenzen aggregieren
        series_data = {}
        for term in terms:
            grouped = dtm_filtered.groupby("_year")[term].sum()
            series_data[term] = grouped
        
        df_trends = pd.DataFrame(series_data)
        df_trends.index.name = "year"
        
        # Gesamtfrequenz pro Jahr berechnen (für Relativierung)
        # Summe aller Term-Spalten pro Jahr
        total_per_year = dtm_filtered.groupby("_year")[term_cols].sum().sum(axis=1)
        
        # Relative Frequenzen berechnen (pro Million Tokens)
        df_trends_rel = pd.DataFrame(index=df_trends.index)
        for term in terms:
            # Frequenz pro Million Tokens
            df_trends_rel[term] = (df_trends[term] / total_per_year) * 1_000_000
        
        # Plots
        if absolute_var.get():
            fig = plt.figure(figsize=(12, 6))
            for term in terms:
                plt.plot(df_trends.index, df_trends[term], label=term)
            plt.title("Rohfrequenzen (absolut)")
            plt.xlabel("Jahr")
            plt.ylabel("Frequenz")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            result["fig"] = fig
            btn_png.configure(state="normal")
            plt.show()
        
        if smooth_var.get():
            fig = plt.figure(figsize=(12, 6))
            for term in terms:
                # Relativierte Werte glätten
                smoothed = df_trends_rel[term].rolling(window=win, center=True, min_periods=1).mean()
                plt.plot(df_trends_rel.index, smoothed, label=f"{term}")
            plt.title(f"Relative Frequenz (geglättet, Fenster={win})")
            plt.xlabel("Jahr")
            plt.ylabel("Frequenz pro Million Tokens")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            result["fig"] = fig
            btn_png.configure(state="normal")
            plt.show()
        
        if poly_var.get():
            fig = plt.figure(figsize=(12, 6))
            for term in terms:
                x = df_trends_rel.index.astype(float).values
                y = df_trends_rel[term].values
                # NaN-Werte entfernen
                mask = ~np.isnan(y)
                x_clean, y_clean = x[mask], y[mask]
                if len(x_clean) > deg:
                    coeffs = np.polyfit(x_clean, y_clean, deg)
                    xx = np.linspace(x_clean.min(), x_clean.max(), 200)
                    yy = np.polyval(coeffs, xx)
                    plt.plot(xx, yy, label=term)
            plt.title(f"Relative Frequenz (Polynom Grad {deg})")
            plt.xlabel("Jahr")
            plt.ylabel("Frequenz pro Million Tokens")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            result["fig"] = fig
            btn_png.configure(state="normal")
            plt.show()
        
        # DataFrame mit absoluten und relativen Werten für Export
        df_export = df_trends.reset_index()
        for term in terms:
            df_export[f"{term}_rel_pmw"] = df_trends_rel[term].values
        result["df"] = df_export
        btn_csv.configure(state="normal")
    
    btn_csv.configure(command=lambda: save_dataframe(result["df"], "Wortverläufe", "_".join(ent_terms.get().split(",")[:3]), root))
    btn_png.configure(command=lambda: save_figure(result["fig"], "Wortverläufe", "trends", root))
    
    row += 1
    ttk.Button(frame, text="Plotten", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: KOLLOKATION
# =============================================================================

def build_tab_collocations(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """Tab: Kollokationsanalyse mit PMI/Frequenz."""
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Kollokation")
    
    row = 0
    ttk.Label(frame, text="Zielausdrücke (Komma):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_targets = create_entry(frame, width=50)
    ent_targets.grid(row=row, column=1, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    row += 1
    ttk.Label(frame, text="Fensterweite (±):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_window = create_entry(frame, width=6)
    ent_window.insert(0, "5")
    ent_window.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Top-N:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_topn = create_entry(frame, width=6)
    ent_topn.insert(0, "100")
    ent_topn.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="N-Gram:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_ng = create_entry(frame, width=6)
    ent_ng.insert(0, "1")
    ent_ng.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Min. Frequenz:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_minf = create_entry(frame, width=6)
    ent_minf.insert(0, "3")
    ent_minf.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Metrik:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    metric_var = tk.StringVar(value="FREQ")
    ttk.Combobox(frame, textvariable=metric_var, values=["FREQ", "PMI"], width=8, state="readonly").grid(
        row=row, column=1, sticky="w", padx=6, pady=4
    )
    
    # Kollokationsliste
    row += 1
    columns = ("target", "collocate", "freq", "score")
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)
    for c, w in [("target", 120), ("collocate", 180), ("freq", 70), ("score", 80)]:
        tree.heading(c, text=c)
        tree.column(c, width=w)
    tree.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.grid(row=row, column=3, sticky="ns")
    enable_treeview_sort(tree)
    
    # Metadaten-Auswahl für Dokumentliste
    row += 1
    ttk.Label(frame, text="Dokumente (Klick auf Kollokation):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    
    row += 1
    meta_vars, row = create_metadata_selector(frame, row, "Anzeigespalten:")
    
    row += 1
    cols_doc = ("doc_id", "meta1", "meta2", "meta3", "freq")
    tree_doc = ttk.Treeview(frame, columns=cols_doc, show="headings", height=8)
    for c, w in zip(cols_doc, [80, 120, 200, 100, 60]):
        tree_doc.heading(c, text=c)
        tree_doc.column(c, width=w)
    tree_doc.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    scroll2 = ttk.Scrollbar(frame, orient="vertical", command=tree_doc.yview)
    tree_doc.configure(yscrollcommand=scroll2.set)
    scroll2.grid(row=row, column=3, sticky="ns")
    enable_treeview_sort(tree_doc)
    
    result = {"df": None, "docs_df": None, "docs_cache": {}}
    
    row += 1
    btn_save = ttk.Button(frame, text="Kollokationen speichern", state="disabled")
    btn_save_docs = ttk.Button(frame, text="Dokumente speichern", state="disabled")
    btn_save.grid(row=row, column=1, sticky="w", padx=6, pady=6)
    btn_save_docs.grid(row=row, column=2, sticky="e", padx=6, pady=6)
    
    import math
    from collections import Counter, defaultdict, deque
    
    tok_re = re.compile(r"\w+", flags=re.UNICODE)
    
    def tokenize(text):
        if not isinstance(text, str):
            return []
        return [t.lower() for t in tok_re.findall(text)]
    
    def iter_ngrams(tokens, n):
        if n <= 1:
            yield from tokens
        else:
            buf = deque(maxlen=n)
            for t in tokens:
                buf.append(t)
                if len(buf) == n:
                    yield " ".join(buf)
    
    def compute():
        try:
            df = DATA.load_corpus()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        targets = [t.strip().lower() for t in ent_targets.get().split(",") if t.strip()]
        if not targets:
            messagebox.showerror("Fehler", "Bitte Zielausdrücke eingeben.", parent=root)
            return
        
        try:
            W = max(1, int(ent_window.get()))
            topn = max(1, int(ent_topn.get()))
            minf = max(1, int(ent_minf.get()))
            ng = max(1, int(ent_ng.get()))
        except ValueError:
            messagebox.showerror("Fehler", "Parameter prüfen.", parent=root)
            return
        
        tree.delete(*tree.get_children())
        tree_doc.delete(*tree_doc.get_children())
        result["docs_cache"].clear()
        
        total_tokens = 0
        freq_w = Counter()
        freq_tw = defaultdict(Counter)
        freq_t = Counter()
        
        for _, r in df.iterrows():
            tokens = tokenize(r.get("text", ""))
            if not tokens:
                continue
            
            ngrams = list(iter_ngrams(tokens, ng))
            total_tokens += len(ngrams)
            for w in ngrams:
                freq_w[w] += 1
            
            for i, w in enumerate(tokens):
                if w in targets:
                    freq_t[w] += 1
                    L, R = max(0, i - W), min(len(tokens), i + W + 1)
                    if ng == 1:
                        ctx = tokens[L:i] + tokens[i+1:R]
                    else:
                        ctx = list(iter_ngrams(tokens[L:i], ng)) + list(iter_ngrams(tokens[i+1:R], ng))
                    for cw in ctx:
                        freq_tw[w][cw] += 1
        
        rows = []
        eps = 1e-12
        for t in targets:
            ct = max(1, freq_t[t])
            for cw, ctw in freq_tw[t].items():
                if ctw < minf:
                    continue
                if metric_var.get() == "FREQ":
                    score = float(ctw)
                else:
                    pw = freq_w[cw] / max(1, total_tokens)
                    pt = ct / max(1, total_tokens)
                    ptw = ctw / max(1, total_tokens)
                    score = math.log2(max(eps, ptw) / max(eps, pt * pw))
                rows.append((t, cw, int(ctw), float(score)))
        
        if not rows:
            messagebox.showinfo("Info", "Keine Kollokationen gefunden.", parent=root)
            return
        
        res = pd.DataFrame(rows, columns=["target", "collocate", "freq", "score"])
        sort_col = "freq" if metric_var.get() == "FREQ" else "score"
        res = res.sort_values(["target", sort_col], ascending=[True, False]).groupby("target", group_keys=False).head(topn)
        
        for _, r in res.iterrows():
            tree.insert("", "end", values=(r["target"], r["collocate"], int(r["freq"]), round(r["score"], 4)))
        
        result["df"] = res
        btn_save.configure(state="normal")
        messagebox.showinfo("Info", f"{len(res)} Kollokationen gefunden.", parent=root)
    
    def on_select(event):
        sel = tree.selection()
        if not sel:
            return
        vals = tree.item(sel[0], "values")
        if not vals or len(vals) < 2:
            return
        
        t, cw = str(vals[0]), str(vals[1])
        key = (t, cw)
        
        # Gewählte Metadaten-Spalten
        selected_cols = [v.get() for v in meta_vars if v.get()]
        
        # Spaltenüberschriften aktualisieren
        tree_doc.heading("doc_id", text="doc_id")
        for i, col in enumerate(selected_cols[:3]):
            tree_doc.heading(f"meta{i+1}", text=col)
        for i in range(len(selected_cols), 3):
            tree_doc.heading(f"meta{i+1}", text="")
        tree_doc.heading("freq", text="freq")
        
        if key in result["docs_cache"]:
            docs_df = result["docs_cache"][key]
        else:
            try:
                df = DATA.load_corpus()
                metadata = DATA.load_metadata()
                W = max(1, int(ent_window.get()))
                ng = max(1, int(ent_ng.get()))
            except Exception:
                return
            
            # Metadaten-Lookup erstellen
            id_col = find_column(metadata, ["_id", "id", "doc_id", "document_id"])
            meta_lookup = {}
            if id_col:
                for _, m_row in metadata.iterrows():
                    doc_id = str(m_row[id_col])
                    meta_lookup[doc_id] = {col: m_row.get(col, "") for col in selected_cols}
            
            rows_doc = []
            for _, d in df.iterrows():
                tokens = tokenize(d.get("text", ""))
                positions = [i for i, w in enumerate(tokens) if w == t]
                if not positions:
                    continue
                
                count = 0
                for i in positions:
                    L, R = max(0, i - W), min(len(tokens), i + W + 1)
                    if ng == 1:
                        ctx = tokens[L:i] + tokens[i+1:R]
                    else:
                        ctx = list(iter_ngrams(tokens[L:i], ng)) + list(iter_ngrams(tokens[i+1:R], ng))
                    count += sum(1 for c in ctx if c == cw)
                
                if count > 0:
                    doc_id = str(d.get("doc_id", ""))
                    row_data = {"doc_id": doc_id, "freq": count}
                    meta_vals = meta_lookup.get(doc_id, {})
                    for col in selected_cols:
                        row_data[col] = meta_vals.get(col, "")
                    rows_doc.append(row_data)
            
            docs_df = pd.DataFrame(rows_doc).sort_values("freq", ascending=False) if rows_doc else pd.DataFrame()
            result["docs_cache"][key] = docs_df
        
        tree_doc.delete(*tree_doc.get_children())
        for _, r in docs_df.head(200).iterrows():
            values = [r["doc_id"]]
            for col in selected_cols[:3]:
                val = r.get(col, "")
                if pd.isna(val):
                    val = ""
                elif str(col).lower().startswith("year") and not pd.isna(val):
                    try:
                        val = int(float(val))
                    except (ValueError, TypeError):
                        pass
                values.append(str(val)[:35])
            while len(values) < 4:
                values.append("")
            values.append(int(r["freq"]))
            tree_doc.insert("", "end", values=tuple(values))
        
        result["docs_df"] = docs_df
        btn_save_docs.configure(state="normal")
    
    tree.bind("<<TreeviewSelect>>", on_select)
    
    btn_save.configure(command=lambda: save_dataframe(result["df"], "Kollokation", f"colloc_{ent_targets.get()}", root))
    btn_save_docs.configure(command=lambda: save_dataframe(result["docs_df"], "Kollokation", "docs", root))
    
    ttk.Button(frame, text="Berechnen", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: EMBEDDINGS
# =============================================================================

def build_tab_embeddings(notebook: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Embeddings")
    
    row = 0
    ttk.Label(frame, text="Wort:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_word = create_entry(frame, width=25)
    ent_word.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    ttk.Label(frame, text="Top-N:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_topn = create_entry(frame, width=6)
    ent_topn.insert(0, "20")
    ent_topn.grid(row=row, column=3, sticky="w", padx=6, pady=4)
    
    row += 1
    columns = ("rank", "word", "similarity")
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=18)
    for c, w in [("rank", 50), ("word", 180), ("similarity", 100)]:
        tree.heading(c, text=c)
        tree.column(c, width=w)
    tree.grid(row=row, column=0, columnspan=4, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.grid(row=row, column=4, sticky="ns")
    
    result = {"df": None}
    
    row += 1
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_save.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            kv = MODEL.load_model()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        word = ent_word.get().strip().lower()
        if not word:
            messagebox.showerror("Fehler", "Bitte Wort eingeben.", parent=root)
            return
        
        if word not in kv:
            messagebox.showerror("Fehler", f"'{word}' nicht im Modell.", parent=root)
            return
        
        top_n = int(ent_topn.get() or "20")
        similar = kv.most_similar(word, topn=top_n)
        
        df = pd.DataFrame(similar, columns=["word", "similarity"])
        df["rank"] = np.arange(1, len(df)+1)
        df = df[["rank", "word", "similarity"]]
        
        tree.delete(*tree.get_children())
        for _, r in df.iterrows():
            tree.insert("", "end", values=(int(r["rank"]), r["word"], round(r["similarity"], 4)))
        
        result["df"] = df
        btn_save.configure(state="normal")
    
    btn_save.configure(command=lambda: save_dataframe(result["df"], "Embeddings", f"sim_{ent_word.get()}", root))
    ttk.Button(frame, text="Suchen", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: EMBEDDINGS VERGLEICH
# =============================================================================

def build_tab_embed_compare(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """Tab: Vergleich von Wortvektoren mit gemeinsamen Nachbarn."""
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Embeddings Vergleich")

    row = 0
    info_frame = ttk.LabelFrame(frame, text="ℹ️ Vergleich zwischen Ausdrücken anhand gemeinsamer Embeddings", padding=6)
    info_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
    ttk.Label(info_frame).pack(anchor="w")
   
    row += 1
    ttk.Label(frame, text="Zentraler Ausdruck:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_central = create_entry(frame, width=30)
    ent_central.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Vergleichsausdrücke (Komma):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_comps = create_entry(frame, width=50)
    ent_comps.grid(row=row, column=1, columnspan=2, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    row += 1
    ttk.Label(frame, text="Top-N Nachbarn:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_topn = create_entry(frame, width=6)
    ent_topn.insert(0, "50")
    ent_topn.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    # Ergebnis-Tabelle (Übersicht)
    row += 1
    ttk.Label(frame, text="Übersicht:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    
    row += 1
    columns_main = ("vergleich", "score", "anzahl")
    tree_main = ttk.Treeview(frame, columns=columns_main, show="headings", height=8)
    for c, w in [("vergleich", 150), ("score", 100), ("anzahl", 100)]:
        tree_main.heading(c, text=c)
        tree_main.column(c, width=w, anchor="w")
    tree_main.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    
    scroll_main = ttk.Scrollbar(frame, orient="vertical", command=tree_main.yview)
    tree_main.configure(yscrollcommand=scroll_main.set)
    scroll_main.grid(row=row, column=3, sticky="ns")
    enable_treeview_sort(tree_main)
    
    # Gemeinsame Nachbarn (Detail-Liste)
    row += 1
    ttk.Label(frame, text="Gemeinsame Nachbarn (Klick oben):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    
    row += 1
    columns_detail = ("nachbar", f"sim_zentral", "sim_vergleich")
    tree_detail = ttk.Treeview(frame, columns=columns_detail, show="headings", height=10)
    for c, w in [("nachbar", 180), (f"sim_zentral", 120), ("sim_vergleich", 120)]:
        tree_detail.heading(c, text=c)
        tree_detail.column(c, width=w, anchor="w")
    tree_detail.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    scroll_detail = ttk.Scrollbar(frame, orient="vertical", command=tree_detail.yview)
    tree_detail.configure(yscrollcommand=scroll_detail.set)
    scroll_detail.grid(row=row, column=3, sticky="ns")
    enable_treeview_sort(tree_detail)
    
    result = {"df_main": None, "df_detail": None, "gemeinsame_cache": {}}
    
    row += 1
    btn_save_main = ttk.Button(frame, text="Übersicht speichern", state="disabled")
    btn_save_detail = ttk.Button(frame, text="Details speichern", state="disabled")
    btn_save_main.grid(row=row, column=1, sticky="w", padx=6, pady=6)
    btn_save_detail.grid(row=row, column=2, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            kv = MODEL.load_model()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        central = ent_central.get().strip().lower()
        comps = [w.strip().lower() for w in ent_comps.get().split(",") if w.strip()]
        
        if not central or not comps:
            messagebox.showerror("Fehler", "Bitte Zentral- und Vergleichsausdrücke eingeben.", parent=root)
            return
        
        if central not in kv:
            messagebox.showerror("Fehler", f"'{central}' nicht im Modell.", parent=root)
            return
        
        try:
            top_n = max(1, int(ent_topn.get()))
        except ValueError:
            top_n = 50
        
        from scipy.spatial.distance import cosine
        
        central_vec = kv[central]
        central_neighbors = dict(kv.most_similar(central, topn=top_n))
        
        result["gemeinsame_cache"].clear()
        tree_main.delete(*tree_main.get_children())
        tree_detail.delete(*tree_detail.get_children())
        
        rows_main = []
        threshold = 0.3
        
        for wort in comps:
            if wort not in kv:
                rows_main.append((wort, "N/A", 0))
                result["gemeinsame_cache"][wort] = []
                continue
            
            score = 1 - cosine(central_vec, kv[wort])
            wort_neighbors = dict(kv.most_similar(wort, topn=top_n))
            
            gemeinsame = []
            for gw in set(central_neighbors.keys()) & set(wort_neighbors.keys()):
                sim_a, sim_b = central_neighbors[gw], wort_neighbors[gw]
                if sim_a >= threshold and sim_b >= threshold:
                    gemeinsame.append((gw, round(sim_a, 4), round(sim_b, 4)))
            
            gemeinsame.sort(key=lambda x: -(x[1] + x[2]))
            result["gemeinsame_cache"][wort] = gemeinsame
            
            rows_main.append((wort, round(score, 4), len(gemeinsame)))
        
        # Übersicht anzeigen
        df_main = pd.DataFrame(rows_main, columns=["vergleich", "score", "anzahl"])
        for _, r in df_main.iterrows():
            tree_main.insert("", "end", values=(r["vergleich"], r["score"], r["anzahl"]))
        
        result["df_main"] = df_main
        btn_save_main.configure(state="normal")
    
    def on_select_main(event):
        sel = tree_main.selection()
        if not sel:
            return
        vals = tree_main.item(sel[0], "values")
        if not vals:
            return
        
        wort = str(vals[0])
        gemeinsame = result["gemeinsame_cache"].get(wort, [])
        
        tree_detail.delete(*tree_detail.get_children())
        for gw, sim_a, sim_b in gemeinsame:
            tree_detail.insert("", "end", values=(gw, sim_a, sim_b))
        
        # Detail DataFrame erstellen
        if gemeinsame:
            df_detail = pd.DataFrame(gemeinsame, columns=["nachbar", f"sim_{ent_central.get().strip()}", f"sim_{wort}"])
            result["df_detail"] = df_detail
            btn_save_detail.configure(state="normal")
        else:
            result["df_detail"] = None
            btn_save_detail.configure(state="disabled")
    
    tree_main.bind("<<TreeviewSelect>>", on_select_main)
    
    btn_save_main.configure(command=lambda: save_dataframe(result["df_main"], "EmbeddingsVergleich", f"{ent_central.get()}_uebersicht", root))
    btn_save_detail.configure(command=lambda: save_dataframe(result["df_detail"], "EmbeddingsVergleich", f"{ent_central.get()}_details", root) if result["df_detail"] is not None else None)
    
    ttk.Button(frame, text="Vergleichen", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: NETZWERK
# =============================================================================

def build_tab_network(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """Tab: Semantisches Netzwerk aus Word2Vec mit Kosinus-Schwelle."""
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Netzwerk")
    
    row = 0
    ttk.Label(frame, text="Wörter (Komma):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_words = create_entry(frame, width=50)
    ent_words.grid(row=row, column=1, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    row += 1
    ttk.Label(frame, text="Top-N Nachbarn:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_n = create_entry(frame, width=6)
    ent_n.insert(0, "8")
    ent_n.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Kosinus-Schwelle (0-1):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_threshold = create_entry(frame, width=6)
    ent_threshold.insert(0, "0.3")
    ent_threshold.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Auflösung:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    resolution_var = tk.StringVar(value="Klein")
    ttk.Combobox(frame, textvariable=resolution_var, values=["Klein", "Mittel", "Groß"], width=10, state="readonly").grid(
        row=row, column=1, sticky="w", padx=6, pady=4
    )
    
    result = {"fig": None}
    
    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled")
    btn_png.grid(row=row, column=2, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            kv = MODEL.load_model()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        words = [w.strip().lower() for w in ent_words.get().replace(",", " ").split() if w.strip()]
        words = [w for w in words if w in kv]
        if not words:
            messagebox.showerror("Fehler", "Keine Wörter im Modell.", parent=root)
            return
        
        try:
            n = max(1, int(ent_n.get()))
            threshold = float(ent_threshold.get())
            if not 0 <= threshold <= 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Fehler", "Parameter prüfen.", parent=root)
            return
        
        # Graph aufbauen
        import itertools
        G = nx.Graph()
        
        for word in words:
            G.add_node(word, is_seed=True)
        
        for word in words:
            for neighbor, sim in kv.most_similar(word, topn=n):
                if sim >= threshold:
                    G.add_node(neighbor, is_seed=False)
                    G.add_edge(word, neighbor, weight=sim)
        
        # Verbindungen zwischen Seed-Wörtern
        for w1, w2 in itertools.combinations(words, 2):
            sim = float(kv.similarity(w1, w2))
            if sim >= threshold:
                G.add_edge(w1, w2, weight=sim)
        
        if G.number_of_edges() == 0:
            messagebox.showinfo("Info", "Keine Verbindungen über Schwelle.", parent=root)
            return
        
        # Auflösung
        res = resolution_var.get()
        if res == "Klein":
            figsize, node_size, font_size = (14, 14), 300, 10
        elif res == "Mittel":
            figsize, node_size, font_size = (20, 16), 400, 12
        else:
            figsize, node_size, font_size = (28, 20), 500, 14
        
        pos = nx.spring_layout(G, seed=42, k=0.4)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        seeds = [n for n in G.nodes() if G.nodes[n].get("is_seed")]
        others = [n for n in G.nodes() if not G.nodes[n].get("is_seed")]
        
        nx.draw_networkx_nodes(G, pos, nodelist=seeds, node_color="coral", node_size=node_size*1.5, ax=ax, alpha=0.9)
        nx.draw_networkx_nodes(G, pos, nodelist=others, node_color="lightblue", node_size=node_size, ax=ax, alpha=0.7)
        
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=[w * 2 for w in edge_weights], alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)
        
        ax.set_title(f"Netzwerk: {', '.join(words[:5])}{'...' if len(words) > 5 else ''} | Top-N: {n} | Schwelle: {threshold}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        
        result["fig"] = fig
        btn_png.configure(state="normal")
    
    btn_png.configure(command=lambda: save_figure(result["fig"], "Netzwerk", f"network_{resolution_var.get()}", root))
    ttk.Button(frame, text="Netzwerk erzeugen", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: CLUSTER
# =============================================================================

def build_tab_cluster(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """Tab: UMAP-Clustering mit erweiterten Optionen."""
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Cluster")

    row = 0
    info_frame = ttk.LabelFrame(frame, text="ℹ️ Streudiagramm mit hierarchischem Clusterverfahren eines Termsets", padding=6)
    info_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
    ttk.Label(info_frame, text="💡 manuelle Auswahl der Cluster und Variation der UMAP-Parameter möglich", foreground="blue").pack(anchor="w")

    row += 1
    ttk.Label(frame, text="Termliste:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    lbl_terms = ttk.Label(frame, text=str(MODEL.termlist_path), foreground="blue")
    lbl_terms.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Button(frame, text="...", width=3, command=lambda: MODEL.choose_termlist(root, lbl_terms)).grid(row=row, column=2)
    
    row += 1
    ttk.Label(frame, text="Cluster (k):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_k = create_entry(frame, width=6)
    ent_k.insert(0, "5")
    ent_k.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    # UMAP-Parameter
    row += 1
    ttk.Label(frame, text="UMAP n_neighbors:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_neighbors = create_entry(frame, width=6)
    ent_neighbors.insert(0, "15")
    ent_neighbors.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="UMAP min_dist:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_mindist = create_entry(frame, width=6)
    ent_mindist.insert(0, "0.1")
    ent_mindist.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Auflösung:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    resolution_var = tk.StringVar(value="Klein")
    ttk.Combobox(frame, textvariable=resolution_var, values=["Klein", "Mittel", "Groß"], width=10, state="readonly").grid(
        row=row, column=1, sticky="w", padx=6, pady=4
    )
    
    row += 1
    show_labels_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Labels anzeigen", variable=show_labels_var).grid(row=row, column=0, sticky="w", padx=6, pady=4)
    
    # Info-Box
    row += 1
    info = tk.Text(frame, height=8, width=70, font=("Courier", 9))
    info.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    result = {"fig": None, "df": None}
    
    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled")
    btn_csv = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_png.grid(row=row, column=2, sticky="e", padx=6, pady=6)
    btn_csv.grid(row=row, column=1, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            kv = MODEL.load_model()
            df_terms = read_csv_auto(MODEL.termlist_path)
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        all_terms = set()
        for c in df_terms.columns:
            all_terms.update(df_terms[c].dropna().astype(str).str.lower().tolist())
        
        terms = [t for t in all_terms if t in kv]
        if len(terms) < 3:
            messagebox.showerror("Fehler", "Zu wenige Terme.", parent=root)
            return
        
        try:
            k = min(int(ent_k.get() or "5"), len(terms))
            n_neighbors = min(int(ent_neighbors.get() or "15"), len(terms) - 1)
            min_dist = float(ent_mindist.get() or "0.1")
        except ValueError:
            k, n_neighbors, min_dist = 5, 15, 0.1
        
        vectors = np.array([kv[t] for t in terms])
        
        clustering = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = clustering.fit_predict(vectors)
        
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=42)
        coords = reducer.fit_transform(vectors)
        
        # Auflösung
        res = resolution_var.get()
        if res == "Klein":
            figsize, marker_size, font_size = (12, 8), 80, 8
        elif res == "Mittel":
            figsize, marker_size, font_size = (16, 12), 100, 10
        else:
            figsize, marker_size, font_size = (20, 16), 120, 12
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        for i in range(k):
            mask = labels == i
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=f"Cluster {i+1}", s=marker_size, alpha=0.7)
        
        if show_labels_var.get():
            for i, t in enumerate(terms):
                ax.annotate(t, (coords[i, 0], coords[i, 1]), fontsize=font_size, alpha=0.8)
        
        ax.set_title(f"UMAP-Clustering (k={k}, n={len(terms)})")
        ax.legend(loc="upper right")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        
        # Info
        info.delete(1.0, tk.END)
        info.insert(tk.END, f"Terme: {len(terms)} | Cluster: {k}\n\n")
        from collections import defaultdict
        clusters = defaultdict(list)
        for t, l in zip(terms, labels):
            clusters[l].append(t)
        for cid in sorted(clusters):
            info.insert(tk.END, f"Cluster {cid+1}: {', '.join(clusters[cid][:15])}\n")
        
        # DataFrame
        df_result = pd.DataFrame({"term": terms, "cluster": labels + 1, "x": coords[:, 0], "y": coords[:, 1]})
        
        result["fig"] = fig
        result["df"] = df_result
        btn_png.configure(state="normal")
        btn_csv.configure(state="normal")
    
    btn_png.configure(command=lambda: save_figure(result["fig"], "Cluster", f"cluster_k{ent_k.get()}", root))
    btn_csv.configure(command=lambda: save_dataframe(result["df"], "Cluster", f"cluster_k{ent_k.get()}", root))
    ttk.Button(frame, text="Clustern", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: WORTWOLKE
# =============================================================================

def build_tab_wordcloud(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """Tab: Wortwolke aus TF-IDF mit Termset-Kategorien."""
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Wortwolke")
    
    if not WORDCLOUD_AVAILABLE:
        ttk.Label(frame, text="⚠️ wordcloud nicht installiert\npip install wordcloud", foreground="red").pack(pady=20)
        return
    
    row = 0
    ttk.Label(frame, text="Termliste:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    lbl_terms = ttk.Label(frame, text=str(MODEL.termlist_path), foreground="blue")
    lbl_terms.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Button(frame, text="...", width=3, command=lambda: MODEL.choose_termlist(root, lbl_terms)).grid(row=row, column=2)
    
    row += 1
    ttk.Label(frame, text="TF-IDF Datei:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    tfidf_var = tk.StringVar(value=str(DATA.path_tfidf))
    create_entry(frame, width=50, textvariable=tfidf_var).grid(row=row, column=1, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    def browse_tfidf():
        p = filedialog.askopenfilename(parent=root, filetypes=[("CSV", "*.csv")])
        if p:
            tfidf_var.set(p)
    ttk.Button(frame, text="...", width=3, command=browse_tfidf).grid(row=row, column=2)
    
    row += 1
    ttk.Label(frame, text="Farbschema (cmap):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    cmap_var = tk.StringVar(value="tab10")
    create_entry(frame, textvariable=cmap_var, width=12).grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    whole_word_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Nur ganze Wörter", variable=whole_word_var).grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=4)
    
    result = {"fig": None}
    
    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled")
    btn_png.grid(row=row, column=2, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            df_terms = read_csv_auto(MODEL.termlist_path)
        except Exception as e:
            messagebox.showerror("Fehler", f"Termliste: {e}", parent=root)
            return
        
        try:
            df_tfidf = read_csv_auto(Path(tfidf_var.get()))
            term_cols = get_term_columns(df_tfidf)
            tfidf_avg = df_tfidf[term_cols].mean().reset_index()
            tfidf_avg.columns = ["word", "tfidf_avg"]
        except Exception as e:
            messagebox.showerror("Fehler", f"TF-IDF: {e}", parent=root)
            return
        
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        word_infos = []
        for tag in df_terms.columns:
            for word in df_terms[tag].dropna():
                word = str(word).strip().lower()
                if whole_word_var.get():
                    val = tfidf_avg.loc[tfidf_avg["word"] == word, "tfidf_avg"]
                else:
                    val = tfidf_avg.loc[tfidf_avg["word"].str.contains(word, case=False, na=False, regex=False), "tfidf_avg"]
                if not val.empty:
                    word_infos.append({"word": word, "tag": tag, "tfidf": float(val.values[0])})
        
        if not word_infos:
            messagebox.showinfo("Info", "Keine Überschneidung Termset ↔ TF-IDF.", parent=root)
            return
        
        df_combined = pd.DataFrame(word_infos)
        tags = df_combined["tag"].unique()
        
        try:
            colormap = cm.get_cmap(cmap_var.get(), len(tags))
        except Exception:
            colormap = cm.get_cmap("tab10", len(tags))
        
        tag_colors = {tag: mcolors.rgb2hex(colormap(i)) for i, tag in enumerate(tags)}
        
        def color_func(word, *args, **kwargs):
            row = df_combined[df_combined["word"] == word]
            if not row.empty:
                return tag_colors.get(row.iloc[0]["tag"], "black")
            return "black"
        
        word_size_dict = df_combined.groupby("word")["tfidf"].max().to_dict()
        word_size_scaled = {w: np.log(v + 1.0) for w, v in word_size_dict.items()}
        
        wc = WordCloud(width=1200, height=600, background_color="white", prefer_horizontal=1.0)
        wc.generate_from_frequencies(word_size_scaled)
        wc.recolor(color_func=color_func)
        
        fig = plt.figure(figsize=(16, 8))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Wortwolke: {Path(MODEL.termlist_path).stem}")
        plt.tight_layout()
        plt.show()
        
        result["fig"] = fig
        btn_png.configure(state="normal")
    
    btn_png.configure(command=lambda: save_figure(result["fig"], "Wortwolke", Path(MODEL.termlist_path).stem, root))
    ttk.Button(frame, text="Erzeugen", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: DENDROGRAMM
# =============================================================================

def build_tab_dendrogram(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """Tab: Dendrogramme pro Cluster mit automatischem Speichern."""
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Dendrogramme")

    row = 0
    info_frame = ttk.LabelFrame(frame, text="ℹ️ Hierarchisches Clustering eines Termsets als Dendrogramm", padding=6)
    info_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
    ttk.Label(info_frame, text="💡 Manuelle Einstellung der Cluster. \nVoreinstellung für die Verknüpfung der Ausdrücke: 'average'.", foreground="blue").pack(anchor="w")

    row += 0
    ttk.Label(frame, text="Termliste:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    lbl_terms = ttk.Label(frame, text=str(MODEL.termlist_path), foreground="blue")
    lbl_terms.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Button(frame, text="...", width=3, command=lambda: MODEL.choose_termlist(root, lbl_terms)).grid(row=row, column=2)
    
    row += 1
    ttk.Label(frame, text="Clusteranzahl (k):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_k = create_entry(frame, width=6)
    ent_k.insert(0, "3")
    ent_k.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Linkage:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    linkage_var = tk.StringVar(value="average")
    ttk.Combobox(frame, textvariable=linkage_var, values=["average", "ward", "complete", "single"], 
                 width=10, state="readonly").grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    info = tk.Text(frame, height=10, width=70, font=("Courier", 9))
    info.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    def compute():
        try:
            kv = MODEL.load_model()
            df_terms = read_csv_auto(MODEL.termlist_path)
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        try:
            k = max(1, int(ent_k.get()))
        except ValueError:
            k = 3
        
        # Terme sammeln
        all_terms = {}
        for tag in df_terms.columns:
            for entry in df_terms[tag].dropna().astype(str):
                word = re.sub(r"\s*\(.*?\)\s*$", "", entry.strip()).lower()
                if word and word in kv:
                    all_terms[word] = entry.strip()
        
        if len(all_terms) < 2:
            messagebox.showerror("Fehler", "Zu wenige Terme im Modell.", parent=root)
            return
        
        words = list(all_terms.keys())
        labels = [all_terms[w] for w in words]
        vectors = np.array([kv[w] for w in words])
        
        # Clustering
        k_eff = min(k, len(words))
        if k_eff == 1:
            cluster_labels = np.zeros(len(words), dtype=int)
        else:
            clustering = AgglomerativeClustering(n_clusters=k_eff, linkage="ward")
            cluster_labels = clustering.fit_predict(vectors)
        
        # Output-Verzeichnis
        termset_name = Path(MODEL.termlist_path).stem
        out_dir = EXPLORATION_DIR / "dendrograms" / f"Dendrogramme_k{k_eff}_{termset_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        created = []
        method = linkage_var.get()
        
        for cluster_id in range(k_eff):
            idx = np.where(cluster_labels == cluster_id)[0]
            if len(idx) < 2:
                continue
            
            cluster_labels_list = [labels[i] for i in idx]
            cluster_vecs = vectors[idx]
            
            # Linkage berechnen
            if method == "ward":
                Z = linkage(cluster_vecs, method="ward")
            else:
                from scipy.spatial.distance import pdist
                Z = linkage(pdist(cluster_vecs, metric="cosine"), method=method)
            
            # Plot
            fig_height = max(6, len(cluster_labels_list) * 0.3)
            plt.figure(figsize=(10, fig_height))
            dendrogram(Z, labels=cluster_labels_list, orientation="right", leaf_font_size=9)
            plt.title(f"Dendrogramm — Cluster {cluster_id}")
            plt.tight_layout()
            
            filename = f"dendro_k{k_eff}_cluster{cluster_id}_{method}.png"
            filepath = out_dir / filename
            plt.savefig(filepath, dpi=300)
            plt.close()
            created.append(filepath)
        
        # Info anzeigen
        info.delete(1.0, tk.END)
        info.insert(tk.END, f"Dendrogramme gespeichert in:\n{out_dir}\n\n")
        for p in created:
            info.insert(tk.END, f"  • {p.name}\n")
        
        info.insert(tk.END, f"\n{len(created)} Dendrogramme erstellt.")
    
    row += 1
    ttk.Button(frame, text="Dendrogramme erzeugen", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: STREUDIAGRAMM
# =============================================================================

def build_tab_scatter(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """Tab: UMAP-Streudiagramm mit Plotly (interaktiv) oder Matplotlib (statisch)."""
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Streudiagramm 1")
    
    row = 0
    
    # Info
    info_frame = ttk.LabelFrame(frame, text="ℹ️ Streudiagramm mit hierarchischem Clusterverfahren oder Metadaten", padding=6)
    info_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
    ttk.Label(info_frame, text="💡 Einstellung entweder Anzeige 'Hierarchisches Clustering' oder 'Legende-Metadatum'. \nEin Metadatum wird markiert und differenziert.", foreground="blue").pack(anchor="w")
    
    # Dateiauswahl
    row += 1
    ttk.Label(frame, text="Kosinus-Matrix:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    cosine_var = tk.StringVar(value=str(DATA.path_cosine))
    create_entry(frame, width=60, textvariable=cosine_var).grid(row=row, column=1, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    def browse_cos():
        p = filedialog.askopenfilename(parent=root, filetypes=[("CSV", "*.csv")])
        if p:
            cosine_var.set(p)
    ttk.Button(frame, text="...", width=3, command=browse_cos).grid(row=row, column=2)
    
    row += 1
    ttk.Label(frame, text="Metadaten:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    meta_var = tk.StringVar(value=str(DATA.path_metadata))
    create_entry(frame, width=60, textvariable=meta_var).grid(row=row, column=1, sticky="we", padx=6, pady=4)
    
    def browse_meta():
        p = filedialog.askopenfilename(parent=root, filetypes=[("CSV", "*.csv")])
        if p:
            meta_var.set(p)
    ttk.Button(frame, text="...", width=3, command=browse_meta).grid(row=row, column=2)
    
    # Separator
    row += 1
    ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
    
    # Clustering
    row += 1
    clustering_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame, text="Hierarchisches Clustering aktivieren", variable=clustering_var).grid(
        row=row, column=0, columnspan=2, sticky="w", padx=6, pady=4
    )
    
    row += 1
    ttk.Label(frame, text="  Clusteranzahl (k):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_k = create_entry(frame, width=6)
    ent_k.insert(0, "5")
    ent_k.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="  Linkage:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    linkage_var = tk.StringVar(value="average")
    ttk.Combobox(frame, textvariable=linkage_var, values=["ward", "average", "complete", "single"],
                 width=10, state="readonly").grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    # UMAP-Parameter
    row += 1
    umap_frame = ttk.LabelFrame(frame, text="UMAP-Parameter", padding=6)
    umap_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=4)
    
    ttk.Label(umap_frame, text="n_neighbors:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ent_neighbors = create_entry(umap_frame, width=6)
    ent_neighbors.insert(0, "15")
    ent_neighbors.grid(row=0, column=1, sticky="w", padx=6, pady=4)
    
    ttk.Label(umap_frame, text="min_dist:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    ent_dist = create_entry(umap_frame, width=6)
    ent_dist.insert(0, "0.1")
    ent_dist.grid(row=1, column=1, sticky="w", padx=6, pady=4)
    
    # Legende-Metadaten (wenn kein Clustering)
    row += 1
    ttk.Label(frame, text="Legende-Metadatum (+ _id):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    legend_meta_var = tk.StringVar(value="textclass")
    legend_combo = ttk.Combobox(frame, textvariable=legend_meta_var, width=15, state="readonly")
    legend_combo.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="(Färbung ohne Clustering)", foreground="gray").grid(row=row, column=2, sticky="w", padx=6)
    
    def update_legend_combo():
        """Aktualisiert die Metadaten-Dropdown-Liste aus metadata.csv."""
        try:
            meta_path = Path(meta_var.get())
            if meta_path.exists():
                df = read_csv_auto(meta_path)
                id_cols = {"_id", "id", "doc_id", "document_id"}
                content_cols_exact = {"content_stop", "content", "text", "clean_text", "cleaned_text"}
                
                cols = []
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in id_cols:
                        continue
                    if col_lower in content_cols_exact:
                        continue
                    if col_lower in {"content", "text", "clean_text", "cleaned_text"}:
                        continue
                    if col_lower.startswith("text_"):   # nur text_*, nicht textclass
                        continue
                    cols.append(col)

                cols = sorted(cols, key=str.lower)
                
                legend_combo['values'] = cols
                if "textclass" in cols:
                    legend_meta_var.set("textclass")
                elif cols:
                    legend_meta_var.set(cols[0])
        except Exception:
            legend_combo['values'] = ["author_surname", "title", "year_final", "textclass"]
    
    # Initial aktualisieren nach kurzer Verzögerung
    root.after(500, update_legend_combo)
    
    row += 1
    ttk.Label(frame, text="Marker-Größe:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_size = create_entry(frame, width=6)
    ent_size.insert(0, "8")
    ent_size.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    interactive_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Interaktiv (Plotly) - sonst statisch (PNG)", variable=interactive_var).grid(
        row=row, column=0, columnspan=2, sticky="w", padx=6, pady=4
    )
    
    # Info-Box
    row += 1
    info_text = tk.Text(frame, height=8, width=70, font=("Courier", 9))
    info_text.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=4)
    frame.rowconfigure(row, weight=1)
    
    result = {"fig": None, "df": None, "html_path": None}
    
    row += 1
    btn_save_html = ttk.Button(frame, text="HTML speichern", state="disabled")
    btn_save_csv = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_save_png = ttk.Button(frame, text="PNG speichern", state="disabled")
    btn_open = ttk.Button(frame, text="🌐 Im Browser öffnen", state="disabled")
    
    # Buttons-Frame für bessere Anordnung
    row += 1
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=6)
    
    ttk.Button(btn_frame, text="🔄 Berechnen", command=lambda: compute()).pack(side="left", padx=(0, 10))
    
    btn_save_html = ttk.Button(btn_frame, text="HTML speichern", state="disabled")
    btn_save_csv  = ttk.Button(btn_frame, text="CSV speichern", state="disabled")
    btn_save_png  = ttk.Button(btn_frame, text="PNG speichern", state="disabled")
    btn_open      = ttk.Button(btn_frame, text="🌐 Im Browser öffnen", state="disabled")
    
    btn_save_html.pack(side="left", padx=2)
    btn_save_csv.pack(side="left", padx=2)
    btn_save_png.pack(side="left", padx=2)
    btn_open.pack(side="left", padx=2)
    
    def compute():
        info_text.delete(1.0, tk.END)
        info_text.insert(tk.END, "🔄 Lade Daten...\n")
        root.update_idletasks()
        
        # Kosinus-Matrix laden
        try:
            cos_path = Path(cosine_var.get())
            cos_df = read_csv_auto(cos_path, index_col=0)
            cosine_matrix = cos_df.values
            doc_ids = [str(d) for d in cos_df.index.tolist()]
            info_text.insert(tk.END, f"✔ Matrix: {cosine_matrix.shape[0]} × {cosine_matrix.shape[1]}\n")
        except Exception as e:
            messagebox.showerror("Fehler", f"Kosinus-Matrix: {e}", parent=root)
            return
        
        # Metadaten laden
        try:
            meta_path = Path(meta_var.get())
            metadata_df = read_csv_auto(meta_path)
            # Flexible ID-Spalten-Erkennung
            id_col = find_column(metadata_df, ["_id", "id", "doc_id", "document_id", "ID", "Id"])
            if id_col:
                # Normalisiere zu "doc_id"
                metadata_df["doc_id"] = metadata_df[id_col].astype(str)
            else:
                raise ValueError(f"Keine ID-Spalte gefunden. Vorhanden: {[repr(c) for c in metadata_df.columns[:10]]}")
            metadata_df = metadata_df[metadata_df["doc_id"].isin(doc_ids)]
            metadata_df = metadata_df.set_index("doc_id").reindex(doc_ids).reset_index()
            info_text.insert(tk.END, f"✔ Metadaten: {len(metadata_df)} Docs (ID-Spalte: '{id_col}')\n")
        except Exception as e:
            messagebox.showerror("Fehler", f"Metadaten: {e}", parent=root)
            return
        
        # Parameter
        try:
            n_neighbors = min(int(ent_neighbors.get()), len(doc_ids) - 1)
            min_dist = float(ent_dist.get())
            marker_size = int(ent_size.get())
        except ValueError:
            n_neighbors, min_dist, marker_size = 15, 0.1, 8
        
        # Distanzmatrix
        distance_matrix = 1 - cosine_matrix
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.clip(distance_matrix, 0, None)
        
        # UMAP
        info_text.insert(tk.END, f"🔄 UMAP (n={n_neighbors}, d={min_dist})...\n")
        root.update_idletasks()
        
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, 
                           metric='precomputed', random_state=42)
        coords = reducer.fit_transform(distance_matrix)
        info_text.insert(tk.END, "✔ UMAP fertig\n")
        
        # Clustering
        clusters = None
        k_eff = 0
        if clustering_var.get():
            k = int(ent_k.get())
            k_eff = min(k, len(doc_ids))
            method = linkage_var.get()
            info_text.insert(tk.END, f"🔄 Clustering (k={k_eff}, {method})...\n")
            
            if method == "ward":
                from sklearn.manifold import MDS
                mds = MDS(n_components=min(50, len(distance_matrix)-1), dissimilarity='precomputed', random_state=42)
                X_embedded = mds.fit_transform(distance_matrix)
                clusters = AgglomerativeClustering(n_clusters=k_eff, linkage="ward").fit_predict(X_embedded)
            else:
                from scipy.spatial.distance import squareform
                from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
                condensed = squareform(distance_matrix, checks=False)
                Z = scipy_linkage(condensed, method=method)
                clusters = fcluster(Z, k_eff, criterion='maxclust') - 1
            
            info_text.insert(tk.END, "✔ Clustering fertig\n")
        
        # DataFrame
        umap_df = pd.DataFrame({"UMAP-1": coords[:, 0], "UMAP-2": coords[:, 1]})
        umap_df = umap_df.join(metadata_df.reset_index(drop=True))
        
        if clusters is not None:
            umap_df["cluster"] = clusters
            umap_df["cluster_label"] = "Cluster " + umap_df["cluster"].astype(str)
        
        # Färbung und Legende bestimmen
        color_column = None
        legend_meta = legend_meta_var.get()
        
        if clustering_var.get() and clusters is not None:
            # Mit Clustering: Farbe nach Cluster
            color_column = "cluster_label"
        else:
            # Ohne Clustering: Farbe nach gewähltem Metadatum, Legende = meta - _id
            if legend_meta and legend_meta in umap_df.columns:
                # Legende: meta - _id
                umap_df["legend_label"] = umap_df[legend_meta].fillna("?").astype(str) + " - " + umap_df["doc_id"].astype(str)
                # Färbung nach dem Metadatum
                color_column = legend_meta
            else:
                umap_df["legend_label"] = umap_df["doc_id"].astype(str)
                color_column = "legend_label"
        
        # Plot
        if interactive_var.get():
            # Plotly interaktiv
            try:
                import plotly.express as px
                
                umap_df["hover_text"] = (
                    umap_df["doc_id"].astype(str) + "<br>" +
                    "Author: " + umap_df.get("author_surname", pd.Series("N/A")).fillna("N/A").astype(str) + "<br>" +
                    "Title: " + umap_df.get("title", pd.Series("N/A")).fillna("N/A").astype(str)
                )
                
                # Bestimme was für Farbe und Legende verwendet wird
                if clustering_var.get() and clusters is not None:
                    plot_color = "cluster_label"
                else:
                    plot_color = color_column
                
                fig = px.scatter(
                    umap_df, x="UMAP-1", y="UMAP-2",
                    color=plot_color,
                    hover_name="hover_text",
                    opacity=0.7
                )
                
                title = f"UMAP ({len(umap_df)} Docs)"
                if clusters is not None:
                    title += f" | k={k_eff}"
                
                fig.update_layout(title=title, width=1200, height=800)
                fig.update_traces(marker=dict(size=marker_size))
                
                result["fig"] = fig
                result["df"] = umap_df
                
                btn_save_html.configure(state="normal")
                btn_save_csv.configure(state="normal")
                
                # HTML temporär speichern und Browser öffnen
                import tempfile
                import webbrowser
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
                fig.write_html(temp_file.name)
                temp_file.close()
                result["html_path"] = temp_file.name
                
                info_text.insert(tk.END, f"💡 HTML: {temp_file.name}\n")
                webbrowser.open('file://' + temp_file.name)
                info_text.insert(tk.END, "✔ Browser geöffnet!\n")
                btn_open.configure(state="normal")
                
            except ImportError:
                messagebox.showerror("Fehler", "Plotly nicht installiert. Wähle statisches PNG.", parent=root)
                return
        else:
            # Matplotlib statisch
            fig, ax = plt.subplots(figsize=(12, 10))
            
            legend_meta = legend_meta_var.get()
            
            if clustering_var.get() and clusters is not None:
                # Mit Clustering: Farbe nach Cluster
                categories = umap_df["cluster_label"].fillna("unknown").astype(str)
                unique_cats = categories.unique()
                colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_cats), 20)))
                color_map = dict(zip(unique_cats, colors))
                
                for cat in unique_cats:
                    mask = categories == cat
                    ax.scatter(umap_df.loc[mask, "UMAP-1"], umap_df.loc[mask, "UMAP-2"],
                               c=[color_map[cat]], label=str(cat)[:35], s=marker_size, alpha=0.7)
                
                if len(unique_cats) <= 20:
                    ax.legend(loc="upper right", fontsize=8)
            elif legend_meta and legend_meta in umap_df.columns:
                # Ohne Clustering: Farbe nach Metadatum, Legende = meta - _id
                categories = umap_df[legend_meta].fillna("?").astype(str)
                unique_cats = categories.unique()
                colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_cats), 20)))
                color_map = dict(zip(unique_cats, colors))
                
                for cat in unique_cats:
                    mask = categories == cat
                    # Legende zeigt meta - _id für jeden Punkt dieser Kategorie
                    for idx in umap_df[mask].index:
                        row_data = umap_df.loc[idx]
                        label = f"{cat} - {row_data['doc_id']}"
                        ax.scatter(row_data["UMAP-1"], row_data["UMAP-2"],
                                   c=[color_map[cat]], label=label[:40], s=marker_size, alpha=0.7)
                
                # Legende nur bei nicht zu vielen Dokumenten
                if len(umap_df) <= 50:
                    ax.legend(loc="upper right", fontsize=6, ncol=2)
            else:
                ax.scatter(umap_df["UMAP-1"], umap_df["UMAP-2"], s=marker_size, alpha=0.7)
            
            title = f"UMAP ({len(umap_df)} Docs)"
            if clusters is not None:
                title += f" | k={k_eff}"
            
            ax.set_title(title)
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            plt.tight_layout()
            plt.show()
            
            result["fig"] = fig
            result["df"] = umap_df
            btn_save_png.configure(state="normal")
            btn_save_csv.configure(state="normal")
        
        info_text.insert(tk.END, "✔ Fertig!\n")
    
    def save_html():
        if result["fig"] is None:
            return
        path = filedialog.asksaveasfilename(parent=root, defaultextension=".html", filetypes=[("HTML", "*.html")])
        if path:
            result["fig"].write_html(path)
            messagebox.showinfo("Gespeichert", path, parent=root)
    
    def save_csv():
        if result["df"] is None:
            return
        save_dataframe(result["df"], "Streudiagramm", "umap_data", root)
    
    def save_png():
        if result["fig"] is None:
            return
        save_figure(result["fig"], "Streudiagramm", "umap_scatter", root)
    
    def open_browser():
        if result.get("html_path"):
            import webbrowser
            webbrowser.open('file://' + result["html_path"])
    
    btn_save_html.configure(command=save_html)
    btn_save_csv.configure(command=save_csv)
    btn_save_png.configure(command=save_png)
    btn_open.configure(command=open_browser)


# =============================================================================
# TAB: STREUDIAGRAMM MIT SCROLLBARER LEGENDE
# =============================================================================

def build_tab_scatter_legend(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """
    Tab: UMAP-Streudiagramm mit scrollbarer Legende.
    
    Jeder Text wird mit seiner ID und ausgewählten Metadaten in der Legende
    repräsentiert. Farben entsprechen den Kategorien der ausgewählten Metadaten.
    """
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Streudiagramm 2")
    
    row = 0
    
    # Info-Header
    info_frame = ttk.LabelFrame(frame, text="ℹ️ Streudiagramm mit Textlegende", padding=6)
    info_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=6, pady=6)
    ttk.Label(info_frame, text="💡 Jeder Text wird mit ID und Metadaten in einer scrollbaren Legende dargestellt. \nDer Fokus auf die Metadaten erfolgt mit dem Drop-Down-Menü.", 
              foreground="blue").pack(anchor="w")
    
    # Dateiauswahl: Kosinus-Matrix
    row += 1
    ttk.Label(frame, text="Kosinus-Matrix:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    cosine_var = tk.StringVar(value=str(DATA.path_cosine))
    create_entry(frame, width=50, textvariable=cosine_var).grid(row=row, column=1, columnspan=2, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    def browse_cos():
        p = filedialog.askopenfilename(parent=root, filetypes=[("CSV", "*.csv")])
        if p:
            cosine_var.set(p)
    ttk.Button(frame, text="...", width=3, command=browse_cos).grid(row=row, column=3)
    
    # Dateiauswahl: Metadaten
    row += 1
    ttk.Label(frame, text="Metadaten:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    meta_var = tk.StringVar(value=str(DATA.path_metadata))
    create_entry(frame, width=50, textvariable=meta_var).grid(row=row, column=1, columnspan=2, sticky="we", padx=6, pady=4)
    
    def browse_meta():
        p = filedialog.askopenfilename(parent=root, filetypes=[("CSV", "*.csv")])
        if p:
            meta_var.set(p)
            update_metadata_combo()
    ttk.Button(frame, text="...", width=3, command=browse_meta).grid(row=row, column=3)
    
    # Separator
    row += 1
    ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=6)
    
    # Metadaten-Auswahl für Färbung
    row += 1
    ttk.Label(frame, text="Färbung nach Metadaten:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    color_meta_var = tk.StringVar(value="textclass")
    color_combo = ttk.Combobox(frame, textvariable=color_meta_var, width=20, state="readonly")
    color_combo.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    def update_metadata_combo():
        """Aktualisiert die Metadaten-Dropdown-Liste aus metadata.csv."""
        try:
            # Versuche Metadaten zu laden falls noch nicht geschehen
            meta_path = Path(meta_var.get())
            if meta_path.exists():
                df = read_csv_auto(meta_path)
                # ID- und Content-Spalten ausschließen

                id_cols = {"_id", "id", "doc_id", "document_id"}
                content_cols_exact = {"content_stop", "content", "text", "clean_text", "cleaned_text"}
                
                cols = []
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in id_cols:
                        continue
                    if col_lower in content_cols_exact:
                        continue
                    if col_lower in {"content", "text", "clean_text", "cleaned_text"}:
                        continue
                    if col_lower.startswith("text_"):   # nur text_*, nicht textclass
                        continue
                    cols.append(col)
                
                cols = sorted(cols, key=str.lower)
                color_combo['values'] = cols

                # Setze textclass als Default wenn vorhanden
                if "textclass" in cols:
                    color_meta_var.set("textclass")
                elif cols:
                    color_meta_var.set(cols[0])
        except Exception:
            # Fallback
            color_combo['values'] = ["author_surname", "title", "year_final", "textclass"]
    
    # Button zum Aktualisieren der Liste
    ttk.Button(frame, text="🔄", width=3, command=update_metadata_combo).grid(row=row, column=2, sticky="w", padx=2)
    
    # Initial aktualisieren nach kurzer Verzögerung
    root.after(500, update_metadata_combo)
    
    # Metadaten für Legende
    row += 1
    ttk.Label(frame, text="In Legende anzeigen:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    legend_meta_var = tk.StringVar(value="author_surname, title")
    create_entry(frame, width=40, textvariable=legend_meta_var).grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="(Komma-getrennt)", foreground="gray").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    
    # UMAP-Parameter
    row += 1
    umap_frame = ttk.LabelFrame(frame, text="UMAP-Parameter", padding=6)
    umap_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
    
    ttk.Label(umap_frame, text="n_neighbors:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ent_neighbors = create_entry(umap_frame, width=6)
    ent_neighbors.insert(0, "15")
    ent_neighbors.grid(row=0, column=1, sticky="w", padx=6, pady=4)
    
    ttk.Label(umap_frame, text="min_dist:").grid(row=0, column=2, sticky="w", padx=6, pady=4)
    ent_dist = create_entry(umap_frame, width=6)
    ent_dist.insert(0, "0.1")
    ent_dist.grid(row=0, column=3, sticky="w", padx=6, pady=4)
    
    ttk.Label(umap_frame, text="Marker-Größe:").grid(row=0, column=4, sticky="w", padx=6, pady=4)
    ent_size = create_entry(umap_frame, width=6)
    ent_size.insert(0, "8")
    ent_size.grid(row=0, column=5, sticky="w", padx=6, pady=4)
    
    # Optionen
    row += 1
    opt_frame = ttk.Frame(frame)
    opt_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
    
    show_ids_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(opt_frame, text="IDs in Legende anzeigen", variable=show_ids_var).pack(side="left", padx=6)
    
    truncate_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(opt_frame, text="Titel kürzen (max. 30 Zeichen)", variable=truncate_var).pack(side="left", padx=6)
    
    interactive_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(opt_frame, text="Interaktiv (Plotly)", variable=interactive_var).pack(side="left", padx=6)
    
    # Info-Box
    row += 1
    info_text = tk.Text(frame, height=6, width=70, font=("Courier", 9))
    info_text.grid(row=row, column=0, columnspan=4, sticky="nsew", padx=6, pady=4)
    
    result = {"fig": None, "df": None}
    
    # Buttons
    row += 1
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=6, pady=6)
    
    btn_compute = ttk.Button(btn_frame, text="🔄 Berechnen")
    btn_compute.pack(side="left", padx=6)
    
    btn_save_html = ttk.Button(btn_frame, text="HTML speichern", state="disabled")
    btn_save_html.pack(side="left", padx=6)
    
    btn_save_csv = ttk.Button(btn_frame, text="CSV speichern", state="disabled")
    btn_save_csv.pack(side="left", padx=6)
    
    btn_save_png = ttk.Button(btn_frame, text="PNG speichern", state="disabled")
    btn_save_png.pack(side="left", padx=6)
    
    def compute():
        info_text.delete(1.0, tk.END)
        info_text.insert(tk.END, "🔄 Lade Daten...\n")
        root.update_idletasks()
        
        # Kosinus-Matrix laden
        try:
            cos_path = Path(cosine_var.get())
            cos_df = read_csv_auto(cos_path, index_col=0)
            cosine_matrix = cos_df.values
            doc_ids = [str(d) for d in cos_df.index.tolist()]
            info_text.insert(tk.END, f"✔ Matrix: {cosine_matrix.shape[0]} × {cosine_matrix.shape[1]}\n")
        except Exception as e:
            messagebox.showerror("Fehler", f"Kosinus-Matrix: {e}", parent=root)
            return
        
        # Metadaten laden
        try:
            meta_path = Path(meta_var.get())
            metadata_df = read_csv_auto(meta_path)
            id_col = find_column(metadata_df, ["_id", "id", "doc_id", "document_id", "ID", "Id"])
            if id_col:
                metadata_df["doc_id"] = metadata_df[id_col].astype(str)
            else:
                raise ValueError(f"Keine ID-Spalte gefunden.")
            metadata_df = metadata_df[metadata_df["doc_id"].isin(doc_ids)]
            metadata_df = metadata_df.set_index("doc_id").reindex(doc_ids).reset_index()
            info_text.insert(tk.END, f"✔ Metadaten: {len(metadata_df)} Docs\n")
        except Exception as e:
            messagebox.showerror("Fehler", f"Metadaten: {e}", parent=root)
            return
        
        # Parameter
        try:
            n_neighbors = min(int(ent_neighbors.get()), len(doc_ids) - 1)
            min_dist = float(ent_dist.get())
            marker_size = int(ent_size.get())
        except ValueError:
            n_neighbors, min_dist, marker_size = 15, 0.1, 8
        
        # Distanzmatrix
        distance_matrix = 1 - cosine_matrix
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.clip(distance_matrix, 0, None)
        
        # UMAP
        info_text.insert(tk.END, f"🔄 UMAP (n={n_neighbors}, d={min_dist})...\n")
        root.update_idletasks()
        
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, 
                           metric='precomputed', random_state=42)
        coords = reducer.fit_transform(distance_matrix)
        info_text.insert(tk.END, "✔ UMAP fertig\n")
        
        # DataFrame erstellen
        umap_df = pd.DataFrame({"UMAP-1": coords[:, 0], "UMAP-2": coords[:, 1]})
        umap_df = umap_df.join(metadata_df.reset_index(drop=True))
        
        # Färbung nach ausgewählter Metadaten-Spalte
        color_column = color_meta_var.get()
        if color_column not in umap_df.columns:
            color_column = None
        
        # Legendentext erstellen
        legend_cols = [c.strip() for c in legend_meta_var.get().split(",") if c.strip()]
        legend_cols = [c for c in legend_cols if c in umap_df.columns]
        
        def create_legend_text(row):
            parts = []
            if show_ids_var.get():
                parts.append(f"[{row.get('doc_id', 'N/A')}]")
            for col in legend_cols:
                val = str(row.get(col, 'N/A'))
                if truncate_var.get() and len(val) > 30:
                    val = val[:27] + "..."
                parts.append(val)
            return " | ".join(parts) if parts else str(row.get('doc_id', 'N/A'))
        
        umap_df["legend_text"] = umap_df.apply(create_legend_text, axis=1)
        
        if interactive_var.get():
            # Plotly interaktiv mit scrollbarer Legende
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                
                unique_cats = []
                if color_column:
                    categories = umap_df[color_column].fillna("N/A").astype(str)
                    unique_cats = sorted(categories.unique())
                    n_colors = len(unique_cats)
                    
                    if n_colors <= 10:
                        colors = px.colors.qualitative.D3[:n_colors]
                    elif n_colors <= 20:
                        colors = px.colors.qualitative.Alphabet[:n_colors]
                    else:
                        colors = px.colors.sample_colorscale("turbo", [i/(n_colors-1) for i in range(n_colors)])
                    
                    color_map = dict(zip(unique_cats, colors))
                    umap_df["color_value"] = categories.map(color_map)
                    umap_df["category"] = categories
                    
                    fig = go.Figure()
                    
                    # Jeden Punkt einzeln mit eigenem Legendeneintrag
                    for idx, row in umap_df.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row["UMAP-1"]],
                            y=[row["UMAP-2"]],
                            mode="markers",
                            marker=dict(size=marker_size, color=row["color_value"]),
                            name=row["legend_text"],
                            legendgroup=row["category"],
                            legendgrouptitle_text=row["category"],
                            hovertemplate=(
                                f"<b>{row.get('doc_id', 'N/A')}</b><br>" +
                                f"{color_column}: {row['category']}<br>" +
                                f"UMAP-1: {row['UMAP-1']:.3f}<br>" +
                                f"UMAP-2: {row['UMAP-2']:.3f}<extra></extra>"
                            ),
                            showlegend=True
                        ))
                    
                    title = f"UMAP Scatterplot ({len(umap_df)} Texte) | Färbung: {color_column}"
                else:
                    fig = go.Figure()
                    
                    for idx, row in umap_df.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row["UMAP-1"]],
                            y=[row["UMAP-2"]],
                            mode="markers",
                            marker=dict(size=marker_size),
                            name=row["legend_text"],
                            hovertemplate=(
                                f"<b>{row.get('doc_id', 'N/A')}</b><br>" +
                                f"UMAP-1: {row['UMAP-1']:.3f}<br>" +
                                f"UMAP-2: {row['UMAP-2']:.3f}<extra></extra>"
                            ),
                            showlegend=True
                        ))
                    
                    title = f"UMAP Scatterplot ({len(umap_df)} Texte)"
                
                # Layout mit scrollbarer Legende
                fig.update_layout(
                    title=title,
                    width=1400,
                    height=900,
                    xaxis_title="UMAP-1",
                    yaxis_title="UMAP-2",
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.02,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="gray",
                        borderwidth=1,
                        font=dict(size=9),
                        itemsizing='constant',
                        tracegroupgap=5,
                        itemwidth=30
                    ),
                    margin=dict(r=350)
                )
                
                result["fig"] = fig
                result["df"] = umap_df
                
                btn_save_html.configure(state="normal")
                btn_save_csv.configure(state="normal")
                
                info_text.insert(tk.END, f"✔ Plot erstellt mit {len(unique_cats) if color_column else 1} Kategorien\n")
                
                # HTML speichern und im Browser öffnen
                import tempfile
                import webbrowser
                
                html_path = tempfile.mktemp(suffix='.html', prefix='umap_legend_')
                fig.write_html(html_path)
                result["html_path"] = html_path
                
                info_text.insert(tk.END, "💡 Öffne Browser...\n")
                webbrowser.open('file://' + html_path)
                
            except ImportError:
                messagebox.showerror("Fehler", "Plotly nicht installiert.", parent=root)
                return
        else:
            # Matplotlib statisch mit Legende
            fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=(18, 10), 
                                                      gridspec_kw={'width_ratios': [3, 1]})
            
            if color_column and color_column in umap_df.columns:
                categories = umap_df[color_column].fillna("N/A").astype(str)
                unique_cats = sorted(categories.unique())
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cats)))
                color_map = dict(zip(unique_cats, colors))
                
                for cat in unique_cats:
                    mask = categories == cat
                    ax_plot.scatter(umap_df.loc[mask, "UMAP-1"], 
                                   umap_df.loc[mask, "UMAP-2"],
                                   c=[color_map[cat]], 
                                   label=cat[:20], 
                                   s=marker_size, 
                                   alpha=0.7)
                
                if len(unique_cats) <= 15:
                    ax_plot.legend(loc="upper right", fontsize=8, title=color_column)
            else:
                ax_plot.scatter(umap_df["UMAP-1"], umap_df["UMAP-2"], 
                               s=marker_size, alpha=0.7)
            
            ax_plot.set_title(f"UMAP ({len(umap_df)} Texte)")
            ax_plot.set_xlabel("UMAP-1")
            ax_plot.set_ylabel("UMAP-2")
            ax_plot.grid(True, alpha=0.3)
            
            # Legende als Text im rechten Panel
            ax_legend.axis('off')
            legend_text_str = "\n".join([
                f"{row['doc_id']}: {row['legend_text'][:50]}" 
                for _, row in umap_df.head(50).iterrows()
            ])
            if len(umap_df) > 50:
                legend_text_str += f"\n... und {len(umap_df) - 50} weitere"
            
            ax_legend.text(0, 1, legend_text_str, transform=ax_legend.transAxes,
                          fontsize=7, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.show()
            
            result["fig"] = fig
            result["df"] = umap_df
            btn_save_png.configure(state="normal")
            btn_save_csv.configure(state="normal")
        
        info_text.insert(tk.END, "✔ Fertig!\n")
    
    def save_html():
        if result["fig"] is None:
            return
        try:
            path = filedialog.asksaveasfilename(
                parent=root, defaultextension=".html", 
                filetypes=[("HTML", "*.html")],
                initialdir=str(EXPLORATION_DIR)
            )
            if path:
                result["fig"].write_html(path)
                messagebox.showinfo("Gespeichert", path, parent=root)
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
    
    def save_csv():
        if result["df"] is None:
            return
        save_dataframe(result["df"], "Streudiagramm_Legende", "umap_legend_data", root)
    
    def save_png():
        if result["fig"] is None:
            return
        save_figure(result["fig"], "Streudiagramm_Legende", "umap_legend_scatter", root)
    
    btn_compute.configure(command=compute)
    btn_save_html.configure(command=save_html)
    btn_save_csv.configure(command=save_csv)
    btn_save_png.configure(command=save_png)




# =============================================================================
# TAB: TOPICVERLÄUFE
# =============================================================================

def build_tab_topics(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """Tab: Topicverläufe mit Glättung, Polynom-Regression und Topic-Auswahl."""
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Topicverläufe")

    row = 0
    info_frame = ttk.LabelFrame(frame, text="ℹ️ Auswahl von Topics für Anzeige ihrer diachronen Entwicklung", padding=6)
    info_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
    ttk.Label(info_frame).pack(anchor="w")

    row += 1
    ttk.Label(frame, text="Document-Topic-Matrix:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    topics_label = ttk.Label(frame, text=str(DATA.path_topics), foreground="blue")
    topics_label.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    def pick_topics():
        p = filedialog.askopenfilename(parent=root, filetypes=[("CSV", "*.csv")])
        if p:
            DATA.path_topics = Path(p)
            DATA._cache.pop("topics", None)
            topics_label.config(text=p)
            load_topics_to_listbox()
    
    ttk.Button(frame, text="...", width=3, command=pick_topics).grid(row=row, column=2)
    
    row += 1
    ttk.Label(frame, text="Metadaten:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    meta_label = ttk.Label(frame, text=str(DATA.path_metadata), foreground="blue")
    meta_label.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    def pick_meta():
        p = filedialog.askopenfilename(parent=root, filetypes=[("CSV", "*.csv")])
        if p:
            DATA.path_metadata = Path(p)
            DATA._cache.pop("metadata", None)
            meta_label.config(text=p)
    
    ttk.Button(frame, text="...", width=3, command=pick_meta).grid(row=row, column=2)
    
    # Parameter
    row += 1
    ttk.Label(frame, text="Schwelle (Cosinus):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_thr = create_entry(frame, width=6)
    ent_thr.insert(0, "0.2")
    ent_thr.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Glättung (MA-Fenster):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_ma = create_entry(frame, width=6)
    ent_ma.insert(0, "3")
    ent_ma.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Polynom-Grad:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_deg = create_entry(frame, width=6)
    ent_deg.insert(0, "3")
    ent_deg.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    # Checkboxen
    row += 1
    abs_var = tk.BooleanVar(value=False)
    smooth_var = tk.BooleanVar(value=True)
    poly_var = tk.BooleanVar(value=True)
    
    ttk.Checkbutton(frame, text="Absolut", variable=abs_var).grid(row=row, column=0, sticky="w", padx=6)
    ttk.Checkbutton(frame, text="Geglättet", variable=smooth_var).grid(row=row, column=1, sticky="w", padx=6)
    ttk.Checkbutton(frame, text="Polynom", variable=poly_var).grid(row=row, column=2, sticky="w", padx=6)
    
    # Topic-Auswahl
    row += 1
    ttk.Label(frame, text="Topics auswählen:").grid(row=row, column=0, sticky="nw", padx=6, pady=4)
    
    listbox_frame = ttk.Frame(frame)
    listbox_frame.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, width=60, height=12, exportselection=False)
    listbox.pack(side="left", fill="both", expand=True)
    
    scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")


    def natural_key(s: str):
        """
        Sortierschlüssel für natürliche Sortierung:
        'Topic_2' < 'Topic_10'
        """
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', s)]

    def load_topics_to_listbox():
        listbox.delete(0, tk.END)
        try:
            df = DATA.load_topics()
            for c in sorted(df.columns.tolist(), key=natural_key):
                listbox.insert(tk.END, c)
        except Exception:
            pass
    
    load_topics_to_listbox()
    
    def compute():
        try:
            df_topics = DATA.load_topics()
            mapping_df = DATA.load_metadata()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        try:
            thr = float(ent_thr.get())
            ma = max(1, int(ent_ma.get()))
            deg = max(1, int(ent_deg.get()))
        except ValueError:
            messagebox.showerror("Fehler", "Parameter prüfen.", parent=root)
            return
        
        # Jahr-Mapping
        mapping_df = mapping_df.copy()
        # Flexible ID-Spalten-Erkennung
        id_col = find_column(mapping_df, ["_id", "id", "doc_id", "document_id"])
        if id_col:
            mapping_df["_id"] = mapping_df[id_col].astype(str)
        else:
            raise ValueError("Keine ID-Spalte gefunden (_id, id, doc_id)")
        
        year_col = "Jahr_final" if "Jahr_final" in mapping_df.columns else ("year_final" if "year_final" in mapping_df.columns else "year")
        mapping_df[year_col] = pd.to_numeric(mapping_df[year_col], errors="coerce")
        jahr_mapping = dict(zip(mapping_df["_id"], mapping_df[year_col]))
        
        df = df_topics.copy()
        df["Jahr"] = df.index.astype(str).map(jahr_mapping)
        df = df.dropna(subset=["Jahr"])
        df["Jahr"] = df["Jahr"].astype(int)
        df = df[df["Jahr"] >= 1840]
        
        # Auswahl
        selected_indices = listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Fehler", "Keine Topics ausgewählt.", parent=root)
            return
        
        selected_topics = [listbox.get(i) for i in selected_indices]
        df_grouped = df.groupby("Jahr").mean().fillna(0.0)
        
        # Plot: Absolut
        if abs_var.get():
            plt.figure(figsize=(14, 8))
            for topic in selected_topics:
                if topic in df_grouped.columns:
                    plt.plot(df_grouped.index, df_grouped[topic], label=topic)
            plt.xlabel("Jahr")
            plt.ylabel("Durchschnittliche Cosinus-Ähnlichkeit")
            plt.title("Absolute Topic-Verläufe (Jahresmittel)")
            plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        # Plot: Geglättet
        if smooth_var.get():
            plt.figure(figsize=(14, 8))
            for topic in selected_topics:
                if topic in df_grouped.columns:
                    values_ma = pd.Series(df_grouped[topic].values).rolling(window=ma, min_periods=1, center=True).mean()
                    plt.plot(df_grouped.index, values_ma, label=topic)
            plt.xlabel("Jahr")
            plt.ylabel("Durchschnittliche Cosinus-Ähnlichkeit")
            plt.title(f"Gleitender Mittelwert (Fenster={ma})")
            plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        # Plot: Anzahl über Schwelle
        relevant_counts = pd.DataFrame(index=sorted(df["Jahr"].unique()))
        for topic in selected_topics:
            if topic in df.columns:
                counts_per_year = df.groupby("Jahr")[topic].apply(lambda x: (x >= thr).sum())
                relevant_counts[topic] = counts_per_year
        
        plt.figure(figsize=(14, 8))
        for topic in selected_topics:
            if topic in relevant_counts.columns:
                plt.plot(relevant_counts.index, relevant_counts[topic], label=topic)
        plt.xlabel("Jahr")
        plt.ylabel(f"Anzahl Texte mit Cosinus ≥ {thr}")
        plt.title(f"Relevante Dokumente pro Jahr (Schwelle {thr})")
        plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Plot: Polynom
        if poly_var.get():
            plt.figure(figsize=(14, 8))
            for topic in selected_topics:
                if topic in df_grouped.columns:
                    years = df_grouped.index.values
                    values = df_grouped[topic].values
                    mask = ~np.isnan(values)
                    years_clean, values_clean = years[mask], values[mask]
                    
                    if len(years_clean) > deg:
                        z = np.polyfit(years_clean, values_clean, deg)
                        p = np.poly1d(z)
                        plt.plot(years_clean, p(years_clean), label=topic)
            
            plt.xlabel("Jahr")
            plt.ylabel("Durchschnittliche Cosinus-Ähnlichkeit")
            plt.title(f"Polynomiale Regression (Grad {deg})")
            plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    row += 1
    ttk.Button(frame, text="Berechnen", command=compute).grid(row=row, column=0, sticky="w", padx=6, pady=6)


# =============================================================================
# TAB: DATEN
# =============================================================================

def build_tab_data(notebook: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="📊 Daten")
    
    row = 0
    ttk.Label(frame, text="Datenverwaltung", font=("TkDefaultFont", 12, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=6)
    
    def add_path(label, path_attr):
        nonlocal row
        row += 1
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        var = tk.StringVar(value=str(getattr(DATA, path_attr)))
        create_entry(frame, width=60, textvariable=var).grid(row=row, column=1, sticky="we", padx=6, pady=3)
        frame.columnconfigure(1, weight=1)
        
        def browse():
            p = filedialog.askopenfilename(parent=root, filetypes=[("CSV", "*.csv")])
            if p:
                var.set(p)
                setattr(DATA, path_attr, Path(p))
        ttk.Button(frame, text="...", width=3, command=browse).grid(row=row, column=2, padx=4)
    
    add_path("Korpus:", "path_corpus")
    add_path("DTM:", "path_dtm")
    add_path("TF-IDF:", "path_tfidf")
    add_path("Metadaten:", "path_metadata")
    add_path("Topics:", "path_topics")
    add_path("Kosinus:", "path_cosine")
    
    row += 1
    ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
    
    row += 1
    ttk.Label(frame, text="Word2Vec:").grid(row=row, column=0, sticky="w", padx=6, pady=3)
    lbl_model = ttk.Label(frame, text=str(MODEL.model_path), foreground="blue")
    lbl_model.grid(row=row, column=1, sticky="w", padx=6, pady=3)
    ttk.Button(frame, text="...", width=3, command=lambda: MODEL.choose_model(root, lbl_model)).grid(row=row, column=2, padx=4)
    
    row += 1
    ttk.Label(frame, text="Termliste:").grid(row=row, column=0, sticky="w", padx=6, pady=3)
    lbl_terms = ttk.Label(frame, text=str(MODEL.termlist_path), foreground="blue")
    lbl_terms.grid(row=row, column=1, sticky="w", padx=6, pady=3)
    ttk.Button(frame, text="...", width=3, command=lambda: MODEL.choose_termlist(root, lbl_terms)).grid(row=row, column=2, padx=4)
    
    row += 1
    info = tk.Text(frame, height=12, width=70, font=("Courier", 9))
    info.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    def load_check():
        DATA.invalidate_cache()
        info.delete(1.0, tk.END)
        
        # WICHTIG: Metadaten ZUERST laden, um Spalten zu registrieren
        try:
            DATA.load_metadata()
            info.insert(tk.END, "✅ Metadaten geladen → Spalten für Mapping registriert\n\n")
        except Exception as e:
            info.insert(tk.END, f"⚠️ Metadaten nicht geladen: {e}\n")
            info.insert(tk.END, "   (DTM/TF-IDF Spaltenanalyse wird eingeschränkt sein)\n\n")
        
        def check(name, loader):
            try:
                df = loader()
                return f"✅ {name}: {df.shape[0]:,} × {df.shape[1]}"
            except Exception as e:
                return f"❌ {name}: {e}"
        
        results = [
            check("Korpus", DATA.load_corpus),
            check("DTM", DATA.load_dtm),
            check("TF-IDF", DATA.load_tfidf),
            check("Topics", DATA.load_topics),
            check("Kosinus", DATA.load_cosine),
        ]
        
        try:
            kv = MODEL.load_model()
            results.append(f"✅ Word2Vec: {len(kv)} Wörter")
        except Exception as e:
            results.append(f"❌ Word2Vec: {e}")
        
        try:
            df = read_csv_auto(MODEL.termlist_path)
            n = sum(df[c].notna().sum() for c in df.columns)
            results.append(f"✅ Termliste: {n} Terme")
        except Exception as e:
            results.append(f"❌ Termliste: {e}")
        
        info.insert(tk.END, "\n".join(results))
        
        # Spaltenanalyse für DTM/TF-IDF (basierend auf metadata.csv Mapping)
        info.insert(tk.END, f"\n\n{'─'*40}\n")
        if METADATA_DETECTOR.is_loaded():
            info.insert(tk.END, f"📋 Metadaten-Spalten aus metadata.csv:\n")
            info.insert(tk.END, f"   {', '.join(METADATA_DETECTOR.get_metadata_column_names()[:8])}...\n\n")
        
        try:
            df_dtm = DATA.load_dtm()
            meta_cols = METADATA_DETECTOR.detect(df_dtm)
            term_cols = METADATA_DETECTOR.get_term_columns(df_dtm)
            info.insert(tk.END, f"DTM-Spaltenanalyse (Mapping mit metadata.csv):\n")
            info.insert(tk.END, f"  • Metadaten: {len(meta_cols)} Spalten\n")
            info.insert(tk.END, f"  • Terme: {len(term_cols)} Spalten\n")
        except Exception:
            pass
        
        ok = sum(1 for r in results if "✅" in r) + (1 if METADATA_DETECTOR.is_loaded() else 0)
        total = len(results) + 1
        info.insert(tk.END, f"\n{'─'*40}\n{ok}/{total} geladen")
    
    def analyze_columns():
        """Detaillierte Spaltenanalyse anzeigen (Mapping mit metadata.csv)."""
        info.delete(1.0, tk.END)
        
        # Prüfe ob Metadaten geladen sind
        if not METADATA_DETECTOR.is_loaded():
            info.insert(tk.END, "⚠️ Metadaten noch nicht geladen!\n\n")
            info.insert(tk.END, "Bitte zuerst 'Laden & Prüfen' klicken,\n")
            info.insert(tk.END, "damit metadata.csv geladen wird.\n\n")
            info.insert(tk.END, "Die Spaltenanalyse verwendet die Spalten\n")
            info.insert(tk.END, "aus metadata.csv um Metadaten von Termen\n")
            info.insert(tk.END, "in DTM/TF-IDF zu unterscheiden.")
            return
        
        try:
            df = DATA.load_dtm()
        except Exception as e:
            info.insert(tk.END, f"❌ DTM nicht geladen: {e}")
            return
        
        info.insert(tk.END, f"Spaltenanalyse DTM ({len(df)} Zeilen, {len(df.columns)} Spalten)\n")
        info.insert(tk.END, f"{'─'*60}\n\n")
        
        info.insert(tk.END, f"📋 MAPPING-QUELLE: metadata.csv\n")
        info.insert(tk.END, f"   Bekannte Metadaten-Spalten: {len(METADATA_DETECTOR.get_metadata_column_names())}\n\n")
        
        meta_cols = METADATA_DETECTOR.detect(df)
        term_cols = METADATA_DETECTOR.get_term_columns(df)
        
        info.insert(tk.END, f"✅ METADATEN in DTM ({len(meta_cols)}):\n")
        info.insert(tk.END, f"   (Spalten die auch in metadata.csv existieren)\n")
        for col in meta_cols:
            info.insert(tk.END, f"   • {col}\n")
        
        info.insert(tk.END, f"\n✅ TERME/AUSDRÜCKE in DTM ({len(term_cols)}):\n")
        info.insert(tk.END, f"   (Numerische Spalten die NICHT in metadata.csv sind)\n")
        for col in term_cols[:15]:
            info.insert(tk.END, f"   • {col}\n")
        if len(term_cols) > 15:
            info.insert(tk.END, f"   ... und {len(term_cols) - 15} weitere\n")
    
    row += 1
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=6)
    
    ttk.Button(btn_frame, text="Laden & Prüfen", command=load_check).pack(side="left", padx=(0, 10))
    ttk.Button(btn_frame, text="🔍 Spaltenanalyse der DTM", command=analyze_columns).pack(side="left")


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def main():
    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("1150x700")
    setup_window(root)
    
    main_nb = ttk.Notebook(root)
    main_nb.pack(fill="both", expand=True, padx=5, pady=5)
    
    build_tab_data(main_nb, root)
    
    nb_expr = ttk.Notebook(main_nb)
    main_nb.add(nb_expr, text="Ausdrücke")
    build_tab_frequency(nb_expr, root)
    build_tab_tfidf_rank(nb_expr, root)
    build_tab_docfreq(nb_expr, root)
    build_tab_concordance(nb_expr, root)
    build_tab_collocations(nb_expr, root)
    build_tab_wordtrends(nb_expr, root)
    
    nb_w2v = ttk.Notebook(main_nb)
    main_nb.add(nb_w2v, text="Wort-Vektor-Modell")
    build_tab_embeddings(nb_w2v, root)
    build_tab_embed_compare(nb_w2v, root)
    build_tab_network(nb_w2v, root)
    
    nb_term = ttk.Notebook(main_nb)
    main_nb.add(nb_term, text="Termset")
    build_tab_cluster(nb_term, root)
    build_tab_wordcloud(nb_term, root)
    build_tab_dendrogram(nb_term, root)
    
    nb_texts = ttk.Notebook(main_nb)
    main_nb.add(nb_texts, text="Texte")
    build_tab_scatter(nb_texts, root)
    build_tab_scatter_legend(nb_texts, root)
    
    nb_topics = ttk.Notebook(main_nb)
    main_nb.add(nb_topics, text="Topics")
    build_tab_topics(nb_topics, root)
    
    root.mainloop()


if __name__ == "__main__":
    main()
