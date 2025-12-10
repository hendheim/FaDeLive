# file: src/tools_visualisations/gui_explore_integrated.py
from __future__ import annotations

import sys
import os
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Iterable
import itertools
from collections import defaultdict, Counter

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec, KeyedVectors
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import umap
import networkx as nx
from adjustText import adjust_text
from matplotlib.text import Text
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Optional: Wortwolke
try:
    from wordcloud import WordCloud
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False
    # Fallback für Clusterfarben
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

# =========================
# Projektpfade & Output
# =========================

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
MODEL_DIR = OUTPUT_DIR / "word2vec_models"
TERMSET_DIR = RESOURCES_DIR / "termsets"

DEFAULT_MODEL_PATH = MODEL_DIR / "korpus_gen.wordvectors"
DEFAULT_TERMLIST_PATH = TERMSET_DIR / "Termset_Begriffe_2.3.csv"

EXPLORATION_DIR = OUTPUT_DIR / "exploration"
NETWORK_OUTPUT_DIR = EXPLORATION_DIR / "networks"
SCATTER_OUTPUT_DIR = EXPLORATION_DIR / "scatterplots"
DENDRO_OUTPUT_DIR = EXPLORATION_DIR / "dendrogramme"

for d in [NETWORK_OUTPUT_DIR, SCATTER_OUTPUT_DIR, DENDRO_OUTPUT_DIR, EXPLORATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# DTM/TF-IDF: Spalten 0–23 = Metadaten (vom Nutzer vorgegeben)
METADATA_COLS: int = 24

# Suite-Name
SUITE_NAME = "Korpus-Explorer-Suite"

# =========================
# Tk-Utils
# =========================

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

# =========================
# Matplotlib-Style
# =========================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "figure.figsize": (15, 10),
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})

# =========================
# Output-Helper
# =========================

def _norm_ctx(s: str) -> str:
    return re.sub(r"[^\w\-.,+]+", "_", s).strip("_") or "export"

def exp_dir(tab_title: str, context: str) -> Path:
    d = EXPLORATION_DIR / tab_title
    d.mkdir(parents=True, exist_ok=True)
    return d / _norm_ctx(context)

def ask_save_df(df: pd.DataFrame, tab_title: str, context: str, parent: tk.Tk) -> None:
    try:
        initdir = (EXPLORATION_DIR / tab_title); initdir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=parent, title="Als CSV speichern",
            initialdir=str(initdir),
            initialfile=f"{_norm_ctx(context)}.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Alle Dateien", "*.*")]
        )
        if not path: return
        df.to_csv(path, index=False)
        messagebox.showinfo("Gespeichert", path, parent=parent)
    except Exception as e:
        messagebox.showerror("Fehler beim Speichern", str(e), parent=parent)

def ask_save_current_figure(tab_title: str, context: str, parent: tk.Tk, fig: Optional[plt.Figure] = None, dpi: int = 300) -> None:
    try:
        initdir = (EXPLORATION_DIR / tab_title); initdir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=parent, title="Als PNG speichern",
            initialdir=str(initdir),
            initialfile=f"{_norm_ctx(context)}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("Alle Dateien", "*.*")]
        )
        if not path: return
        (fig or plt.gcf()).savefig(path, dpi=dpi, bbox_inches="tight")
        messagebox.showinfo("Gespeichert", path, parent=parent)
    except Exception as e:
        messagebox.showerror("Fehler beim Speichern", str(e), parent=parent)

# =========================
# Jahre / Model-Loader
# =========================

def coalesce_years(df: pd.DataFrame, col_year_first="year_first", col_year="year", out="year_final") -> pd.DataFrame:
    to_num = lambda s: pd.to_numeric(s, errors="coerce")
    yf = to_num(df[col_year_first]) if col_year_first in df.columns else pd.Series(index=df.index, dtype="float64")
    y  = to_num(df[col_year]) if col_year in df.columns else pd.Series(index=df.index, dtype="float64")
    df[out] = yf.where(~yf.isna(), y)
    return df

def load_w2v_or_kv(path: Path) -> KeyedVectors:
    """
    Lädt Word2Vec-Modell oder KeyedVectors flexibel.
    Priorität: .wordvectors > .kv > .model > word2vec-format
    """
    if not path.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {path}")
    
    # 1. KeyedVectors direkt (.wordvectors, .kv)
    if path.suffix in {'.wordvectors', '.kv'}:
        try:
            kv = KeyedVectors.load(str(path))
            setattr(kv, "_loaded_path", str(path))
            return kv
        except Exception as e:
            raise RuntimeError(f"KeyedVectors laden fehlgeschlagen: {e}") from e
    
    # 2. Vollständiges Word2Vec-Modell (.model)
    if path.suffix == '.model':
        try:
            m = Word2Vec.load(str(path))
            kv = m.wv
            setattr(kv, "_loaded_path", str(path))
            return kv
        except Exception as e:
            raise RuntimeError(f"Word2Vec-Modell laden fehlgeschlagen: {e}") from e
    
    # 3. Word2Vec-Format (.bin, .txt, .gz)
    try:
        binary = path.suffix.lower() in {".bin", ".gz"}
        kv = KeyedVectors.load_word2vec_format(str(path), binary=binary)
        setattr(kv, "_loaded_path", str(path))
        return kv
    except Exception as exc:
        raise RuntimeError(f"Kein unterstütztes Format: {path.suffix}") from exc

CURRENT_MODEL_PATH: Path = DEFAULT_MODEL_PATH
CURRENT_TERMLIST_PATH: Path = DEFAULT_TERMLIST_PATH
W2V_GLOBAL: Optional[KeyedVectors] = None

def ensure_model_loaded(root: tk.Tk) -> Optional[KeyedVectors]:
    global W2V_GLOBAL
    if W2V_GLOBAL is not None:
        return W2V_GLOBAL
    try:
        W2V_GLOBAL = load_w2v_or_kv(CURRENT_MODEL_PATH)
        return W2V_GLOBAL
    except Exception as e:
        messagebox.showerror("Fehler", f"Modell konnte nicht geladen werden:\n{e}", parent=root)
        return None

def choose_model(root: tk.Tk, label_widget: Optional[tk.Label | ttk.Label] = None) -> None:
    global CURRENT_MODEL_PATH, W2V_GLOBAL
    path_str = filedialog.askopenfilename(
        parent=root, title="Word2Vec-/KeyedVectors-Modell wählen",
        initialdir=str(MODEL_DIR),
        filetypes=[("KeyedVectors", "*.wordvectors *.kv"), ("Gensim Model", "*.model"),
                   ("Word2Vec Bin/Txt", "*.bin *.txt *.gz"), ("Alle Dateien", "*.*")]
    )
    if not path_str: return
    CURRENT_MODEL_PATH = Path(path_str); W2V_GLOBAL = None
    if label_widget is not None: label_widget.config(text=str(CURRENT_MODEL_PATH))

def choose_termset(root: tk.Tk, label_widget: Optional[tk.Label | ttk.Label] = None) -> None:
    global CURRENT_TERMLIST_PATH
    path_str = filedialog.askopenfilename(
        parent=root, title="Termliste wählen",
        initialdir=str(TERMSET_DIR),
        filetypes=[("CSV", "*.csv"), ("Alle Dateien", "*.*")]
    )
    if not path_str: return
    CURRENT_TERMLIST_PATH = Path(path_str)
    if label_widget is not None: label_widget.config(text=str(CURRENT_TERMLIST_PATH))

# =========================
# DataManager (globale Laden)
# =========================

META_NAME_BLACKLIST = {
    "_id", "id", "doc_id", "filename",
    "author", "author_surname", "author_surname_norm",
    "title", "title_norm",
    "source", "journal", "magazine",
    "year", "year_first", "year_final", "Jahr_final",
    "textclass", "address", "address_author",
    "lang", "language"
}

class DataManager:
    def __init__(self) -> None:
        self.path_corpus: Path = PROJECT_ROOT / "output" / "processed_corpus" / "korpus_stop.csv"
        self.path_dtm: Path = PROJECT_ROOT / "output" / "dtm_tfidf_stop" / "dtm_minfreq6.csv"
        self.path_topics: Path = RESOURCES_DIR / "topic-models" / "topics_v3" / "document-topics-distribution_tag.csv"
        self.path_metadata: Path = PROJECT_ROOT / "data" / "raw" / "metadata.csv"
        self.path_tfidf_for_cloud: Path = PROJECT_ROOT / "output" / "dtm_tfidf_stop" / "tfidf-2000.csv"
        self.path_cosine: Path = PROJECT_ROOT / "output" / "cosine" / "cosine_tfidf2000.csv"

        self.corpus_df: Optional[pd.DataFrame] = None
        self.dtm_df: Optional[pd.DataFrame] = None
        self.tokens_per_year_df: Optional[pd.DataFrame] = None
        self.topics_df: Optional[pd.DataFrame] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.tfidf_avg_df: Optional[pd.DataFrame] = None
        self.cosine_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _detect_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
        lower_map = {str(c).lower(): c for c in df.columns}
        for cand in candidates:
            if cand in df.columns: return cand
            lc = str(cand).lower()
            if lc in lower_map: return lower_map[lc]
        return None

    @staticmethod
    def term_columns(df: pd.DataFrame) -> List[str]:
        cols = list(df.columns)
        cand = cols[METADATA_COLS:] if len(cols) > METADATA_COLS else []
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        def is_term(c: str) -> bool:
            cl = str(c).strip()
            return cl not in META_NAME_BLACKLIST
        merged = [c for c in dict.fromkeys(list(cand) + numeric) if is_term(c)]
        return merged

    def load_corpus(self) -> pd.DataFrame:
        if self.corpus_df is not None: return self.corpus_df
        if not self.path_corpus.exists(): raise FileNotFoundError(f"Korpus fehlt: {self.path_corpus}")
        df = pd.read_csv(self.path_corpus, sep=";")
        col_text = self._detect_col(df, ["text", "clean_text", "content", "content_lem", "content_stop", "content_min"])
        if col_text is None: raise ValueError("Keine Textspalte (text/clean_text/content) gefunden.")
        col_id = self._detect_col(df, ["_id", "id", "doc_id", "filename"])
        col_year_first = self._detect_col(df, ["year_first"])
        col_year = self._detect_col(df, ["year", "jahr"])
        col_author = self._detect_col(df, ["author_surname", "author"])
        col_title = self._detect_col(df, ["title"])
        col_source = self._detect_col(df, ["source", "journal", "magazine"])    
        ren = {col_text: "text"}
        if col_id: ren[col_id] = "doc_id"
        if col_year_first: ren[col_year_first] = "year_first"
        if col_year: ren[col_year] = "year"
        if col_author: ren[col_author] = "author_surname"
        if col_title: ren[col_title] = "title"
        if col_source: ren[col_source] = "source"
        df = df.rename(columns=ren)
        if "doc_id" not in df.columns:
            df["doc_id"] = np.arange(1, len(df) + 1)
        df = coalesce_years(df, "year_first", "year", "year_final")
        self.corpus_df = df
        return df

    def load_dtm(self) -> pd.DataFrame:
        if self.dtm_df is not None: return self.dtm_df
        if not self.path_dtm.exists(): raise FileNotFoundError(f"DTM fehlt: {self.path_dtm}")
        df = pd.read_csv(self.path_dtm)
        if "year_first" in df.columns or "year" in df.columns:
            df = coalesce_years(df, "year_first", "year", "year_final")
        self.dtm_df = df
        fname = str(self.path_dtm.name).lower()
        if "tfidf" not in fname and "tf-idf" not in fname:
            self.tokens_per_year_df = self.tokens_per_year_from_dtm(df)
        return df

    def tokens_per_year_from_dtm(self, dtm: pd.DataFrame) -> pd.DataFrame:
        if "year_final" not in dtm.columns and "year" not in dtm.columns and "year_first" not in dtm.columns:
            raise ValueError("DTM hat keine year/year_first Spalten.")
        term_cols = self.term_columns(dtm)
        numeric_term_cols = [c for c in term_cols if pd.api.types.is_numeric_dtype(dtm[c])]
        freq_sum = dtm[numeric_term_cols].sum(axis=1, numeric_only=True)
        year_series = None
        for c in ("year_final", "year_first", "year"):
            if c in dtm.columns:
                year_series = dtm[c]
                break
        tmp = pd.DataFrame({"year": pd.to_numeric(year_series, errors="coerce").astype("Int64"), "tokens": freq_sum})
        out = tmp.groupby("year", dropna=True)["tokens"].sum().reset_index().rename(columns={"tokens": "anzahl_tokens"})
        return out

    def load_topics(self) -> pd.DataFrame:
        if self.topics_df is not None: return self.topics_df
        if not self.path_topics.exists(): raise FileNotFoundError(f"Document-Topic-Matrix fehlt: {self.path_topics}")
        df = pd.read_csv(self.path_topics, index_col=0)
        df.index = df.index.astype(str).str.replace(".txt", "", regex=False)
        def dec(col: str) -> str:
            if str(col).startswith("Topic "):
                try:
                    _, rest = str(col).split(" ", 1)
                    num, *words = rest.split("_")
                    return f"Topic {int(num)-1}_{'_'.join(words)}"
                except Exception:
                    return str(col)
            return str(col)
        df.columns = [dec(c) for c in df.columns]
        self.topics_df = df
        return df

    def load_metadata(self) -> pd.DataFrame:
        if self.metadata_df is not None: return self.metadata_df
        if not self.path_metadata.exists(): raise FileNotFoundError(f"Metadata fehlt: {self.path_metadata}")
        df = pd.read_csv(self.path_metadata, sep=";")
        df["_id"] = df["_id"].astype(str)
        df = coalesce_years(df, "year_first", "year", "Jahr_final")
        self.metadata_df = df
        return df

    def load_tfidf_for_cloud(self) -> pd.DataFrame:
        if self.tfidf_avg_df is not None: return self.tfidf_avg_df
        if not self.path_tfidf_for_cloud.exists(): raise FileNotFoundError(f"TF-IDF-Datei fehlt: {self.path_tfidf_for_cloud}")
        df = pd.read_csv(self.path_tfidf_for_cloud)
        expr = df.iloc[:, METADATA_COLS:]
        expr = expr[[c for c in expr.columns
                     if pd.api.types.is_numeric_dtype(expr[c]) and str(c) not in META_NAME_BLACKLIST]]
        avg = expr.mean(axis=0, numeric_only=True).reset_index()
        avg.columns = ["term", "tfidf_avg"]
        self.tfidf_avg_df = avg
        return avg
    
    def load_cosine(self) -> pd.DataFrame:
        """
        Lädt Kosinus-Matrix und cached sie
        """
        if self.cosine_df is not None:
            return self.cosine_df
        if not self.path_cosine.exists():
            raise FileNotFoundError(f"Kosinus-Matrix fehlt: {self.path_cosine}")
        df = pd.read_csv(self.path_cosine, index_col=0)
        self.cosine_df = df
        return df

DATA = DataManager()

# =========================
# Treeview Sortierung & Eingabefelder
# =========================

def enable_treeview_sort(tree: ttk.Treeview):
    sort_state = {}
    def _sort_by(col: str, descending: bool):
        data = []
        for iid in tree.get_children(""):
            v = tree.set(iid, col)
            try:
                v_key = float(str(v).replace(",", ".")) if v not in ("", None) else float("inf")
            except Exception:
                v_key = v
            data.append((v_key, iid))
        data.sort(reverse=descending)
        for idx, (_val, iid) in enumerate(data):
            tree.move(iid, "", idx)
        sort_state[col] = not descending
    for col in tree["columns"]:
        tree.heading(col, command=lambda c=col: _sort_by(c, sort_state.get(c, False)))

def _mk_entry(parent, **kwargs):
    e = ttk.Entry(parent, **kwargs)
    try:
        e.configure(state="normal", takefocus=True)
    except Exception:
        pass
    return e

# =========================
# Hilfen – Termspalten robust extrahieren
# =========================

def get_term_columns_strict(df: pd.DataFrame) -> List[str]:
    cols = DATA.term_columns(df)
    cols = [c for c in cols if str(c) not in {"year", "year_first", "year_final", "Jahr_final"}]
    return cols

# =========================
# NEU: doc_id aus _id/id/filename sicherstellen
# =========================

def ensure_doc_id_inplace(df: pd.DataFrame) -> None:
    """
    Füllt/erstellt 'doc_id' aus doc_id | _id | id | filename (in dieser Priorität).
    Ergebnis ist string-basiert und niemals leer.
    """
    candidates = ["doc_id", "_id", "id", "filename"]
    src = None
    for c in candidates:
        if c in df.columns and not df[c].isna().all():
            src = c
            break
    if src is None:
        df["doc_id"] = (np.arange(1, len(df) + 1)).astype(str)
        return
    out = df[src].astype(str)
    out = out.replace({"nan": "", "None": "", "NaN": ""})
    mask_empty = (out.str.len() == 0)
    if mask_empty.any():
        filler = (np.arange(1, mask_empty.sum() + 1)).astype(str)
        out.loc[mask_empty] = filler
    df["doc_id"] = out

# =========================
# AUSDRÜCKE – Tabs
# =========================

def build_tab_vocab(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="Frequenz")

    row=0
    ttk.Label(frame, text="Suche (optional, Komma):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_terms = _mk_entry(frame, width=60); ent_terms.grid(row=row, column=1, sticky="we", padx=6, pady=4); frame.columnconfigure(1, weight=1)
    row+=1
    ttk.Label(frame, text="Top-N:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_topn = _mk_entry(frame, width=8); ent_topn.insert(0,"2000"); ent_topn.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row+=1
    tree = ttk.Treeview(frame, columns=("rank","term","freq"), show="headings", height=18)
    for c,w in [("rank",80),("term",260),("freq",120)]:
        tree.heading(c, text=c); tree.column(c, width=w, anchor="w")
    tree.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6); frame.rowconfigure(row, weight=1)
    scroll=ttk.Scrollbar(frame, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=scroll.set); scroll.grid(row=row, column=3, sticky="ns")
    enable_treeview_sort(tree)

    last_df: Optional[pd.DataFrame] = None
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled",
                          command=lambda: ask_save_df(last_df, "Frequenz",
                                                     ("search_"+"_".join([t.strip().lower() for t in ent_terms.get().split(",") if t.strip()]) or f"top{ent_topn.get().strip()}"),
                                                     root) if last_df is not None else None)
    btn_save.grid(row=row+1, column=2, padx=6, pady=6, sticky="e")

    def run():
        nonlocal last_df
        try: dtm=DATA.load_dtm()
        except Exception as e: messagebox.showerror("Fehler", str(e), parent=root); return
        try: N=max(1,int(ent_topn.get().strip()))
        except Exception: messagebox.showerror("Fehler","Top-N ungültig", parent=root); return

        term_cols = get_term_columns_strict(dtm)
        if not term_cols:
            messagebox.showerror("Fehler","Keine Termspalten in DTM erkannt.", parent=root); return

        sums = dtm[term_cols].sum(axis=0, numeric_only=True).sort_values(ascending=False)
        base = pd.DataFrame({"term": sums.index.astype(str), "freq": sums.values})
        base["rank"] = np.arange(1, len(base)+1)

        q = [t.strip().lower() for t in ent_terms.get().split(",") if t.strip()]
        dfshow = base[base["term"].isin(q)].copy() if q else base.head(N).copy()
        dfshow = dfshow[["rank","term","freq"]]

        tree.delete(*tree.get_children())
        for _, r in dfshow.iterrows():
            tree.insert("", "end", values=(int(r["rank"]), r["term"], int(r["freq"])))
        last_df = dfshow; btn_save.configure(state="normal")

    ttk.Button(frame, text="Berechnen", command=run).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")

def build_tab_tfidf_rank(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="TF-IDF-Rang")

    row=0
    ttk.Label(frame, text="Suche (optional, Komma):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_terms = _mk_entry(frame, width=60); ent_terms.grid(row=row, column=1, sticky="we", padx=6, pady=4); frame.columnconfigure(1, weight=1)
    row+=1
    ttk.Label(frame, text="Top-N:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_topn = _mk_entry(frame, width=8); ent_topn.insert(0,"2000"); ent_topn.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row+=1
    tree = ttk.Treeview(frame, columns=("rank","term","tfidf_avg"), show="headings", height=18)
    for c,w in [("rank",80),("term",260),("tfidf_avg",120)]:
        tree.heading(c, text=c); tree.column(c, width=w, anchor="w")
    tree.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6); frame.rowconfigure(row, weight=1)
    scroll=ttk.Scrollbar(frame, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=scroll.set); scroll.grid(row=row, column=3, sticky="ns")
    enable_treeview_sort(tree)

    last_df: Optional[pd.DataFrame] = None
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled",
                          command=lambda: ask_save_df(last_df, "TF-IDF-Rang",
                                                     ("search_"+"_".join([t.strip().lower() for t in ent_terms.get().split(",") if t.strip()]) or f"top{ent_topn.get().strip()}"),
                                                     root) if last_df is not None else None)
    btn_save.grid(row=row+1, column=2, padx=6, pady=6, sticky="e")

    def run():
        nonlocal last_df
        try:
            df = pd.read_csv(DATA.path_tfidf_for_cloud)
            expr = df.iloc[:, METADATA_COLS:]
            keep_cols = [c for c in expr.columns
                         if pd.api.types.is_numeric_dtype(expr[c]) and str(c) not in META_NAME_BLACKLIST]
            expr = expr[keep_cols]
            avg = expr.mean(axis=0, numeric_only=True).sort_values(ascending=False)
            base = pd.DataFrame({"term": avg.index.astype(str), "tfidf_avg": avg.values})
            base["rank"] = np.arange(1, len(base)+1)
        except Exception as e:
            messagebox.showerror("Fehler", f"TF-IDF nicht geladen: {e}", parent=root); return

        try: N=max(1,int(ent_topn.get().strip()))
        except Exception: messagebox.showerror("Fehler","Top-N ungültig", parent=root); return

        q = [t.strip().lower() for t in ent_terms.get().split(",") if t.strip()]
        dfshow = base[base["term"].isin(q)].copy() if q else base.head(N).copy()
        dfshow = dfshow[["rank","term","tfidf_avg"]]

        tree.delete(*tree.get_children())
        for _, r in dfshow.iterrows():
            tree.insert("", "end", values=(int(r["rank"]), r["term"], round(float(r["tfidf_avg"]),6)))
        last_df = dfshow; btn_save.configure(state="normal")

    ttk.Button(frame, text="Berechnen", command=run).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")

def build_tab_docfreq(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="Dokument-Frequenz")

    row=0
    ttk.Label(frame, text="Ausdrücke (kommagetrennt; Regex optional):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_terms=_mk_entry(frame, width=60); ent_terms.grid(row=row, column=1, sticky="we", padx=6, pady=4); frame.columnconfigure(1, weight=1)
    row+=1
    regex_var=tk.BooleanVar(value=False); case_var=tk.BooleanVar(value=False); only_hits_var=tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Regex", variable=regex_var).grid(row=row, column=0, sticky="w", padx=6, pady=2)
    ttk.Checkbutton(frame, text="Groß-/Klein beachten", variable=case_var).grid(row=row, column=1, sticky="w", padx=6, pady=2)
    ttk.Checkbutton(frame, text="Nur Dokumente mit Treffern", variable=only_hits_var).grid(row=row, column=2, sticky="w", padx=6, pady=2)

    row+=1
    ttk.Label(frame, text="Aggregation (bei mehreren Ausdrücken):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    agg_var=tk.StringVar(value="sum")
    ttk.Combobox(frame, textvariable=agg_var, values=["sum","mean","max"], width=10, state="readonly").grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row+=1
    cols=("doc_id","year","author_surname","title","source","terms","value")
    tree = ttk.Treeview(frame, columns=cols, show="headings", height=18)
    widths=[100,60,160,260,180,220,100]
    for c,w in zip(cols,widths):
        tree.heading(c, text=c); tree.column(c, width=w, anchor="w")
    tree.grid(row=row, column=0, columnspan=4, sticky="nsew", padx=6, pady=6); frame.rowconfigure(row, weight=1)
    scroll=ttk.Scrollbar(frame, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=scroll.set); scroll.grid(row=row, column=4, sticky="ns")
    enable_treeview_sort(tree)

    results_df: Optional[pd.DataFrame]=None
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled",
                          command=lambda: ask_save_df(results_df, "Dokument-Frequenz",
                                                     f"{agg_var.get()}_"+"_".join([t.strip() for t in ent_terms.get().split(",") if t.strip()]),
                                                     root) if results_df is not None else None)
    btn_save.grid(row=row+1, column=3, padx=6, pady=6, sticky="e")

    def run_count():
        nonlocal results_df
        try: dfc=DATA.load_corpus()
        except Exception as e: messagebox.showerror("Fehler", f"Korpus nicht geladen: {e}", parent=root); return
        terms=[t.strip() for t in ent_terms.get().split(",") if t.strip()]
        if not terms: messagebox.showerror("Fehler","Bitte Ausdrücke eingeben.", parent=root); return
        flags=0 if case_var.get() else re.IGNORECASE
        patterns=[re.compile(t if regex_var.get() else re.escape(t), flags) for t in terms]

        rows=[]
        for _, r in dfc.iterrows():
            text=str(r["text"]); doc_id=r.get("doc_id","")
            year=r.get("year_final", r.get("year",""))
            author=r.get("author_surname",""); title=r.get("title",""); source=r.get("source","")
            counts=[]
            for pat in patterns:
                counts.append(len(list(pat.finditer(text))))
            if only_hits_var.get() and sum(counts)==0:
                continue
            if agg_var.get()=="sum":
                val=sum(counts)
            elif agg_var.get()=="mean":
                val=float(np.mean(counts))
            else:
                val=max(counts) if counts else 0
            rows.append((str(doc_id), int(year) if pd.notna(year) else None, author, title, source, ",".join(terms), float(val)))
        if not rows:
            messagebox.showinfo("Info","Keine Treffer.", parent=root); return
        results_df=pd.DataFrame(rows, columns=list(cols)).sort_values(["value"], ascending=False)
        tree.delete(*tree.get_children())
        for _, rr in results_df.iterrows():
            tree.insert("", "end", values=(rr["doc_id"], "" if pd.isna(rr["year"]) else int(rr["year"]),
                                           rr["author_surname"], rr["title"], rr["source"], rr["terms"],
                                           round(float(rr["value"]),6)))
        btn_save.configure(state="normal")

    ttk.Button(frame, text="Berechnen", command=run_count).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")

def build_tab_doc_tfidf(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="Dokument-TF-IDF")

    row=0
    ttk.Label(frame, text="Ausdrücke (kommagetrennt; exakte Spaltennamen):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_terms = _mk_entry(frame, width=60); ent_terms.grid(row=row, column=1, sticky="we", padx=6, pady=4); frame.columnconfigure(1, weight=1)
    row+=1
    ttk.Label(frame, text="Aggregation:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    agg_var=tk.StringVar(value="sum")
    ttk.Combobox(frame, textvariable=agg_var, values=["sum","mean","max"], width=10, state="readonly").grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row+=1
    cols=("doc_id","year","author_surname","title","source","terms","value")
    tree = ttk.Treeview(frame, columns=cols, show="headings", height=18)
    widths=[100,60,160,260,180,220,120]
    for c,w in zip(cols,widths):
        tree.heading(c, text=c); tree.column(c, width=w, anchor="w")
    tree.grid(row=row, column=0, columnspan=4, sticky="nsew", padx=6, pady=6); frame.rowconfigure(row, weight=1)
    scroll=ttk.Scrollbar(frame, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=scroll.set); scroll.grid(row=row, column=4, sticky="ns")
    enable_treeview_sort(tree)

    results_df: Optional[pd.DataFrame] = None
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled",
                          command=lambda: ask_save_df(results_df, "Dokument-TF-IDF",
                                                     f"{agg_var.get()}_"+"_".join([t.strip() for t in ent_terms.get().split(",") if t.strip()]) ,
                                                     root) if results_df is not None else None)
    btn_save.grid(row=row+1, column=3, padx=6, pady=6, sticky="e")

    def run_calc():
        nonlocal results_df
        try:
            dtm = pd.read_csv(DATA.path_tfidf_for_cloud)
            if "year_first" in dtm.columns or "year" in dtm.columns:
                dtm = coalesce_years(dtm, "year_first", "year", "year_final")
            # >>> sicherstellen: doc_id korrekt aus _id/id/filename
            ensure_doc_id_inplace(dtm)
        except Exception as e:
            messagebox.showerror("Fehler", f"TF-IDF nicht geladen: {e}", parent=root); return

        terms=[t.strip() for t in ent_terms.get().split(",") if t.strip()]
        if not terms: messagebox.showerror("Fehler","Bitte Ausdrücke eingeben.", parent=root); return

        term_cols_all = get_term_columns_strict(dtm)
        missing=[t for t in terms if t not in term_cols_all]
        if missing:
            messagebox.showwarning("Hinweis", f"Nicht gefunden (Termspalten): {', '.join(missing)}", parent=root)
        terms=[t for t in terms if t in term_cols_all]
        if not terms:
            messagebox.showerror("Fehler","Keine gültigen Terme in TF-IDF.", parent=root); return

        sub = dtm[terms]
        if agg_var.get()=="sum":
            score = sub.sum(axis=1, numeric_only=True)
        elif agg_var.get()=="mean":
            score = sub.mean(axis=1, numeric_only=True)
        else:
            score = sub.max(axis=1, numeric_only=True)

        meta_cols=["doc_id","year_first","year","year_final","author_surname","title","source","_id"]
        for c in meta_cols:
            if c not in dtm.columns: dtm[c] = np.nan
        out = dtm[meta_cols].copy()
        out["value"]=score
        out["terms"]=",".join(terms)
        year = out["year_final"].fillna(out["year"]).astype("Int64")
        out["year"]=year

        out = out.sort_values("value", ascending=False)
        results_df = out[["doc_id","year","author_surname","title","source","terms","value"]].copy()

        tree.delete(*tree.get_children())
        for _, r in results_df.iterrows():
            tree.insert("", "end", values=(r["doc_id"], "" if pd.isna(r["year"]) else int(r["year"]),
                                           r["author_surname"], r["title"], r["source"], r["terms"],
                                           round(float(r["value"]),6)))
        btn_save.configure(state="normal")

    ttk.Button(frame, text="Berechnen", command=run_calc).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")

def build_tab_concordance(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="Konkordanz")
    row=0
    ttk.Label(frame, text="Suchausdruck (Regex erlaubt):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_query = _mk_entry(frame, width=60); ent_query.grid(row=row, column=1, sticky="we", padx=6, pady=4); frame.columnconfigure(1, weight=1)
    row+=1
    ttk.Label(frame, text="Kontext (± Zeichen):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_ctx = _mk_entry(frame, width=8); ent_ctx.insert(0,"50"); ent_ctx.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    row+=1
    case_var = tk.BooleanVar(value=False); whole_word_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Groß-/Klein beachten", variable=case_var).grid(row=row, column=0, sticky="w", padx=6, pady=2)
    ttk.Checkbutton(frame, text="Nur ganze Wörter", variable=whole_word_var).grid(row=row, column=1, sticky="w", padx=6, pady=2)
    row+=1
    ttk.Label(frame, text="Max. Treffer:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_max = _mk_entry(frame, width=8); ent_max.insert(0,"1000"); ent_max.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    row+=1
    tree = ttk.Treeview(frame, columns=("doc_id","year","left","match","right"), show="headings", height=18)
    for c,w in [("doc_id",120),("year",70),("left",300),("match",160),("right",300)]:
        tree.heading(c, text=c); tree.column(c, width=w, anchor="w")
    tree.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6); frame.rowconfigure(row, weight=1)
    scroll=ttk.Scrollbar(frame, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=scroll.set); scroll.grid(row=row, column=3, sticky="ns")
    enable_treeview_sort(tree)

    results_df_conc: Optional[pd.DataFrame] = None
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled",
                          command=lambda: ask_save_df(results_df_conc, "Konkordanz",
                                                     f"{ent_query.get().strip()}_ctx{ent_ctx.get().strip()}",
                                                     root) if results_df_conc is not None else None)
    btn_save.grid(row=row+1, column=2, padx=6, pady=6, sticky="e")

    def run_conc():
        nonlocal results_df_conc
        try: df = DATA.load_corpus()
        except Exception as e: messagebox.showerror("Fehler", f"Korpus nicht geladen: {e}", parent=root); return
        query = ent_query.get().strip()
        if not query: messagebox.showerror("Fehler","Bitte Suchausdruck eingeben.",parent=root); return
        try:
            ctx = max(1,int(ent_ctx.get().strip())); maxhits = max(1,int(ent_max.get().strip()))
        except Exception:
            messagebox.showerror("Fehler","Kontext/Max. Treffer ungültig.",parent=root); return
        flags = 0 if case_var.get() else re.IGNORECASE
        # Wenn "Nur ganze Wörter" aktiviert: füge \b hinzu
        if whole_word_var.get():
            query_pattern = r"\b" + re.escape(query) + r"\b"
        else:
            query_pattern = re.escape(query)
        pattern = re.compile(query_pattern, flags)
        tree.delete(*tree.get_children()); results=[]; count=0
        for _, r in df.iterrows():
            text=str(r["text"]); doc_id=r.get("doc_id",""); year=r.get("year_final", r.get("year",""))
            for m in pattern.finditer(text):
                s,e=m.start(), m.end()
                left=text[max(0,s-ctx):s]; match=text[s:e]; right=text[e:e+ctx]
                tup=(str(doc_id), int(year) if pd.notna(year) else None, left, match, right)
                results.append(tup)
                tree.insert("", "end", values=(tup[0], "" if tup[1] is None else tup[1], tup[2], tup[3], tup[4]))
                count+=1
                if count>=maxhits: break
            if count>=maxhits: break
        results_df_conc = pd.DataFrame(results,columns=["doc_id","year","left","match","right"])
        btn_save.configure(state="normal")

    ttk.Button(frame, text="Suchen", command=run_conc).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")

def build_tab_collocations(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="Kollokation")

    row = 0
    ttk.Label(frame, text="Zielausdrücke (kommagetrennt):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_targets = _mk_entry(frame, width=60); ent_targets.grid(row=row, column=1, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)

    row += 1
    ttk.Label(frame, text="Fensterweite (± Wörter):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_window = _mk_entry(frame, width=8); ent_window.insert(0, "5"); ent_window.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row += 1
    ttk.Label(frame, text="Top-N:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_topn = _mk_entry(frame, width=8); ent_topn.insert(0, "100"); ent_topn.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row += 1
    ttk.Label(frame, text="N-Gram (1=Token, 2=Bigram, 3=Trigram):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_ng = _mk_entry(frame, width=8); ent_ng.insert(0, "1"); ent_ng.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row += 1
    ttk.Label(frame, text="Mindestfrequenz:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_minf = _mk_entry(frame, width=8); ent_minf.insert(0, "3"); ent_minf.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row += 1
    metric_var = tk.StringVar(value="FREQ")
    ttk.Label(frame, text="Metrik:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ttk.Combobox(frame, textvariable=metric_var, values=["FREQ", "PMI"], width=10, state="readonly").grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row += 1
    ttk.Label(frame, text="Kollokationsliste:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    tree = ttk.Treeview(frame, columns=("target", "collocate", "freq", "score"), show="headings", height=12)
    for c, w in [("target", 160), ("collocate", 260), ("freq", 100), ("score", 100)]:
        tree.heading(c, text=c); tree.column(c, width=w, anchor="w")
    tree.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=scroll.set)
    scroll.grid(row=row, column=3, sticky="ns")
    enable_treeview_sort(tree)

    row += 1
    ttk.Label(frame, text="Dokumente (Klick auf Kollokation):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    cols_doc = ("doc_id", "year", "author_surname", "title", "freq")
    tree_doc = ttk.Treeview(frame, columns=cols_doc, show="headings", height=10)
    for c, w in zip(cols_doc, [100, 60, 220, 360, 80]):
        tree_doc.heading(c, text=c); tree_doc.column(c, width=w, anchor="w")
    tree_doc.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    scroll2 = ttk.Scrollbar(frame, orient="vertical", command=tree_doc.yview); tree_doc.configure(yscrollcommand=scroll2.set)
    scroll2.grid(row=row, column=3, sticky="ns")
    enable_treeview_sort(tree_doc)

    row += 1
    results_df: Optional[pd.DataFrame] = None
    docs_df: Optional[pd.DataFrame] = None

    btn_save_list = ttk.Button(
        frame, text="Kollokationsliste speichern",
        state="disabled",
        command=lambda: ask_save_df(
            results_df, "Kollokation",
            f"{ent_targets.get().strip()}_W{ent_window.get().strip()}_N{ent_ng.get().strip()}_top{ent_topn.get().strip()}",
            root
        ) if results_df is not None else None
    )
    btn_save_list.grid(row=row, column=1, padx=6, pady=6, sticky="w")

    btn_save_docs = ttk.Button(
        frame, text="Dokumentliste speichern",
        state="disabled",
        command=lambda: ask_save_df(
            docs_df, "Kollokation_Dokumente",
            f"{ent_targets.get().strip()}_W{ent_window.get().strip()}_N{ent_ng.get().strip()}",
            root
        ) if docs_df is not None else None
    )
    btn_save_docs.grid(row=row, column=2, padx=6, pady=6, sticky="e")

    docs_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    tok_re = re.compile(r"\w+|\S", flags=re.UNICODE)

    def tokenize_words(text: object) -> List[str]:
        if not isinstance(text, str) or not text:
            return []
        return [t for t in tok_re.findall(text.lower()) if t.strip()]

    from collections import deque
    def iter_ngrams(tokens: List[str], n: int):
        if n <= 1:
            for t in tokens:
                yield t
            return
        buf = deque(maxlen=n)
        for t in tokens:
            buf.append(t)
            if len(buf) == n:
                # <<< Fix: mit Leerzeichen verbinden (statt "g")
                yield " ".join(buf)

    def coalesce_year(row: pd.Series) -> str:
        for c in ("year_final", "year_first", "year"):
            if c in row and pd.notna(row[c]):
                try:
                    return str(int(float(row[c])))
                except Exception:
                    continue
        return ""

    def compute():
        nonlocal results_df
        try:
            df = DATA.load_corpus()
        except Exception as e:
            messagebox.showerror("Fehler", f"Korpus nicht geladen: {e}", parent=root); return

        targets = [t.strip().lower() for t in ent_targets.get().split(",") if t.strip()]
        if not targets:
            messagebox.showerror("Fehler", "Bitte Zielausdrücke eingeben.", parent=root); return

        try:
            W    = max(1, int(ent_window.get().strip()))
            topn = max(1, int(ent_topn.get().strip()))
            minf = max(1, int(ent_minf.get().strip()))
            ng   = max(1, int(ent_ng.get().strip()))
        except Exception:
            messagebox.showerror("Fehler", "Fenster/TopN/MinFreq/N-Gram prüfen.", parent=root); return

        total_tokens = 0
        freq_w  = Counter()
        freq_tw = defaultdict(Counter)
        freq_t  = Counter()

        tree.delete(*tree.get_children())
        tree_doc.delete(*tree_doc.get_children())
        btn_save_list.configure(state="disabled")
        btn_save_docs.configure(state="disabled")
        docs_cache.clear()

        try:
            for _, r in df.iterrows():
                text = r.get("text", "")
                tokens = tokenize_words(text)
                if not tokens:
                    continue

                ngram_stream = list(iter_ngrams(tokens, ng))
                total_tokens += len(ngram_stream)
                for w in ngram_stream:
                    freq_w[w] += 1

                for i, w in enumerate(tokens):
                    if w in targets:
                        freq_t[w] += 1
                        L = max(0, i - W)
                        R = min(len(tokens), i + W + 1)
                        if ng == 1:
                            ctx = tokens[L:i] + tokens[i+1:R]
                        else:
                            ctx = []
                            left_tokens  = tokens[L:i]
                            right_tokens = tokens[i+1:R]
                            for g in iter_ngrams(left_tokens, ng):
                                ctx.append(g)
                            for g in iter_ngrams(right_tokens, ng):
                                ctx.append(g)
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
                        pw  = freq_w[cw] / max(1, total_tokens)
                        pt  = ct / max(1, total_tokens)
                        ptw = ctw / max(1, total_tokens)
                        denom = max(eps, pt * pw)
                        numer = max(eps, ptw)
                        score = math.log2(numer / denom)
                    rows.append((t, cw, int(ctw), float(score)))

            if not rows:
                messagebox.showinfo("Kollokation", "Keine Ergebnisse (Parameter/Frequenzen prüfen).", parent=root); return

            res = pd.DataFrame(rows, columns=["target", "collocate", "freq", "score"])
            if metric_var.get() == "FREQ":
                res = (res.sort_values(["target", "freq"], ascending=[True, False])
                         .groupby("target", as_index=False, group_keys=False)
                         .head(topn))
            else:
                res = (res.sort_values(["target", "score"], ascending=[True, False])
                         .groupby("target", as_index=False, group_keys=False)
                         .head(topn))

            for _, r0 in res.iterrows():
                tree.insert("", "end", values=(r0["target"], r0["collocate"], int(r0["freq"]), round(float(r0["score"]), 4)))

            results_df = res
            btn_save_list.configure(state="normal")

        except MemoryError:
            messagebox.showerror("Fehler", "Speicher erschöpft bei Kollokationsberechnung. Reduziere Top-N/Fensterweite oder N-Gram.", parent=root)
        except Exception as e:
            messagebox.showerror("Fehler (Kollokation)", str(e), parent=root)

    def on_select(_event):
        nonlocal docs_df
        sel = tree.selection()
        if not sel:
            return
        vals = tree.item(sel[0], "values")
        if not vals or len(vals) < 2:
            return
        t, cw = str(vals[0]), str(vals[1])

        key = (t, cw)
        df_cached = docs_cache.get(key)
        if df_cached is not None:
            docs_df = df_cached
            tree_doc.delete(*tree_doc.get_children())
            for _, r in docs_df.iterrows():
                y = r.get("year")
                y_out = "" if (pd.isna(y) or y == "") else int(y)
                tree_doc.insert("", "end", values=(r.get("doc_id",""), y_out,
                                                   r.get("author_surname",""), r.get("title",""), int(r.get("freq",0))))
            btn_save_docs.configure(state="normal")
            return

        try:
            df = DATA.load_corpus()
        except Exception as e:
            messagebox.showerror("Fehler", f"Korpus nicht geladen: {e}", parent=root); return

        try:
            W  = max(1, int(ent_window.get().strip()))
            ng = max(1, int(ent_ng.get().strip()))
        except Exception:
            messagebox.showerror("Fehler", "Fenster/N-Gram prüfen.", parent=root); return

        rows_doc = []
        try:
            for _, d in df.iterrows():
                text = d.get("text", "")
                tokens = tokenize_words(text)
                if not tokens:
                    continue

                positions = [i for i, w in enumerate(tokens) if w == t]
                if not positions:
                    continue

                count_cw = 0
                for i in positions:
                    L = max(0, i - W)
                    R = min(len(tokens), i + W + 1)
                    if ng == 1:
                        ctx = tokens[L:i] + tokens[i+1:R]
                        count_cw += sum(1 for token in ctx if token == cw)
                    else:
                        left_tokens  = tokens[L:i]
                        right_tokens = tokens[i+1:R]
                        for g in iter_ngrams(left_tokens, ng):
                            if g == cw:
                                count_cw += 1
                        for g in iter_ngrams(right_tokens, ng):
                            if g == cw:
                                count_cw += 1

                if count_cw > 0:
                    rows_doc.append({
                        "doc_id": d.get("doc_id", d.get("_id","")),
                        "year":   coalesce_year(d),
                        "author_surname": d.get("author_surname", ""),
                        "title":  d.get("title", ""),
                        "freq":   int(count_cw),
                    })

            docs_df = (pd.DataFrame(rows_doc)
                         .assign(year=lambda x: pd.to_numeric(x["year"], errors="coerce").astype("Int64"))
                         .sort_values(["freq", "year"], ascending=[False, True])
                         .fillna({"year": ""}))

            docs_cache[key] = docs_df

            tree_doc.delete(*tree_doc.get_children())
            for _, r in docs_df.iterrows():
                y = r.get("year")
                y_out = "" if (pd.isna(y) or y == "") else int(y)
                tree_doc.insert("", "end", values=(r.get("doc_id",""), y_out,
                                                   r.get("author_surname",""), r.get("title",""), int(r.get("freq",0))))
            btn_save_docs.configure(state="normal")

        except MemoryError:
            messagebox.showerror("Fehler", "Speicher erschöpft bei Dokumentauflistung. Bitte andere Kollokation wählen.", parent=root)
        except Exception as e:
            messagebox.showerror("Fehler (Dokumentliste)", str(e), parent=root)

    tree.bind("<<TreeviewSelect>>", on_select)

    ttk.Button(frame, text="Berechnen", command=compute).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")

def build_tab_wordtrends(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="Wortverläufe")

    row=0
    ttk.Label(frame, text="Begriffe (kommagetrennt; exakte Spaltennamen):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_terms=_mk_entry(frame, width=60); ent_terms.grid(row=row, column=1, sticky="we", padx=6, pady=4); frame.columnconfigure(1, weight=1)

    row+=1
    ttk.Label(frame, text="Glättung Fenster (roll. MW):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_win=_mk_entry(frame, width=8); ent_win.insert(0,"5"); ent_win.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row+=1
    ttk.Label(frame, text="Poly-Grad (Regression):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_deg=_mk_entry(frame, width=8); ent_deg.insert(0,"6"); ent_deg.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row+=1
    absolute_var = tk.BooleanVar(value=False)
    smooth_var   = tk.BooleanVar(value=True)
    poly_var     = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Absolut",  variable=absolute_var).grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frame, text="Geglättet", variable=smooth_var).grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frame, text="Polynom",   variable=poly_var).grid(row=row, column=2, sticky="w", padx=6, pady=4)

    row+=1
    ttk.Label(frame, text="Jahrbereich (leer = automatisch), z. B. 1782-1891:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_range=_mk_entry(frame, width=20); ent_range.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    last_trends_abs: Optional[pd.DataFrame] = None
    last_trends_rel: Optional[pd.DataFrame] = None
    last_plot_fig: Optional[plt.Figure] = None
    last_plot_ctx: str = ""

    row+=1
    btn_save_csv = ttk.Button(
        frame, text="Daten (CSV) speichern", state="disabled",
        command=lambda: ask_save_df((last_trends_rel if last_trends_rel is not None else last_trends_abs),
                                    "Wortverläufe", last_plot_ctx, root)
    )
    btn_save_png = ttk.Button(
        frame, text="Plot (PNG) speichern", state="disabled",
        command=lambda: ask_save_current_figure("Wortverläufe", last_plot_ctx, root, fig=last_plot_fig)
    )
    btn_save_csv.grid(row=row, column=0, padx=6, pady=6, sticky="w")
    btn_save_png.grid(row=row, column=1, padx=6, pady=6, sticky="w")

    def run():
        nonlocal last_trends_abs, last_trends_rel, last_plot_fig, last_plot_ctx
        # 1) DTM laden – ohne Sonderprüfung: jede im „Daten“-Tab gesetzte Datei wird akzeptiert
        try:
            dtm = DATA.load_dtm()
        except Exception as e:
            messagebox.showerror("Fehler", f"DTM nicht geladen: {e}", parent=root); return

        # 2) Begriffe prüfen
        terms_input = [t.strip() for t in ent_terms.get().split(",") if t.strip()]
        if not terms_input:
            messagebox.showerror("Fehler","Bitte Begriffe eingeben.", parent=root); return

        term_cols_all = set(get_term_columns_strict(dtm))
        missing = [t for t in terms_input if t not in term_cols_all]
        if missing:
            messagebox.showwarning("Hinweis", f"Nicht gefunden (Termspalten): {', '.join(missing)}", parent=root)
        terms = [t for t in terms_input if t in term_cols_all]
        if not terms:
            messagebox.showerror("Fehler","Kein gültiger Begriff in der DTM gefunden.", parent=root); return

        # 3) Parameter
        try:
            win = max(1, int(ent_win.get().strip()))
            deg = max(1, int(ent_deg.get().strip()))
        except Exception:
            messagebox.showerror("Fehler","Glättungsfenster/Polygrad ungültig.", parent=root); return

        # 4) Jahre ermitteln (robust: year_final, sonst year_first/year)
        if "year_final" in dtm.columns:
            years_series = pd.to_numeric(dtm["year_final"], errors="coerce")
        elif "year_first" in dtm.columns or "year" in dtm.columns:
            tmp = dtm.copy()
            tmp = coalesce_years(tmp, "year_first", "year", "year_final")
            years_series = pd.to_numeric(tmp["year_final"], errors="coerce")
        else:
            messagebox.showerror("Fehler", "Keine Jahrspalten in der DTM vorhanden.", parent=root); return

        def parse_year_range(s: pd.Series, raw: str) -> Tuple[int,int]:
            s = s.dropna().astype(int)
            if s.empty: return (1800, 1900)
            if raw and "-" in raw:
                a,b = raw.split("-",1)
                return (int(a.strip()), int(b.strip()))
            return (int(s.min()), int(s.max()))
        y_min, y_max = parse_year_range(years_series, ent_range.get().strip())
        all_years = pd.DataFrame({"year": np.arange(y_min, y_max+1, dtype=int)})

        # 5) Tokenmenge pro Jahr aus der DTM (Summe numerischer Termspalten)
        try:
            if DATA.tokens_per_year_df is None:
                DATA.tokens_per_year_df = DATA.tokens_per_year_from_dtm(dtm)
            tpy = DATA.tokens_per_year_df.set_index("year")["anzahl_tokens"]
        except Exception as e:
            messagebox.showerror("Fehler", f"Tokenmengen pro Jahr konnten nicht aus der DTM ermittelt werden:\n{e}", parent=root); return

        # 6) Reihen bauen (absolut & relativ pro 10.000 Tokens)
        series_rel: Dict[str,pd.DataFrame] = {}
        series_abs: Dict[str,pd.DataFrame] = {}

        for term in terms:
            vals = pd.to_numeric(dtm[term], errors="coerce").fillna(0)
            df_term = pd.DataFrame({
                "year": pd.to_numeric(years_series, errors="coerce").astype("Int64"),
                "value": vals
            }).dropna(subset=["year"])
            df_term = df_term.groupby("year").sum(numeric_only=True).reset_index()
            df_term = all_years.merge(df_term, on="year", how="left").fillna(0)
            df_term["anzahl_tokens"] = df_term["year"].map(tpy).fillna(0)

            # relative Normierung nur dort, wo eine Tokenmenge > 0 vorliegt
            df_term["rel"] = 0.0
            mask = df_term["anzahl_tokens"] > 0
            df_term.loc[mask, "rel"] = (df_term.loc[mask, "value"] / df_term.loc[mask, "anzahl_tokens"]) * 10000.0

            series_abs[term] = df_term[["year","value"]].copy()
            series_rel[term] = df_term[["year","rel"]].copy()

        last_trends_abs = pd.concat([d.assign(term=t).rename(columns={"value":"y"}) for t,d in series_abs.items()], ignore_index=True)
        last_trends_rel = pd.concat([d.assign(term=t).rename(columns={"rel":"y"}) for t,d in series_rel.items()], ignore_index=True)

        # 7) Plots (mit Optionen)
        # Absolut
        if absolute_var.get():
            fig = plt.figure(figsize=(12,6))
            for term, dfa in series_abs.items():
                plt.plot(dfa["year"], dfa["value"], label=term)
            plt.title("Rohfrequenzen der Ausdrücke (pro Jahr, DTM-basiert)")
            plt.xlabel("Jahr"); plt.ylabel("Frequenz")
            plt.grid(True); plt.tight_layout(); plt.legend()
            last_plot_fig = fig; last_plot_ctx = f"{'_'.join(terms)}_absolut"
            btn_save_csv.configure(state="normal"); btn_save_png.configure(state="normal")
            plt.show()

        # Geglättet (relativ)
        if smooth_var.get():
            fig = plt.figure(figsize=(12,6))
            for term, dfr in series_rel.items():
                sm = dfr["rel"].rolling(window=win, center=True, min_periods=1).mean()
                plt.plot(dfr["year"], sm, label=f"{term} (MW)")
            plt.title("Geglättete relative Frequenz (pro 10.000, DTM-basiert)")
            plt.xlabel("Jahr"); plt.ylabel("Rel. Frequenz pro 10.000")
            plt.grid(True); plt.tight_layout(); plt.legend()
            last_plot_fig = fig; last_plot_ctx = f"{'_'.join(terms)}_smooth{win}"
            btn_save_csv.configure(state="normal"); btn_save_png.configure(state="normal")
            plt.show()

        # Polynom (relativ)
        if poly_var.get():
            fig = plt.figure(figsize=(12,6))
            for term, dfr in series_rel.items():
                y = dfr["rel"].values
                x = dfr["year"].astype(float).values
                if np.unique(y).size <= 1:
                    continue
                degree = min(deg, max(1, len(x) - 1))
                coeffs = np.polyfit(x, y, degree)
                xx = np.linspace(x.min(), x.max(), 200)
                yy = np.polyval(coeffs, xx)
                plt.plot(xx, yy, label=term)
            plt.title(f"Polynomiale Regression (Grad {deg}) – relative Frequenz (DTM-basiert)")
            plt.xlabel("Jahr"); plt.ylabel("Rel. Frequenz pro 10.000")
            plt.grid(True); plt.tight_layout(); plt.legend()
            last_plot_fig = fig; last_plot_ctx = f"{'_'.join(terms)}_poly{deg}"
            btn_save_csv.configure(state="normal"); btn_save_png.configure(state="normal")
            plt.show()

    ttk.Button(frame, text="Plotten", command=run).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")


# =========================
# Wort-Vektor-Modell – Tabs
# =========================

def build_tab_embeddings(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="Embeddings")

    row=0
    ttk.Label(frame, text="Modell:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    model_label = ttk.Label(frame, text=str(CURRENT_MODEL_PATH)); model_label.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Button(frame, text="Öffnen …", command=lambda: choose_model(root, model_label)).grid(row=row, column=2, sticky="w", padx=6, pady=4)

    row+=1
    ttk.Label(frame, text="Wörter (kommagetrennt):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_words = _mk_entry(frame, width=60); ent_words.grid(row=row, column=1, columnspan=2, sticky="we", padx=6, pady=4); frame.columnconfigure(1, weight=1)

    row+=1
    ttk.Label(frame, text="Top-N:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_topn = _mk_entry(frame, width=8); ent_topn.insert(0,"20"); ent_topn.grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row+=1
    tree = ttk.Treeview(frame, columns=("query","rank","word","sim"), show="headings", height=18)
    for c,w in [("query",160),("rank",60),("word",220),("sim",100)]:
        tree.heading(c, text=c); tree.column(c, width=w, anchor="w")
    tree.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6); frame.rowconfigure(row, weight=1)
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=scroll.set); scroll.grid(row=row, column=3, sticky="ns")
    enable_treeview_sort(tree)

    last_df: Optional[pd.DataFrame] = None
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled",
                          command=lambda: ask_save_df(last_df, "Embeddings", "_".join([w.strip() for w in ent_words.get().split(",") if w.strip()]) or "queries", root) if last_df is not None else None)
    btn_save.grid(row=row+1, column=2, padx=6, pady=6, sticky="e")

    def run():
        nonlocal last_df
        kv = ensure_model_loaded(root)
        if kv is None: return
        try: n = max(1,int(ent_topn.get().strip()))
        except Exception: messagebox.showerror("Fehler","Top-N ungültig",parent=root); return
        queries = [w.strip() for w in ent_words.get().split(",") if w.strip()]
        if not queries: messagebox.showerror("Fehler","Begriffe eingeben.",parent=root); return
        rows=[]
        for q in queries:
            if q not in kv.key_to_index:
                continue
            res=kv.most_similar(q, topn=n)
            for i,(w,s) in enumerate(res, start=1):
                rows.append((q,i,w,round(float(s),4)))
        df = pd.DataFrame(rows, columns=["query","rank","word","sim"])
        tree.delete(*tree.get_children())
        for _, r in df.iterrows():
            tree.insert("", "end", values=(r["query"], r["rank"], r["word"], r["sim"]))
        last_df = df; btn_save.configure(state="normal")

    ttk.Button(frame, text="Analysieren", command=run).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")

ÄHNLICHKEITS_SCHWELLE = 0.3
TOP_N_NEIGHBORS = 50
def berechne_ähnlichkeit_central(kv: KeyedVectors, a: str, vergleiche: List[str], top_n: int = TOP_N_NEIGHBORS
) -> Tuple[Dict[str, Tuple[object, List[Tuple[str, float, float]]]], Optional[str]]:
    a = a.strip(); vergleiche = [w.strip() for w in vergleiche if w.strip()]
    if a == "" or not vergleiche: return {}, "Zentraler Ausdruck und Vergleichswörter dürfen nicht leer sein."
    if a not in kv.key_to_index: return {}, f"Zentraler Ausdruck '{a}' nicht im Vokabular."
    a_vec = kv[a]; ergebnisse: Dict[str, Tuple[object, List[Tuple[str, float, float]]]] = {}
    nachbarn_a = dict(kv.most_similar(a, topn=top_n))
    for wort in vergleiche:
        if wort not in kv.key_to_index:
            ergebnisse[wort] = ("Nicht im Vokabular", []); continue
        score = float(1 - cosine(a_vec, kv[wort])); wort_nachbarn = dict(kv.most_similar(wort, topn=top_n))
        gemeinsame=[]
        for gw in set(nachbarn_a.keys()).intersection(wort_nachbarn.keys()):
            sim_a=float(nachbarn_a[gw]); sim_b=float(wort_nachbarn[gw])
            if sim_a>=ÄHNLICHKEITS_SCHWELLE and sim_b>=ÄHNLICHKEITS_SCHWELLE:
                gemeinsame.append((gw, round(sim_a,4), round(sim_b,4)))
        gemeinsame_sorted = sorted(gemeinsame, key=lambda x: (-x[1]-x[2], x[0]))
        ergebnisse[wort]=(round(score,4), gemeinsame_sorted)
    return ergebnisse, None

def build_tab_embed_compare(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="Embeddings Vergleich")
    row=0
    ttk.Label(frame, text="Zentraler Ausdruck:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_central = _mk_entry(frame, width=40); ent_central.grid(row=row, column=1, padx=6, pady=4)
    row+=1
    ttk.Label(frame, text="Vergleichsausdrücke (kommagetrennt):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_comps = _mk_entry(frame, width=60); ent_comps.grid(row=row, column=1, columnspan=2, padx=6, pady=4, sticky="we"); frame.columnconfigure(1, weight=1)
    row+=1
    tree = ttk.Treeview(frame, columns=("vergleich","score","gemeinsame"), show="headings", height=18)
    for c,w in [("vergleich",220),("score",100),("gemeinsame",600)]:
        tree.heading(c, text=c); tree.column(c, width=w, anchor="w")
    tree.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6); frame.rowconfigure(row, weight=1)
    scroll=ttk.Scrollbar(frame, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=scroll.set); scroll.grid(row=row, column=3, sticky="ns")
    enable_treeview_sort(tree)

    last_df: Optional[pd.DataFrame]=None
    btn_save = ttk.Button(frame, text="CSV speichern", state="disabled",
                          command=lambda: ask_save_df(last_df, "Embeddings Vergleich", ent_central.get().strip() or "central", root) if last_df is not None else None)
    btn_save.grid(row=row+1, column=2, padx=6, pady=6, sticky="e")

    def run():
        nonlocal last_df
        kv = ensure_model_loaded(root)
        if kv is None: return
        central = ent_central.get().strip()
        comps = [w.strip() for w in ent_comps.get().split(",") if w.strip()]
        if not central or not comps:
            messagebox.showerror("Fehler","Bitte Zentral- und Vergleichsausdrücke eingeben.", parent=root); return
        ergebnisse, fehler = berechne_ähnlichkeit_central(kv, central, comps, top_n=TOP_N_NEIGHBORS)
        if fehler:
            messagebox.showerror("Fehler", fehler, parent=root); return
        rows=[]
        for wort, (score, gemeinsame) in ergebnisse.items():
            if isinstance(score, str):
                rows.append((wort, score, "—"))
            else:
                joined = "; ".join([f"{gw} ({sa:.3f}/{sb:.3f})" for gw,sa,sb in gemeinsame]) if gemeinsame else "(keine)"
                rows.append((wort, float(score), joined))
        df=pd.DataFrame(rows, columns=["vergleich","score","gemeinsame"])
        tree.delete(*tree.get_children())
        for _, r in df.iterrows():
            tree.insert("", "end", values=(r["vergleich"], r["score"], r["gemeinsame"]))
        last_df=df; btn_save.configure(state="normal")

    ttk.Button(frame, text="Vergleichen", command=run).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")

def build_graph(kv: KeyedVectors, keywords: List[str], topn: int, threshold: float) -> Tuple[nx.Graph, List[float]]:
    exclude_tokens = {".", ",", "!", "?", "...", ";", ":", "„", "“", '"', "'", "–", "—", "-", "(", ")", "[", "]", "{", "}"}
    G = nx.Graph(); similarities: List[float] = []
    similar_words_map: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for word in keywords:
        if word in kv.key_to_index and word not in exclude_tokens:
            G.add_node(word)
    for word in keywords:
        if word not in kv.key_to_index:
            continue
        for neighbor, sim in kv.most_similar(word, topn=topn):
            if sim >= threshold and neighbor not in exclude_tokens:
                similar_words_map[neighbor].append((word, float(sim)))
    for neighbor, connections in similar_words_map.items():
        for target, sim in connections:
            if sim >= threshold:
                G.add_node(neighbor)
                G.add_edge(target, neighbor, weight=sim)
                similarities.append(sim)
    for w1, w2 in itertools.combinations(keywords, 2):
        if w1 in kv.key_to_index and w2 in kv.key_to_index and w1 not in exclude_tokens and w2 not in exclude_tokens:
            sim = float(kv.similarity(w1, w2))
            if sim >= threshold:
                G.add_edge(w1, w2, weight=sim)
                similarities.append(sim)
    return G, similarities

def build_tab_network(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="Netzwerk")
    row=0
    ttk.Label(frame, text="Modell:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    model_label = ttk.Label(frame, text=str(CURRENT_MODEL_PATH)); model_label.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Button(frame, text="Öffnen …", command=lambda: choose_model(root, model_label)).grid(row=row, column=2, sticky="w", padx=6, pady=4)
    row+=1
    ttk.Label(frame, text="Begriffe (Leerzeichen/Komma):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_kw=_mk_entry(frame, width=60); ent_kw.grid(row=row, column=1, columnspan=2, sticky="we", padx=6, pady=4); frame.columnconfigure(1, weight=1)
    row+=1
    ttk.Label(frame, text="Top-N:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_topn=_mk_entry(frame, width=8); ent_topn.insert(0,"8"); ent_topn.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    row+=1
    ttk.Label(frame, text="Ähnlichkeits-Schwelle (0–1):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_thr=_mk_entry(frame, width=8); ent_thr.insert(0,"0.3"); ent_thr.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row+=1
    ttk.Label(frame, text="Auflösung:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    resolution_var = tk.StringVar(value="Klein")
    ttk.Combobox(frame, textvariable=resolution_var, values=["Groß", "Mittel", "Klein"], width=12, state="readonly").grid(row=row, column=1, sticky="w", padx=6, pady=4)

    row+=1
    last_ctx={"ctx":""}
    btn_save_png = ttk.Button(frame, text="Plot (PNG) speichern", state="disabled",
                              command=lambda: ask_save_current_figure("Netzwerk", last_ctx["ctx"], root))
    btn_save_png.grid(row=row, column=2, padx=6, pady=6, sticky="e")

    def run():
        kv = ensure_model_loaded(root)
        if kv is None: return
        raw = ent_kw.get().strip()
        if not raw:
            messagebox.showerror("Fehler","Bitte Begriffe eingeben.", parent=root); return
        keywords=[w.strip() for w in raw.replace(",", " ").split() if w.strip()]
        try:
            topn=max(1,int(ent_topn.get().strip())); thr=float(ent_thr.get().strip())
            if not (0.0 <= thr <= 1.0): raise ValueError
        except Exception:
            messagebox.showerror("Fehler","Parameter prüfen.", parent=root); return
        G, sims = build_graph(kv, keywords, topn, thr)
        if not G or G.number_of_edges()==0:
            messagebox.showinfo("Keine Verbindungen","Keine Kanten über Schwelle.", parent=root); return
        
        # Auflösung (Klein/Mittel/Groß)
        resolution = resolution_var.get()
        if resolution == "Klein":
            figsize = (14, 12)
            node_size = 320
            font_size = 14
        elif resolution == "Mittel":
            figsize = (18, 16)
            node_size = 480
            font_size = 16
        else:  # Groß
            figsize = (24, 20)
            node_size = 640
            font_size = 18
        
        pos = nx.spring_layout(G, seed=42, k=0.5)
        edge_weights = [d["weight"] for (_, _, d) in G.edges(data=True)]
        w_min, w_max = min(edge_weights), max(edge_weights)
        norm_weights = [1.0]*len(edge_weights) if w_max==w_min else [(w-w_min)/(w_max-w_min) for w in edge_weights]
        
        # Erstelle Figure mit korrekter Größe
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="sandybrown", alpha=0.4, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)
        nx.draw_networkx_edges(G, pos, width=[max(0.5, w*4) for w in edge_weights], edge_color=plt.cm.Oranges(norm_weights), alpha=0.85, ax=ax)
        
        model_path = getattr(kv, "_loaded_path", str(CURRENT_MODEL_PATH))
        model_name = os.path.splitext(os.path.basename(model_path))[0].replace("_gen","").replace("_"," ")
        wrapped_keywords = ", ".join(keywords[:5]) + (" …" if len(keywords) > 5 else "")
        ax.set_title(f"Word2Vec-Ähnlichkeitsnetzwerk: {wrapped_keywords} | Top-N: {topn} | Schwelle: {thr:.2f} | Modell: {model_name}")
        ax.axis("off")
        plt.tight_layout()
        last_ctx["ctx"]=f"{'_'.join(keywords)}_{topn}_{str(thr).replace('.','_')}_{resolution}"
        btn_save_png.configure(state="normal")
        plt.show()

    ttk.Button(frame, text="Netzwerk erzeugen", command=run).grid(row=row, column=0, padx=6, pady=8, sticky="w")

# =========================
# Termset – Tabs (Cluster, Wortwolke, Dendrogramme)
# =========================

clean_re = re.compile(r"\s*\(.*?\)\s*$")
def clean_word(t: str) -> str: return clean_re.sub("", t.strip())

def load_termset_df(path: Path) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"Termliste nicht gefunden: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: str(c).strip())
    if df.empty: raise ValueError(f"Termliste leer: {path}")
    return df

def extract_word_tags(df: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, str], List[str]]:
    tags = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    wort_tag_map: Dict[str, Set[str]] = defaultdict(set); original_map: Dict[str, str] = {}
    for tag in tags:
        for raw in df[tag].dropna().astype(str):
            w = clean_word(raw)
            if not w: continue
            wort_tag_map[w].add(tag); original_map[w] = raw
    if not wort_tag_map: raise ValueError("Keine Wörter in den Tag-Spalten gefunden.")
    return wort_tag_map, original_map, tags

def vectors_for_words(kv: KeyedVectors, wort_tag_map: Dict[str, Set[str]], original_map: Dict[str, str]) -> Tuple[np.ndarray, List[str], List[str]]:
    words: List[str] = []; texts: List[str] = []; vecs: List[np.ndarray] = []
    in_vocab = kv.key_to_index
    for w in wort_tag_map:
        if w in in_vocab:
            words.append(w); texts.append(original_map[w]); vecs.append(kv[w])
    if not vecs: raise ValueError("Keine Begriffe der Termliste im W2V-Vokabular.")
    return np.asarray(vecs), texts, words

def build_markers(words: List[str], wort_tag_map: Dict[str, Set[str]], tags: List[str], use_markers: bool) -> Tuple[List[str], List[str]]:
    tag_index = {t: i for i, t in enumerate(tags)}
    def key_for(ts: Set[str]) -> Tuple[int, ...]: return tuple(sorted(tag_index[t] for t in ts))
    all_keys = sorted({key_for(wort_tag_map[w]) for w in words})
    marker_pool = ["o", "s", "D", "^", "v", "<", ">", "p", "P", "X", "*", "h", "H", "d", "8", ".", ","]
    key2marker = {k: (marker_pool[i % len(marker_pool)] if use_markers else "o") for i, k in enumerate(all_keys)}
    tag_labels: List[str] = []; markers: List[str] = []
    for w in words:
        k = key_for(wort_tag_map[w]); tag_labels.append(",".join(tags[i] for i in k)); markers.append(key2marker[k])
    return tag_labels, markers

def make_cluster_colors(k: int) -> Dict[int, str]:
    base = cm.get_cmap("tab10", 10)
    return {i: mcolors.to_hex(base(i % 10)) for i in range(k)}

def on_pick(event):
    artist = event.artist
    if not isinstance(artist, Text): return
    is_bold = str(artist.get_fontweight()).lower() in ("bold", "heavy", "700")
    artist.set_fontweight("normal" if is_bold else "bold")
    artist.set_fontsize(11 if is_bold else 13)
    artist.figure.canvas.draw_idle()

def plot_embedding_umap(xy: np.ndarray, texts: Optional[List[str]], clusters: np.ndarray,
                        tag_labels: List[str], markers: List[str], k: int,
                        clickable: bool, use_markers: bool,
                        save_path: Optional[Path] = None, save_dpi: int = 450, figsize: tuple = (15, 10)) -> None:
    colors = make_cluster_colors(k)
    fig, ax = plt.subplots(figsize=figsize)
    size = 80 if use_markers else 30
    for i, (x, y) in enumerate(xy):
        ax.scatter(x, y, c=colors[int(clusters[i])], marker=markers[i], edgecolors="k", s=size, alpha=0.85)
    
    # Nur Labels anzeigen wenn texts nicht None
    texts_obj=[]
    if texts is not None:
        for (x,y),txt in zip(xy, texts):
            t=ax.text(x,y,txt,fontsize=10,ha="center",va="center",picker=clickable); texts_obj.append(t)
        adjust_text(texts_obj, ax=ax)
        if clickable: fig.canvas.mpl_connect("pick_event", on_pick)
    
    ax.set_title(f"UMAP + agglomeratives Clustering (k={k})")
    
    # Cluster-Legende (IMMER anzeigen)
    cluster_ids = sorted(set(int(c) for c in clusters))
    cluster_handles=[Line2D([0],[0], marker="o", linestyle="", color=colors[c], label=f"Cluster {c}", markersize=10, markeredgecolor="k") for c in cluster_ids]
    leg_cluster=ax.legend(handles=cluster_handles, title="Cluster (Farben)", loc="upper left", bbox_to_anchor=(1.01,1))
    
    # Marker-Legende (NUR wenn use_markers aktiv)
    if use_markers:
        # Füge Cluster-Legende als Artist hinzu (damit sie nicht überschrieben wird)
        ax.add_artist(leg_cluster)
        combo_pairs = sorted({(tl, m) for tl, m in zip(tag_labels, markers)})
        marker_handles=[Line2D([0],[0], marker=m, linestyle="None", color="k", markersize=10, markeredgecolor="k") for _,m in combo_pairs]
        marker_labels=[tl for tl,_ in combo_pairs]
        ax.legend(handles=marker_handles, labels=marker_labels, title="Tag-Kombinationen (Marker)", loc="lower left", bbox_to_anchor=(1.01,0))
    
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=save_dpi)
    plt.show()

def build_tab_termset_cluster(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="Cluster")
    row=0
    ttk.Label(frame, text="Modell:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    model_label = ttk.Label(frame, text=str(CURRENT_MODEL_PATH)); model_label.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Button(frame, text="Öffnen …", command=lambda: choose_model(root, model_label)).grid(row=row, column=2, sticky="w", padx=6, pady=4)
    row+=1
    ttk.Label(frame, text="Termliste:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    term_label = ttk.Label(frame, text=str(DEFAULT_TERMLIST_PATH)); term_label.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Button(frame, text="Öffnen …", command=lambda: choose_termset(root, term_label)).grid(row=row, column=2, sticky="w", padx=6, pady=4)
    row+=1
    ttk.Label(frame, text="Clusteranzahl (k):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_k = _mk_entry(frame, width=8); ent_k.insert(0,"3"); ent_k.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row+=1
    ttk.Label(frame, text="UMAP-Parameter:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    row+=1
    ttk.Label(frame, text="  n_neighbors:").grid(row=row, column=0, sticky="w", padx=6, pady=2)
    ent_neighbors = _mk_entry(frame, width=8); ent_neighbors.insert(0, "15"); ent_neighbors.grid(row=row, column=1, sticky="w", padx=6, pady=2)
    row+=1
    ttk.Label(frame, text="  min_dist:").grid(row=row, column=0, sticky="w", padx=6, pady=2)
    ent_mindist = _mk_entry(frame, width=8); ent_mindist.insert(0, "0.05"); ent_mindist.grid(row=row, column=1, sticky="w", padx=6, pady=2)
    
    row+=1
    ttk.Label(frame, text="Auflösung:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    resolution_var = tk.StringVar(value="Klein")
    ttk.Combobox(frame, textvariable=resolution_var, values=["Klein", "Mittel", "Groß"], width=12, state="readonly").grid(row=row, column=1, sticky="w", padx=6, pady=4)
    row+=1
    clickable_var = tk.BooleanVar(value=True); use_markers_var = tk.BooleanVar(value=True)
    show_labels_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Ausdrücke anzeigen", variable=show_labels_var).grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frame, text="Labels klickbar", variable=clickable_var).grid(row=row, column=1, sticky="w", padx=6, pady=4)
    row+=1
    ttk.Checkbutton(frame, text="Tag-Kombinationen als Marker", variable=use_markers_var).grid(row=row, column=0, sticky="w", padx=6, pady=4)
    row+=1
    save_scatter_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text=f"Scatterplot automatisch speichern → {SCATTER_OUTPUT_DIR}", variable=save_scatter_var).grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=4)
    row+=1
    info = tk.Text(frame, height=8, width=90); info.grid(row=row, column=0, columnspan=3, padx=6, pady=6, sticky="nsew"); frame.rowconfigure(row, weight=1)

    def run():
        kv = ensure_model_loaded(root)
        if kv is None: return
        try:
            df = load_termset_df(CURRENT_TERMLIST_PATH)
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return
        try:
            k = int(ent_k.get().strip()); 
            if k < 1: raise ValueError
        except Exception:
            messagebox.showerror("Fehler", "Clusteranzahl k muss ≥ 1 sein.", parent=root); return
        wort_tag_map, original_map, tags = extract_word_tags(df)
        try:
            vecs, texts, words = vectors_for_words(kv, wort_tag_map, original_map)
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return
        
        # UMAP-Parameter auslesen
        try:
            n_neighbors = int(ent_neighbors.get().strip())
            min_dist = float(ent_mindist.get().strip())
        except Exception:
            messagebox.showerror("Fehler", "Ungültige UMAP-Parameter", parent=root); return
        
        # Auflösung (Klein/Mittel/Groß) → figsize
        resolution = resolution_var.get()
        if resolution == "Klein":
            figsize = (12, 8)
            save_dpi = 450
        elif resolution == "Mittel":
            figsize = (16, 12)
            save_dpi = 600
        else:  # Groß
            figsize = (20, 16)
            save_dpi = 750
        
        n_points = vecs.shape[0]; k_eff = min(max(1, k), n_points)
        n_neighbors = min(n_neighbors, max(2, n_points - 1))
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=42)
        xy = reducer.fit_transform(vecs)
        labels = np.zeros(n_points, dtype=int) if k_eff==1 else AgglomerativeClustering(n_clusters=k_eff, linkage="ward").fit_predict(vecs)
        tag_labels, markers = build_markers(words, wort_tag_map, tags, use_markers_var.get())
        beispiele: Dict[int, List[str]] = defaultdict(list)
        for txt, lab in zip(texts, labels): beispiele[int(lab)].append(txt)
        info.delete(1.0, tk.END); info.insert(tk.END, "Beispielwörter pro Cluster:\n")
        for cid in sorted(beispiele): info.insert(tk.END, f"Cluster {cid}: {', '.join(beispiele[cid][:10])}\n")
        save_path = None
        termset_name = Path(CURRENT_TERMLIST_PATH).stem
        if save_scatter_var.get():
            filename = f"{termset_name}_k{k_eff}_{resolution}.png"; save_path = SCATTER_OUTPUT_DIR / filename
        plot_embedding_umap(xy, texts if show_labels_var.get() else None, labels, tag_labels, markers, k_eff, clickable_var.get(), use_markers_var.get(), save_path=save_path, save_dpi=save_dpi, figsize=figsize)

    ttk.Button(frame, text="UMAP-Cluster berechnen", command=run).grid(row=row+1, column=0, padx=6, pady=8, sticky="w")

def build_tab_termset_wordcloud(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame=ttk.Frame(parent_nb); parent_nb.add(frame, text="Wortwolke")
    row=0
    ttk.Label(frame, text="Termset-Datei:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    term_label = ttk.Label(frame, text=str(DEFAULT_TERMLIST_PATH)); term_label.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Button(frame, text="Öffnen …", command=lambda: choose_termset(root, term_label)).grid(row=row, column=2, sticky="w", padx=6, pady=4)
    row+=1
    ttk.Label(frame, text="TF-IDF Datei:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    tfidf_var=tk.StringVar(value=str(DATA.path_tfidf_for_cloud))
    ent_tfidf=_mk_entry(frame, width=60, textvariable=tfidf_var); ent_tfidf.grid(row=row, column=1, sticky="we", padx=6, pady=4); frame.columnconfigure(1, weight=1)
    ttk.Button(frame, text="…", command=lambda: (lambda p=filedialog.askopenfilename(parent=root, initialdir=str(Path(tfidf_var.get()).parent), filetypes=[("CSV","*.csv")]): (tfidf_var.set(p) if p else None))()).grid(row=row, column=2, sticky="w", padx=6, pady=4)
    row+=1
    ttk.Label(frame, text="Farbschema (matplotlib cmap):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    cmap_var=tk.StringVar(value="tab10"); _mk_entry(frame, textvariable=cmap_var, width=16).grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row+=1
    whole_word_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="☑ Nur ganze Wörter (\\b Grenzen)", variable=whole_word_var).grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=4)

    row+=1
    btn_save_png = ttk.Button(frame, text="Cloud (PNG) speichern", state="disabled")
    btn_save_png.grid(row=row, column=2, padx=6, pady=6, sticky="e")

    def run():
        nonlocal btn_save_png
        if not WORDCLOUD_AVAILABLE:
            messagebox.showerror("Fehler", "wordcloud-Paket nicht installiert.", parent=root); return
        try:
            df_words = load_termset_df(CURRENT_TERMLIST_PATH)
        except Exception as e:
            messagebox.showerror("Fehler", f"Termset: {e}", parent=root); return
        try:
            df_tfidf = pd.read_csv(Path(tfidf_var.get()))
            expr = df_tfidf.iloc[:, METADATA_COLS:]
            keep_cols = [c for c in expr.columns
                         if pd.api.types.is_numeric_dtype(expr[c]) and str(c) not in META_NAME_BLACKLIST]
            expr = expr[keep_cols]
            tfidf_avg = expr.mean(axis=0, numeric_only=True).reset_index()
            tfidf_avg.columns=["word","tfidf_avg"]
        except Exception as e:
            messagebox.showerror("Fehler", f"TF-IDF: {e}", parent=root); return

        word_infos=[]
        for tag in df_words.columns:
            for word in df_words[tag].dropna():
                word=str(word).strip().lower()
                # Bei "Nur ganze Wörter": exakte Übereinstimmung, sonst Teilstring
                if whole_word_var.get():
                    val = tfidf_avg.loc[tfidf_avg["word"]==word, "tfidf_avg"]
                else:
                    # Teilstring-Suche: alle Wörter die den Suchstring enthalten
                    val = tfidf_avg.loc[tfidf_avg["word"].str.contains(word, case=False, na=False, regex=False), "tfidf_avg"]
                if not val.empty:
                    word_infos.append({"word":word, "tag":tag, "tfidf":float(val.values[0])})
        if not word_infos:
            messagebox.showinfo("Info","Keine Überschneidung Termset ↔ TF-IDF.", parent=root); return
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
        word_size_dict_scaled = {w: np.log(v + 1.0) for w, v in word_size_dict.items()}
        wc = WordCloud(width=1200, height=600, background_color="white", prefer_horizontal=1.0)
        wc.generate_from_frequencies(word_size_dict_scaled)
        wc.recolor(color_func=color_func)
        fig=plt.figure(figsize=(16,8)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off"); plt.tight_layout()
        btn_save_png.configure(state="normal", command=lambda: ask_save_current_figure("Wortwolke", Path(CURRENT_TERMLIST_PATH).stem, root, fig=fig))
        plt.show()

    ttk.Button(frame, text="Erzeugen", command=run).grid(row=row, column=0, padx=6, pady=6, sticky="w")

def build_tab_termset_dendro(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(parent_nb); parent_nb.add(frame, text="Dendrogramme")
    row=0
    ttk.Label(frame, text="Termliste:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    term_label = ttk.Label(frame, text=str(DEFAULT_TERMLIST_PATH)); term_label.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Button(frame, text="Öffnen …", command=lambda: choose_termset(root, term_label)).grid(row=row, column=2, sticky="w", padx=6, pady=4)
    row+=1
    ttk.Label(frame, text="Clusteranzahl (k):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_k = _mk_entry(frame, width=8); ent_k.insert(0,"2"); ent_k.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    row+=1
    info = tk.Text(frame, height=8, width=90); info.grid(row=row, column=0, columnspan=3, padx=6, pady=6, sticky="nsew"); frame.rowconfigure(row, weight=1)

    def run():
        kv = ensure_model_loaded(root)
        if kv is None: return
        try:
            df = load_termset_df(CURRENT_TERMLIST_PATH)
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return
        try:
            k = int(ent_k.get().strip()); k = max(1, k)
        except Exception:
            messagebox.showerror("Fehler", "k muss ganzzahlig ≥1 sein.", parent=root); return

        wort_tag_map = defaultdict(set); original_map={}
        for tag in df.columns:
            for eintrag in df[tag].dropna().astype(str):
                wort = clean_re.sub("", eintrag.strip())
                if not wort: continue
                wort_tag_map[wort].add(tag); original_map[wort] = eintrag.strip()

        words: List[str] = []; vecs: List[np.ndarray] = []
        for wort in wort_tag_map:
            if wort in kv.key_to_index:
                words.append(original_map.get(wort, wort)); vecs.append(kv[wort])
        if not vecs:
            messagebox.showerror("Fehler", "Keine Terme aus der Liste im Word2Vec-Modell gefunden.", parent=root); return

        vecs_np = np.array(vecs, dtype=np.float32)
        n = vecs_np.shape[0]; k_eff = min(max(1, k), n)
        labels = np.zeros(n, dtype=int) if k_eff==1 else AgglomerativeClustering(n_clusters=k_eff).fit_predict(vecs_np)
        termset_name = Path(CURRENT_TERMLIST_PATH).stem
        model_name = Path(getattr(kv, "_loaded_path", CURRENT_MODEL_PATH)).stem
        out_dir = DENDRO_OUTPUT_DIR / f"Dendrogramme_k{k_eff}_{termset_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        created: List[Path] = []
        for cluster_id in range(k_eff):
            idx = np.where(labels == cluster_id)[0]
            if len(idx) < 2: continue
            cluster_words = [words[i] for i in idx]
            cluster_vecs = vecs_np[idx]
            linkage_matrix = linkage(cluster_vecs, method="average", metric="cosine")
            plt.figure(figsize=(10, max(6, len(cluster_words) * 0.3)))
            dendrogram(linkage_matrix, labels=cluster_words, orientation="right", leaf_font_size=9, color_threshold=None)
            plt.title(f"Dendrogramm – Cluster {cluster_id}"); plt.tight_layout()
            filename = f"dendro_{model_name}_k{k_eff}_cluster{cluster_id}.png"
            filepath = out_dir / filename
            plt.savefig(filepath, dpi=450); plt.close()
            created.append(filepath)
        msg_lines = [f"Dendrogramme gespeichert in: {out_dir}"] + [f"  - {p.name}" for p in created]
        info.delete(1.0, tk.END); info.insert(tk.END, "\n".join(msg_lines))

    ttk.Button(frame, text="Dendrogramme erzeugen", command=run).grid(row=row+1, column=0, padx=6, pady=8, sticky="w")

# =========================
# Text-Cluster
# =========================

def build_tab_texts_scatter(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    """
    Tab für UMAP-Scatterplot mit optionalem hierarchischem Clustering.
    Arbeitet mit Kosinus-Distanzmatrizen aus der Pipeline.
    """
    frame = ttk.Frame(parent_nb)
    parent_nb.add(frame, text="Streudiagramm")
    
    row = 0
    
    # =========================================================================
    # INFO
    # =========================================================================
    
    info_frame = ttk.LabelFrame(frame, text="ℹ️ Datenquellen", padding=8)
    info_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
    
    ttk.Label(
        info_frame,
        text="💡 Tipp: Dateien können im Tab 'Daten' zentral verwaltet werden.",
        foreground="blue"
    ).pack(anchor="w")
    
    row += 1
    
    # =========================================================================
    # DATEIAUSWAHL
    # =========================================================================
    
    default_cosine = DATA.path_cosine if hasattr(DATA, 'path_cosine') else PROJECT_ROOT / "output" / "cosine" / "cosine_tfidf2000.csv"
    default_corpus = DATA.path_metadata
    
    ttk.Label(frame, text="Kosinus-Matrix:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    cosine_var = tk.StringVar(value=str(default_cosine))
    ent_cosine = _mk_entry(frame, width=70, textvariable=cosine_var)
    ent_cosine.grid(row=row, column=1, sticky="we", padx=6, pady=4)
    frame.columnconfigure(1, weight=1)
    
    def browse_cosine():
        p = filedialog.askopenfilename(
            parent=root,
            title="Kosinus-Matrix wählen",
            initialdir=str(default_cosine.parent),
            filetypes=[("CSV", "*.csv"), ("Alle Dateien", "*.*")]
        )
        if p:
            cosine_var.set(p)
    
    ttk.Button(frame, text="…", width=3, command=browse_cosine).grid(row=row, column=2, sticky="w", padx=4)
    
    row += 1
    ttk.Label(frame, text="Korpus-Metadaten:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    corpus_var = tk.StringVar(value=str(default_corpus))
    ent_corpus = _mk_entry(frame, width=70, textvariable=corpus_var)
    ent_corpus.grid(row=row, column=1, sticky="we", padx=6, pady=4)
    
    def browse_corpus():
        p = filedialog.askopenfilename(
            parent=root,
            title="Korpus-Metadaten wählen",
            initialdir=str(default_corpus.parent),
            filetypes=[("CSV", "*.csv"), ("Alle Dateien", "*.*")]
        )
        if p:
            corpus_var.set(p)
    
    ttk.Button(frame, text="…", width=3, command=browse_corpus).grid(row=row, column=2, sticky="w", padx=4)
    
    row += 1
    
    def load_from_data_tab():
        corpus_var.set(str(DATA.path_metadata))
        if hasattr(DATA, 'path_cosine'):
            cosine_var.set(str(DATA.path_cosine))
        messagebox.showinfo(
            "Übernommen",
            f"Pfade aus Daten-Tab übernommen:\n\n"
            f"Korpus: {DATA.path_metadata.name}\n" +
            (f"Kosinus: {DATA.path_cosine.name}" if hasattr(DATA, 'path_cosine') else ""),
            parent=root
        )
    
    ttk.Button(
        frame,
        text="📥 Pfade aus Daten-Tab übernehmen",
        command=load_from_data_tab
    ).grid(row=row, column=0, columnspan=2, padx=6, pady=4, sticky="w")
    
    row += 1
    ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=8)
    
    # =========================================================================
    # CLUSTERING
    # =========================================================================
    
    row += 1
    clustering_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        frame, text="☑ Hierarchisches Clustering aktivieren",
        variable=clustering_var
    ).grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="  Clusteranzahl (k):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_k = _mk_entry(frame, width=8)
    ent_k.insert(0, "5")
    ent_k.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="  Linkage-Methode:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    linkage_var = tk.StringVar(value="ward")
    ttk.Combobox(
        frame,
        textvariable=linkage_var,
        values=["ward", "average", "complete", "single"],
        width=12,
        state="readonly"
    ).grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=8)
    
    # =========================================================================
    # UMAP
    # =========================================================================
    
    row += 1
    umap_frame = ttk.LabelFrame(frame, text="UMAP-Parameter", padding=8)
    umap_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=4)
    
    umap_row = 0
    ttk.Label(umap_frame, text="n_neighbors (5-50):").grid(row=umap_row, column=0, sticky="w", padx=6, pady=4)
    ent_neighbors = _mk_entry(umap_frame, width=8)
    ent_neighbors.insert(0, "15")
    ent_neighbors.grid(row=umap_row, column=1, sticky="w", padx=6, pady=4)
    
    umap_row += 1
    ttk.Label(umap_frame, text="min_dist (0.0-0.99):").grid(row=umap_row, column=0, sticky="w", padx=6, pady=4)
    ent_dist = _mk_entry(umap_frame, width=8)
    ent_dist.insert(0, "0.1")
    ent_dist.grid(row=umap_row, column=1, sticky="w", padx=6, pady=4)
    
    umap_row += 1
    ttk.Label(
        umap_frame,
        text="ℹ️ Metrik: Cosine (via precomputed Distanzmatrix)",
        foreground="gray"
    ).grid(row=umap_row, column=0, columnspan=2, sticky="w", padx=6, pady=2)
    
    row += 1
    ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=8)
    
    # =========================================================================
    # VISUALISIERUNG
    # =========================================================================
    
    row += 1
    ttk.Label(frame, text="Visualisierung:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="  Färbung nach:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    color_mode_var = tk.StringVar(value="auto")
    ttk.Combobox(
        frame,
        textvariable=color_mode_var,
        values=["auto", "cluster", "textclass", "year", "author_surname"],
        width=15,
        state="readonly"
    ).grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="  Marker-Größe:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_size = _mk_entry(frame, width=8)
    ent_size.insert(0, "8")
    ent_size.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="  Transparenz (0.0-1.0):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_opacity = _mk_entry(frame, width=8)
    ent_opacity.insert(0, "0.7")
    ent_opacity.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=8)
    
    # =========================================================================
    # INFO-BOX
    # =========================================================================
    
    row += 1
    ttk.Label(frame, text="Info:").grid(row=row, column=0, sticky="nw", padx=6, pady=4)
    info_text = tk.Text(frame, height=10, width=90, wrap="word")
    info_text.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=6, pady=4)
    frame.rowconfigure(row, weight=1)
    
    info_scroll = ttk.Scrollbar(frame, orient="vertical", command=info_text.yview)
    info_text.configure(yscrollcommand=info_scroll.set)
    info_scroll.grid(row=row, column=3, sticky="ns")
    
    # =========================================================================
    # BUTTONS & STATE
    # =========================================================================
    
    row += 1
    
    state = {
        'last_fig': None,
        'last_df': None,
        'last_context': ""
    }
    
    btn_compute = ttk.Button(frame, text="🔄 Berechnen")
    btn_compute.grid(row=row, column=0, padx=6, pady=8, sticky="w")
    
    # KEIN PNG-Button (kaleido benötigt)
    btn_save_html = ttk.Button(frame, text="🌐 HTML speichern", state="disabled")
    btn_save_html.grid(row=row, column=1, padx=6, pady=8, sticky="w")
    
    btn_save_csv = ttk.Button(frame, text="📄 CSV speichern", state="disabled")
    btn_save_csv.grid(row=row, column=2, padx=6, pady=8, sticky="w")
    
    # =========================================================================
    # COMPUTE
    # =========================================================================
    
    def compute():
        info_text.delete(1.0, tk.END)
        info_text.insert(tk.END, "🔄 Lade Daten...\n")
        info_text.insert(tk.END, "📏 Metrik: Cosine (precomputed)\n\n")
        root.update_idletasks()
        
        # 1. KOSINUS-MATRIX
        try:
            cosine_path = Path(cosine_var.get())
            if not cosine_path.exists():
                raise FileNotFoundError(f"Nicht gefunden: {cosine_path}")
            
            info_text.insert(tk.END, f"📄 Lade Matrix: {cosine_path.name}\n")
            
            if hasattr(DATA, 'cosine_df') and DATA.cosine_df is not None and str(DATA.path_cosine) == str(cosine_path):
                info_text.insert(tk.END, "♻️ Nutze Cache\n")
                cos_df = DATA.cosine_df.copy()
            else:
                cos_df = pd.read_csv(cosine_path, index_col=0)
            
            cosine_matrix = cos_df.values
            doc_ids = cos_df.index.tolist()
            
            info_text.insert(tk.END, f"✔ Matrix: {cosine_matrix.shape[0]:,} × {cosine_matrix.shape[1]:,}\n")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Kosinus-Matrix:\n{e}", parent=root)
            info_text.insert(tk.END, f"❌ {e}\n")
            return
        
        # 2. METADATEN
        try:
            path_metadata = Path(corpus_var.get())
            if not path_metadata.exists():
                raise FileNotFoundError(f"Nicht gefunden: {path_metadata}")
            
            info_text.insert(tk.END, f"📄 Lade Metadaten: {path_metadata.name}\n")
            
            if DATA.corpus_df is not None and str(DATA.path_metadata) == str(path_metadata):
                info_text.insert(tk.END, "♻️ Nutze Cache\n")
                metadata_df = DATA.corpus_df.copy()
            else:
                metadata_df = pd.read_csv(path_metadata, sep=";")
            
            ensure_doc_id_inplace(metadata_df)
            
            doc_ids_str = [str(d) for d in doc_ids]
            metadata_df["doc_id"] = metadata_df["doc_id"].astype(str)
            metadata_df = metadata_df[metadata_df["doc_id"].isin(doc_ids_str)]
            metadata_df = metadata_df.set_index("doc_id").reindex(doc_ids_str).reset_index()
            
            info_text.insert(tk.END, f"✔ Metadaten: {len(metadata_df):,} Docs\n")
            
            if len(cosine_matrix) != len(metadata_df):
                raise ValueError(f"Mismatch: Matrix={len(cosine_matrix)}, Meta={len(metadata_df)}")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Metadaten:\n{e}", parent=root)
            info_text.insert(tk.END, f"❌ {e}\n")
            return
        
        # 3. PARAMETER
        try:
            n_neighbors = max(2, min(50, int(ent_neighbors.get().strip())))
            n_neighbors = min(n_neighbors, len(metadata_df) - 1)
            min_dist = max(0.0, min(0.99, float(ent_dist.get().strip())))
            marker_size = max(1, min(50, int(ent_size.get().strip())))
            opacity = max(0.0, min(1.0, float(ent_opacity.get().strip())))
        except Exception as e:
            messagebox.showerror("Fehler", f"Parameter:\n{e}", parent=root)
            return
        
        # 4. DISTANZ
        info_text.insert(tk.END, "\n🔄 Distanzmatrix...\n")
        root.update_idletasks()
        
        distance_matrix = 1 - cosine_matrix
        distance_matrix = np.clip(distance_matrix, 0, None)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        info_text.insert(tk.END, "✔ Distanz berechnet\n")
        
        # 5. UMAP
        info_text.insert(tk.END, f"\n🔄 UMAP (n={n_neighbors}, d={min_dist})...\n")
        root.update_idletasks()
        
        try:
            import time
            start = time.time()
            
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric='precomputed',
                random_state=42
            )
            
            umap_results = reducer.fit_transform(distance_matrix)
            elapsed = time.time() - start
            
            info_text.insert(tk.END, f"✔ UMAP fertig ({elapsed:.1f}s)\n")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"UMAP:\n{e}", parent=root)
            info_text.insert(tk.END, f"❌ UMAP: {e}\n")
            return
        
        # 6. CLUSTERING (optional)
        clusters = None
        k_eff = 0
        
        if clustering_var.get():
            try:
                k = int(ent_k.get().strip())
                k_eff = max(1, min(k, len(metadata_df)))
                linkage_method = linkage_var.get()
                
                info_text.insert(tk.END, f"\n🔄 Clustering (k={k_eff}, {linkage_method})...\n")
                root.update_idletasks()
                
                if k_eff == 1:
                    clusters = np.zeros(len(metadata_df), dtype=int)
                else:
                    if linkage_method == 'ward':
                        from sklearn.manifold import MDS
                        n_comp = min(50, len(distance_matrix) - 1)
                        mds = MDS(n_components=n_comp, dissimilarity='precomputed', random_state=42)
                        X_embedded = mds.fit_transform(distance_matrix)
                        agglo = AgglomerativeClustering(n_clusters=k_eff, linkage=linkage_method)
                        clusters = agglo.fit_predict(X_embedded)
                    else:
                        from scipy.spatial.distance import squareform
                        from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
                        condensed_dist = squareform(distance_matrix, checks=False)
                        Z = scipy_linkage(condensed_dist, method=linkage_method)
                        clusters = fcluster(Z, k_eff, criterion='maxclust') - 1
                
                unique, counts = np.unique(clusters, return_counts=True)
                info_text.insert(tk.END, f"✔ Clustering fertig\n\n📊 Verteilung:\n")
                for cid, cnt in zip(unique, counts):
                    info_text.insert(tk.END, f"   Cluster {cid}: {cnt:,} ({cnt/len(clusters)*100:.1f}%)\n")
                
            except Exception as e:
                messagebox.showerror("Fehler", f"Clustering:\n{e}", parent=root)
                info_text.insert(tk.END, f"❌ Clustering: {e}\n")
                return
        
        # 7. DATAFRAME
        umap_df = pd.DataFrame(umap_results, columns=["UMAP-1", "UMAP-2"])
        umap_df = umap_df.join(metadata_df.reset_index(drop=True))
        
        if clusters is not None:
            umap_df["cluster"] = clusters
            umap_df["cluster_label"] = "Cluster " + umap_df["cluster"].astype(str)
        
        # Hover-Text (wie im Beispiel)
        umap_df["hover_text"] = (
            umap_df["doc_id"].astype(str) + "<br>" +
            "Author: " + umap_df.get("author_surname", pd.Series("N/A")).fillna("N/A").astype(str) + "<br>" +
            "Title: " + umap_df.get("title", pd.Series("N/A")).fillna("N/A").astype(str)
        )
        
        # 8. LEGEND LABEL & FÄRBUNG (wie im Beispiel)
        color_mode = color_mode_var.get()
        color_column = None
        color_discrete_map = None
        
        if color_mode == "auto":
            if clusters is not None:
                color_column = "cluster_label"
            elif "textclass" in umap_df.columns and umap_df["textclass"].notna().any():
                color_column = "textclass"
                # Legend-Label: Textclass + ID
                umap_df["legend_label"] = umap_df["textclass"].fillna("Unbekannt") + " - " + umap_df["doc_id"].astype(str)
                
                # Farbschema wie im Beispiel
                unique_textclasses = umap_df["textclass"].dropna().unique()
                color_map_tc = {
                    cls: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    for i, cls in enumerate(unique_textclasses)
                }
                color_discrete_map = {
                    lbl: color_map_tc.get(cls, "#CCCCCC")
                    for lbl, cls in zip(umap_df["legend_label"], umap_df["textclass"])
                }
                color_column = "legend_label"
            elif "year_final" in umap_df.columns or "year" in umap_df.columns:
                year_col = "year_final" if "year_final" in umap_df.columns else "year"
                umap_df[year_col] = pd.to_numeric(umap_df[year_col], errors='coerce')
                color_column = year_col
        elif color_mode == "cluster":
            if clusters is not None:
                color_column = "cluster_label"
            else:
                messagebox.showwarning("Hinweis", "Clustering nicht aktiv", parent=root)
        elif color_mode == "textclass":
            if "textclass" in umap_df.columns and umap_df["textclass"].notna().any():
                umap_df["legend_label"] = umap_df["textclass"].fillna("Unbekannt") + " - " + umap_df["doc_id"].astype(str)
                unique_textclasses = umap_df["textclass"].dropna().unique()
                color_map_tc = {
                    cls: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    for i, cls in enumerate(unique_textclasses)
                }
                color_discrete_map = {
                    lbl: color_map_tc.get(cls, "#CCCCCC")
                    for lbl, cls in zip(umap_df["legend_label"], umap_df["textclass"])
                }
                color_column = "legend_label"
        elif color_mode == "year":
            year_col = "year_final" if "year_final" in umap_df.columns else "year"
            if year_col in umap_df.columns:
                umap_df[year_col] = pd.to_numeric(umap_df[year_col], errors='coerce')
                color_column = year_col
        elif color_mode == "author_surname":
            if "author_surname" in umap_df.columns:
                color_column = "author_surname"
        
        # 9. PLOT (wie im Beispiel)
        info_text.insert(tk.END, "\n🎨 Erstelle Plot...\n")
        root.update_idletasks()
        
        try:
            fig = px.scatter(
                umap_df,
                x="UMAP-1",
                y="UMAP-2",
                color=color_column if color_column else "doc_id",
                hover_name="hover_text",
                opacity=opacity,
                color_discrete_map=color_discrete_map,
                labels={color_column: "Textklasse & ID"} if color_column == "legend_label" else {}
            )
            
            title_parts = [f"UMAP ({len(umap_df):,} Docs)"]
            if clusters is not None:
                title_parts.append(f"k={k_eff}, {linkage_method}")
            title_parts.append(f"n_neighbors={n_neighbors}, min_dist={min_dist}")
            
            fig.update_layout(
                title=" | ".join(title_parts),
                width=1200,
                height=800,
                hovermode='closest',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255,255,255,0.5)",
                    itemsizing='constant',
                    title_font=dict(size=14),
                    font=dict(size=10),
                    itemwidth=30,
                    orientation="v",
                    traceorder="normal"
                )
            )
            
            # Marker wie im Beispiel
            fig.update_traces(marker=dict(size=marker_size, line=dict(width=0.5, color='white')))
            
            state['last_fig'] = fig
            state['last_df'] = umap_df
            state['last_context'] = f"scatter_{'cluster' if clusters is not None else 'plain'}_{len(umap_df)}"
            
            info_text.insert(tk.END, "✔ Plot erstellt\n")
            info_text.insert(tk.END, "\n💡 Öffne Browser...\n")
            
            btn_save_html.configure(state="normal")
            btn_save_csv.configure(state="normal")
            
            fig.show()
            
            info_text.insert(tk.END, "✔ Browser geöffnet!\n")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Plot:\n{e}", parent=root)
            info_text.insert(tk.END, f"❌ Plot: {e}\n")
            import traceback
            info_text.insert(tk.END, f"\n{traceback.format_exc()}\n")
    
    # =========================================================================
    # SAVE FUNCTIONS
    # =========================================================================
    
    def save_html():
        if state['last_fig'] is None:
            messagebox.showwarning("Hinweis", "Bitte erst berechnen", parent=root)
            return
        
        try:
            initdir = EXPLORATION_DIR / "Streudiagramm"
            initdir.mkdir(parents=True, exist_ok=True)
            
            path = filedialog.asksaveasfilename(
                parent=root,
                title="HTML speichern",
                initialdir=str(initdir),
                initialfile=f"{state['last_context']}_interactive.html",
                defaultextension=".html",
                filetypes=[("HTML", "*.html"), ("Alle", "*.*")]
            )
            
            if not path:
                return
            
            state['last_fig'].write_html(path)
            messagebox.showinfo("Gespeichert", f"HTML:\n{path}", parent=root)
            info_text.insert(tk.END, f"\n💾 HTML: {Path(path).name}\n")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"HTML:\n{e}", parent=root)
    
    def save_csv():
        if state['last_df'] is None:
            messagebox.showwarning("Hinweis", "Bitte erst berechnen", parent=root)
            return
        
        try:
            initdir = EXPLORATION_DIR / "Streudiagramm"
            initdir.mkdir(parents=True, exist_ok=True)
            
            cols_to_save = ["doc_id", "UMAP-1", "UMAP-2"]
            if "cluster" in state['last_df'].columns:
                cols_to_save.append("cluster")
            for col in ["author_surname", "title", "year_final", "year", "textclass", "source"]:
                if col in state['last_df'].columns:
                    cols_to_save.append(col)
            
            df_export = state['last_df'][cols_to_save].copy()
            
            path = filedialog.asksaveasfilename(
                parent=root,
                title="CSV speichern",
                initialdir=str(initdir),
                initialfile=f"{state['last_context']}_data.csv",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("Alle", "*.*")]
            )
            
            if not path:
                return
            
            df_export.to_csv(path, index=False)
            messagebox.showinfo("Gespeichert", f"CSV:\n{path}\n\nSpalten: {', '.join(cols_to_save)}", parent=root)
            info_text.insert(tk.END, f"\n💾 CSV: {Path(path).name} ({len(df_export):,} Zeilen)\n")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"CSV:\n{e}", parent=root)
    
    # === COMMANDS ===
    btn_compute.configure(command=compute)
    btn_save_html.configure(command=save_html)
    btn_save_csv.configure(command=save_csv)

# =========================
# Topics – Verläufe (ab 1840) + Optionen
# =========================

def build_tab_topics(parent_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame=ttk.Frame(parent_nb); parent_nb.add(frame, text="Topicverläufe")
    row=0
    ttk.Label(frame, text="Document-Topic-Matrix:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    topics_label = ttk.Label(frame, text=str(DATA.path_topics)); topics_label.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    def pick_topicsfile():
        p=filedialog.askopenfilename(parent=root, initialdir=str(DATA.path_topics.parent), filetypes=[("CSV","*.csv")])
        if p:
            DATA.path_topics=Path(p); DATA.topics_df=None; topics_label.config(text=p); load_topics_to_listbox()
    ttk.Button(frame, text="…", command=pick_topicsfile).grid(row=row, column=2, sticky="w", padx=6, pady=4)

    row+=1
    ttk.Label(frame, text="Metadata:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    meta_label = ttk.Label(frame, text=str(DATA.path_metadata)); meta_label.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    def pick_metafile():
        p=filedialog.askopenfilename(parent=root, initialdir=str(DATA.path_metadata.parent), filetypes=[("CSV","*.csv")])
        if p:
            DATA.path_metadata=Path(p); DATA.metadata_df=None; meta_label.config(text=p)
    ttk.Button(frame, text="…", command=pick_metafile).grid(row=row, column=2, sticky="w", padx=6, pady=4)

    row+=1
    ttk.Label(frame, text="Schwelle (Cosinus) für Zählung:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_thr = _mk_entry(frame, width=8); ent_thr.insert(0,"0.2"); ent_thr.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    row+=1
    ttk.Label(frame, text="Glättung (MA-Fenster):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_ma = _mk_entry(frame, width=8); ent_ma.insert(0,"3"); ent_ma.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    row+=1
    ttk.Label(frame, text="Polynom-Grad:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_deg = _mk_entry(frame, width=8); ent_deg.insert(0,"3"); ent_deg.grid(row=row, column=1, sticky="w", padx=6, pady=4)

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
    listbox.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=6, pady=6); frame.rowconfigure(row, weight=1)

    def load_topics_to_listbox():
        listbox.delete(0, tk.END)
        try:
            df = DATA.load_topics()
            for c in sorted(df.columns.tolist()):
                listbox.insert(tk.END, c)
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)

    load_topics_to_listbox()

    def compute():
        try:
            df_topics = DATA.load_topics()
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

        idx = df_topics.index.astype(str)
        jahr_mapping = dict(zip(mapping_df['_id'], mapping_df['Jahr_final']))
        df = df_topics.copy()
        df['Jahr'] = idx.map(jahr_mapping)
        df = df.dropna(subset=['Jahr'])
        df['Jahr'] = df['Jahr'].astype(int)
        df = df[df['Jahr'] >= 1840]

        selected_indices = listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Fehler","Keine Topics ausgewählt.", parent=root); return
        selected_topics = [listbox.get(i) for i in selected_indices]

        df_grouped = df.groupby('Jahr').mean().fillna(0.0)

        if abs_var.get():
            plt.figure(figsize=(14,8))
            for topic_label in selected_topics:
                years = df_grouped.index.values
                values = df_grouped[topic_label].values
                plt.plot(years, values, label=topic_label)
            plt.xlabel('Jahr'); plt.ylabel('Durchschnittliche Cosinus-Ähnlichkeit')
            plt.title('Absolute Topic-Verläufe (Jahresmittel)')
            plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True); plt.tight_layout()
            plt.show()

        if smooth_var.get():
            plt.figure(figsize=(14,8))
            for topic_label in selected_topics:
                years = df_grouped.index.values
                values = df_grouped[topic_label].values
                values_ma = pd.Series(values).rolling(window=ma, min_periods=1, center=True).mean()
                plt.plot(years, values_ma, label=topic_label)
            plt.xlabel('Jahr'); plt.ylabel('Durchschnittliche Cosinus-Ähnlichkeit')
            plt.title('Gleitender Mittelwert der ausgewählten Topics')
            plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True); plt.tight_layout()
            plt.show()

        relevant_counts = pd.DataFrame(index=sorted(df['Jahr'].unique()))
        for topic_label in selected_topics:
            counts_per_year = df.groupby('Jahr')[topic_label].apply(lambda x: (x >= thr).sum())
            relevant_counts[topic_label] = counts_per_year
        plt.figure(figsize=(14,8))
        for topic_label in selected_topics:
            plt.plot(relevant_counts.index, relevant_counts[topic_label], label=topic_label)
        plt.xlabel('Jahr'); plt.ylabel(f'Anzahl Texte mit Cosinus ≥ {thr}')
        plt.title(f'Anzahl relevanter Dokumente pro Jahr (Schwelle {thr})')
        plt.grid(True); plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        if poly_var.get():
            plt.figure(figsize=(14,8))
            for topic_label in selected_topics:
                years = df_grouped.index.values
                values = df_grouped[topic_label].values
                mask = ~np.isnan(values)
                years_clean = years[mask]; values_clean = values[mask]
                if len(years_clean) < deg + 1:
                    continue
                z = np.polyfit(years_clean, values_clean, deg)
                p = np.poly1d(z); values_poly = p(years_clean)
                plt.plot(years_clean, values_poly, label=topic_label)
            if plt.gca().has_data():
                plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xlabel('Jahr'); plt.ylabel('Durchschnittliche Cosinus-Ähnlichkeit')
            plt.title(f'Polynomiale Regression (Grad {deg}) der ausgewählten Topics')
            plt.grid(True); plt.tight_layout()
            plt.show()

    ttk.Button(frame, text="Berechnen", command=compute).grid(row=row+1, column=0, padx=6, pady=8, sticky="w")

# =========================
# Daten-Tab (globale Pfade & Prüfen)
# =========================

def build_tab_data(root_nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(root_nb); root_nb.add(frame, text="Daten")
    row=0

    def row_pick_csv(label: str, setter: callable, default_path: Path, after: Optional[callable]=None):
        nonlocal row
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        var=tk.StringVar(value=str(default_path))
        ent=_mk_entry(frame, width=80, textvariable=var); ent.grid(row=row, column=1, sticky="we", padx=6, pady=4)
        def browse():
            p=filedialog.askopenfilename(parent=root, initialdir=str(default_path.parent), title=label,
                                         filetypes=[("CSV", "*.csv"), ("Alle Dateien", "*.*")])
            if p:
                var.set(p); setter(Path(p)); 
                if after: after()
        ttk.Button(frame, text="…", width=3, command=browse).grid(row=row, column=2, sticky="w", padx=4)
        frame.columnconfigure(1, weight=1); row+=1

    def row_pick_model(label: str, default_path: Path):
        nonlocal row
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        var=tk.StringVar(value=str(default_path))
        ent=_mk_entry(frame, width=80, textvariable=var); ent.grid(row=row, column=1, sticky="we", padx=6, pady=4)
        def browse():
            global CURRENT_MODEL_PATH, W2V_GLOBAL
            p=filedialog.askopenfilename(parent=root, initialdir=str(default_path.parent), title=label,
                                         filetypes=[("Gensim Model","*.model"), ("KeyedVectors","*.kv"),
                                                    ("Word2Vec Bin/Txt","*.bin *.txt *.gz"), ("Alle Dateien","*.*")])
            if p:
                var.set(p); CURRENT_MODEL_PATH = Path(p); W2V_GLOBAL = None
        ttk.Button(frame, text="…", width=3, command=browse).grid(row=row, column=2, sticky="w", padx=4)
        row+=1

    def row_pick_termlist(label: str, default_path: Path):
        nonlocal row
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        var=tk.StringVar(value=str(default_path))
        ent=_mk_entry(frame, width=80, textvariable=var); ent.grid(row=row, column=1, sticky="we", padx=6, pady=4)
        def browse():
            global CURRENT_TERMLIST_PATH
            p=filedialog.askopenfilename(parent=root, initialdir=str(default_path.parent), title=label,
                                         filetypes=[("CSV","*.csv"), ("Alle Dateien","*.*")])
            if p:
                var.set(p); CURRENT_TERMLIST_PATH = Path(p)
        ttk.Button(frame, text="…", width=3, command=browse).grid(row=row, column=2, sticky="w", padx=4)
        row+=1

    # =========================================================================
    # NEU: Kosinus-Matrix-Pfad im DataManager
    # =========================================================================
    
    # Füge Attribut zum DataManager hinzu (falls noch nicht vorhanden)
    if not hasattr(DATA, 'path_cosine'):
        DATA.path_cosine = PROJECT_ROOT / "output" / "cosine" / "cosine_tfidf2000.csv"
        DATA.cosine_df = None  # Cache für geladene Matrix

    # =========================================================================
    # CSV-QUELLEN
    # =========================================================================
    
    row_pick_csv("Korpus:", lambda p:setattr(DATA, "path_corpus", p) or setattr(DATA, "corpus_df", None), DATA.path_corpus)
    row_pick_csv("Document-Term-Matrix:", 
                 lambda p:(setattr(DATA, "path_dtm", p),
                           setattr(DATA, "dtm_df", None),
                           setattr(DATA, "tokens_per_year_df", None)),
                 DATA.path_dtm)
    
    # NEU: Kosinus-Matrix
    row_pick_csv("Kosinus-Matrix:", 
                 lambda p:(setattr(DATA, "path_cosine", p),
                           setattr(DATA, "cosine_df", None)),
                 DATA.path_cosine)
    
    row_pick_csv("Document-Topic-Matrix:", lambda p:setattr(DATA, "path_topics", p) or setattr(DATA, "topics_df", None), DATA.path_topics)
    row_pick_csv("Metadata:", lambda p:setattr(DATA, "path_metadata", p) or setattr(DATA, "metadata_df", None), DATA.path_metadata)
    row_pick_csv("TF-IDF:", lambda p:setattr(DATA, "path_tfidf_for_cloud", p) or setattr(DATA, "tfidf_avg_df", None), DATA.path_tfidf_for_cloud)

    # Modell & Termliste
    row_pick_model("Wort-Vektor-Modell:", CURRENT_MODEL_PATH)
    row_pick_termlist("Termset:", CURRENT_TERMLIST_PATH)

    row+=1
    info = tk.Text(frame, height=12, width=100); info.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)

    # =========================================================================
    # LOAD & CHECK FUNKTION (erweitert)
    # =========================================================================
    
    def load_check():
        info.delete(1.0, tk.END); msgs=[]

        # Korpus
        try:
            DATA.corpus_df=None; dfc=DATA.load_corpus(); yrs=dfc["year_final"].dropna().astype(int)
            msgs.append(f"✅ Korpus: {len(dfc):,} Zeilen | Jahre: {yrs.min() if not yrs.empty else '—'}–{yrs.max() if not yrs.empty else '—'} | Spalten: {len(dfc.columns)}")
        except Exception as e:
            msgs.append(f"❌ Korpus: {e}")

        # DTM (+ Tokens/Jahr)
        try:
            DATA.dtm_df=None; DATA.tokens_per_year_df=None
            dfd=DATA.load_dtm()
            yrs = dfd.get("year_final", pd.Series(dtype=int))
            if not yrs.empty:
                try: yrs=yrs.dropna().astype(int); yrspan=f"{yrs.min()}–{yrs.max()}"
                except Exception: yrspan="—"
            else:
                yrspan="—"
            tok = DATA.tokens_per_year_df
            tok_msg = f"{int(tok['anzahl_tokens'].sum()):,} Tokens gesamt" if isinstance(tok, pd.DataFrame) and "anzahl_tokens" in tok.columns else "—"
            msgs.append(f"✅ Document-Term-Matrix: {len(dfd):,} Zeilen | Jahre: {yrspan} | Spalten: {len(dfd.columns)} | Tokens/Jahr: {tok_msg}")
        except Exception as e:
            msgs.append(f"❌ Document-Term-Matrix: {e}")

        # NEU: Kosinus-Matrix
        try:
            DATA.cosine_df = None
            if not DATA.path_cosine.exists():
                raise FileNotFoundError(f"Datei nicht gefunden: {DATA.path_cosine}")
            
            cos_df = pd.read_csv(DATA.path_cosine, index_col=0)
            DATA.cosine_df = cos_df  # Cache
            
            n_docs = cos_df.shape[0]
            doc_ids_sample = list(cos_df.index[:3]) + (["..."] if n_docs > 3 else [])
            
            # Statistik: Durchschnittliche Ähnlichkeit
            values = cos_df.values
            np.fill_diagonal(values, np.nan)  # Diagonale ausschließen
            avg_sim = float(np.nanmean(values))
            
            msgs.append(f"✅ Kosinus-Matrix: {n_docs:,} × {n_docs:,} Dokumente | ⌀ Ähnlichkeit: {avg_sim:.3f} | IDs: {', '.join(str(x) for x in doc_ids_sample)}")
        except Exception as e:
            msgs.append(f"❌ Kosinus-Matrix: {e}")

        # Topics
        try:
            DATA.topics_df=None; dft=DATA.load_topics(); msgs.append(f"✅ Topics: {dft.shape[0]:,} Docs × {dft.shape[1]:,} Topics")
        except Exception as e:
            msgs.append(f"❌ Topics: {e}")

        # Metadata
        try:
            DATA.metadata_df=None; dfm=DATA.load_metadata(); msgs.append(f"✅ Metadata: {len(dfm):,} Zeilen")
        except Exception as e:
            msgs.append(f"❌ Metadata: {e}")

        # TF-IDF avg
        try:
            DATA.tfidf_avg_df=None; dfa=DATA.load_tfidf_for_cloud(); msgs.append(f"✅ TF-IDF avg: {len(dfa):,} Terme")
        except Exception as e:
            msgs.append(f"❌ TF-IDF avg: {e}")

        # W2V/Embeddings
        try:
            global W2V_GLOBAL
            W2V_GLOBAL = load_w2v_or_kv(CURRENT_MODEL_PATH)
            vocab_size = len(W2V_GLOBAL.key_to_index); dim = W2V_GLOBAL.vector_size
            msgs.append(f"✅ Wort-Vektor-Modell: {vocab_size:,} Vokabeln × {dim} Dimensionen")
        except Exception as e:
            msgs.append(f"❌ Wort-Vektor-Modell: {e}")

        # Termliste
        try:
            df_terms = load_termset_df(CURRENT_TERMLIST_PATH)
            non_empty = int(df_terms.count().sum())
            msgs.append(f"✅ Termliste: {non_empty:,} Einträge in {len(df_terms.columns)} Spalten")
        except Exception as e:
            msgs.append(f"❌ Termliste: {e}")

        info.insert(tk.END, "\n".join(msgs))

    ttk.Button(frame, text="Laden & Prüfen", command=load_check).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")

# =========================
# Haupt-Reiterstruktur
# =========================

def main() -> None:
    root = tk.Tk()
    root.title(f"{SUITE_NAME} – Ausdrücke · Wort-Vektor-Modell · Termset · Topics · Texte")
    install_safe_exit(root)
    bring_front(root)
    install_focus_minimize(root, enable=True)

    root_nb = ttk.Notebook(root)
    root_nb.pack(fill="both", expand=True)

    # Daten (zentraler Loader)
    build_tab_data(root_nb, root)

    # Oberreiter:
    nb_expr = ttk.Notebook(root_nb); root_nb.add(nb_expr, text="Ausdrücke")
    nb_w2v  = ttk.Notebook(root_nb); root_nb.add(nb_w2v,  text="Wort-Vektor-Modell")
    nb_term = ttk.Notebook(root_nb); root_nb.add(nb_term, text="Termset")
    nb_top  = ttk.Notebook(root_nb); root_nb.add(nb_top,  text="Topics")
    nb_texts = ttk.Notebook(root_nb); root_nb.add(nb_texts, text="Texte")

    # Ausdrücke – Untertabs
    build_tab_vocab(nb_expr, root)
    build_tab_tfidf_rank(nb_expr, root)
    build_tab_docfreq(nb_expr, root)
    build_tab_doc_tfidf(nb_expr, root)
    build_tab_concordance(nb_expr, root)
    build_tab_collocations(nb_expr, root)
    build_tab_wordtrends(nb_expr, root)

    # Wort-Vektor-Modell – Untertabs
    build_tab_embeddings(nb_w2v, root)
    build_tab_embed_compare(nb_w2v, root)
    build_tab_network(nb_w2v, root)

    # Termset – Untertabs
    build_tab_termset_cluster(nb_term, root)
    build_tab_termset_wordcloud(nb_term, root)
    build_tab_termset_dendro(nb_term, root)

    # Cluster - Texte
    build_tab_texts_scatter(nb_texts, root)
    
    # Topics – Untertabs
    build_tab_topics(nb_top, root)

    root.mainloop()

if __name__ == "__main__":
    main()