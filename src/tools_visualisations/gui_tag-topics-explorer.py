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
# Matplotlib Style (kleiner, kompakt)
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
    """Nutze constrained_layout wenn verfügbar, sonst tight_layout."""
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
    # Kein always-on-top -> Alt+Tab funktioniert
    win.update_idletasks()
    try: win.attributes("-topmost", False)
    except Exception: pass
    try: win.lift()
    except Exception: pass
    try: win.focus_force()
    except Exception: pass

def install_focus_minimize(root: tk.Tk, enable: bool = True) -> None:
    """
    Minimiere das Fenster nur dann, wenn der Fokusverlust durch einen
    Maus-Klick außerhalb verursacht wurde. Reines Mouse-Leave oder Alt+Tab
    sollen NICHT minimieren.

    Umsetzung: Wir werten im FocusOut-Event das state-Bitfeld aus.
    Wenn während des Fokusverlustes ein Mausbutton gedrückt war,
    sind Button-Masks gesetzt (Button1..3). Zusätzlich stellen wir sicher,
    dass der Fokus wirklich das gesamte Tk verlässt (focus_displayof() ist None).
    """
    if not enable:
        return

    # Bitmasken für Button1..3 in Tk (X11/Win): 0x100, 0x200, 0x400
    BUTTON_MASK = 0x100 | 0x200 | 0x400

    def _on_focus_out(event):
        try:
            # Fokus liegt nicht mehr auf irgendeinem Tk-Widget → Verlassen der App
            focus_inside = (root.focus_displayof() is not None)
            # War beim Fokusverlust eine Maustaste gedrückt?
            st = getattr(event, "state", 0)
            mouse_down = bool(st & BUTTON_MASK)
            if (not focus_inside) and mouse_down:
                root.iconify()
        except Exception:
            # Failsafe: niemals hart crashen
            pass

    # Nur auf dem Root binden, nicht global, damit interne Widget-Wechsel nicht feuern
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
        # Termset↔Topic-Abbildungen (Termset-Topic Visuals)
        self.path_termset: Path = RESOURCES_DIR / "termsets" / "Termset_Begriffe_2.3.csv"
        self.path_topic_words: Path = RESOURCES_DIR / "topic-models" / "topics_v3" / "fadelive_mallet_stop_topic_words_100_words_tag.csv"
        self.path_tfidf: Path = OUTPUT_DIR / "dtm_tfidf_stop" / "tfidf-2000.csv"
        self.path_ranks: Path = OUTPUT_DIR / "processed_termset" / "Termset_Begriffe_2.3" / "Termset_Begriffe_2.3_tag_topic_rank.csv"
        self.path_relevance: Path = OUTPUT_DIR / "processed_termset" / "Termset_Begriffe_2.3" / "Termset_Begriffe_2.3_tag_topic_relevance.csv"
        self.path_counts_per_year: Path = OUTPUT_DIR / "processed_termset" / "Termset_Begriffe_2.3" / "Termset_Begriffe_2.3_dtti_topdocs_topic_counts_per_year.csv"
        self.path_top10_year_value: Path = OUTPUT_DIR / "processed_termset" / "Termset_Begriffe_2.3" / "Termset_Begriffe_2.3_dtti_topdocs_top10_year_value.csv"
        self.path_top10_value_per_text_topic: Path = OUTPUT_DIR / "processed_termset" / "Termset_Begriffe_2.3" / "Termset_Begriffe_2.3_dtti_topdocs_top10_value_per_text_topic.csv"
        # Topic-only Visuals + Verläufe (Initial-Code)
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
# Termset-Topic Visualisierungen (Tag-Topic-Exploration)
# -----------------------------
def build_tab_bubbles_rank_topn(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="Bubbles (Top-N nach Rang)")
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
        ax.set_title(f"Top {len(ordered_labels)} Topics – TF-IDF-Summen (Termset-Overlap)")
        _shrink_axes(ax); _apply_layout(fig); plt.show()

        out_df = df_top.rename(columns={"tfidf_sum": "value"})
        ctx["v"] = f"bubbles_rank_top{len(ordered_labels)}"
        btn_csv.configure(state="normal", command=lambda: ask_save_df(out_df, "Bubbles_rank_top", ctx["v"], root))
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("Bubbles_rank_top", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Berechnen", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)


def build_tab_bubbles_ranked_score(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="Bubbles (Score nach Rang)")
    row = 0
    ttk.Label(frame, text="N Topics (Rang):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_topn = _mk_entry(frame, width=8); ent_topn.insert(0, "30"); ent_topn.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="Bubble-Skalierung:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_scale = _mk_entry(frame, width=8); ent_scale.insert(0, "4"); ent_scale.grid(row=row, column=3, sticky="w", padx=6, pady=4)

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
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root); return

        r = (df_ranks.assign(Topic=df_ranks["Topic"].astype(str),
                             rank=pd.to_numeric(df_ranks["TFIDF-Positions-Rang"], errors="coerce"))
                      .dropna(subset=["rank"]).sort_values("rank"))

        try:
            n = max(1, int((ent_topn.get() or "30").strip()))
            scale = float((ent_scale.get() or "4").strip())
        except Exception:
            messagebox.showerror("Fehler", "Ungültige Parameter (N / Skalierung).", parent=root); return

        topics_available = set(df_topics.index.astype(str))
        top_topics_ordered = [t for t in r["Topic"].tolist() if t in topics_available][:n]
        if not top_topics_ordered:
            messagebox.showerror("Fehler", "Keine passenden Topics aus der Rangliste gefunden.", parent=root); return

        tag_dict = tag_dict_from_df(df_tags)
        topic_map = topic_word_map_from_df(df_topics)
        tfidf_dict = tfidf_series_sum(DATA).to_dict()
        tfidf_index = set(tfidf_dict.keys())

        rows = []
        for topic in top_topics_ordered:
            twords = topic_map[topic]
            tset = set(twords)
            pos = {w: i for i, w in enumerate(twords, start=1)}  # 1-basiert
            for tag, expr in tag_dict.items():
                common = (tset & set(expr)) & tfidf_index
                score = 0.0
                for w in common:
                    tv = tfidf_dict.get(w, 0.0)
                    p = pos.get(w, 0)
                    if tv > 0 and p > 0:
                        score += tv / math.log(p + 1)
                rows.append({"Topic": topic, "Tag": tag, "tag_topic_score": score})

        df_top = pd.DataFrame(rows)
        df_top["Topic_Label"] = df_top["Topic"].astype(str).str.split("(", n=1).str[0].str.strip()
        df_top["rank"] = df_top["Topic"].map(dict(zip(r["Topic"], r["rank"])))
        df_top = df_top.sort_values(["rank","Tag"])
        ordered_labels = topic_labels_ordered_from_rank(r, top_topics_ordered)
        df_top["Topic_Label"] = pd.Categorical(df_top["Topic_Label"], categories=ordered_labels, ordered=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        base = df_top["tag_topic_score"].clip(lower=0).to_numpy()
        sizes = (base * scale) + 10.0
        ax.scatter(df_top["Topic_Label"], df_top["Tag"], s=sizes, color="sienna", alpha=0.6, edgecolors="black")
        ax.set_xlabel("Topic (TFIDF-Positions-Rang)")
        ax.set_ylabel("Tag")
        ax.set_title(f"Top {len(ordered_labels)} Topics – Bubblegröße = Tag-Topic-Score")
        _shrink_axes(ax); _apply_layout(fig); plt.show()

        out_df = df_top.rename(columns={"tag_topic_score": "value"})
        ctx["v"] = f"bubbles_ranked_score_top{len(ordered_labels)}"
        btn_csv.configure(state="normal", command=lambda: ask_save_df(out_df, "Bubbles_ranked_score", ctx["v"], root))
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("Bubbles_ranked_score", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Berechnen", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)


def build_tab_tag_relevance(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="Tag-Relevanz")
    row = 0
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled"); btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    ctx = {"v": "tag_relevance"}

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
        plt.title("Tag-Topic-Verhältnisse vs. TF-IDF-Summen", fontsize=12)
        _apply_layout(fig); plt.show()

        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("Tag_Relevanz", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)


def build_tab_topics_year_stacked(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="Topics/Jahr (Stacked)")
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

        topics_in_df = df.columns.astype(str)
        r = (df_ranks.assign(Topic=df_ranks["Topic"].astype(str),
                             rank=pd.to_numeric(df_ranks["TFIDF-Positions-Rang"], errors="coerce"))
                      .dropna(subset=["rank"]).sort_values("rank"))
        try:
            topn = max(1, int((ent_topn.get() or "10").strip()))
        except Exception:
            topn = 10

        ranked_topics = [t for t in r["Topic"].tolist() if t in topics_in_df][:topn]
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
        ax.set_xlabel("Jahr"); ax.set_ylabel("Anzahl der Top-50-Texte pro Topic")
        ax.set_title("Anzahl der Top-50-Texte pro Topic pro Jahr (Stacked; Top-N nach Rang)", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        labels = [lab.replace("(", "\n(") for lab in labels]
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 0.5), loc="center left", title="Topics", fontsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"topics_year_stacked_{ymin}_{ymax}_top{topn}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("Topics_Year_Stacked", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)
    
def build_tab_topics_year_poly(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="Topics/Jahr (Poly)")
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

        topics_in_df = df.columns.astype(str)
        r = df_ranks.assign(
            Topic=df_ranks["Topic"].astype(str),
            rank=pd.to_numeric(df_ranks["TFIDF-Positions-Rang"], errors="coerce")
        ).dropna(subset=["rank"]).sort_values("rank")
        try:
            n = max(1, int((ent_topn.get() or "10").strip()))
        except Exception:
            n = 10
        ranked_topics = [t for t in r["Topic"].tolist() if t in topics_in_df][:n]
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

        ax.set_xticks([jahr for jahr in x.astype(int) if jahr % 10 == 0])
        ax.set_xlabel("Jahr"); ax.set_ylabel("Anzahl der Top-50-Texte pro Topic")
        ax.set_title(f"Pro Jahr (Polynom Grad {deg}) – Top-N nach TFIDF-Positions-Rang", fontsize=12)
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", title="Topics", fontsize=8)
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"topics_year_poly_deg{deg}_top{n}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("Topics_Year_Poly", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

def build_tab_series_top10(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="Top10 Zeitreihe")
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
            messagebox.showinfo("Info", "Keine numerischen Werte für Top10-Zeitreihe.", parent=root); return
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
        ax.set_title("Top-10-Text-Werte pro Jahr", fontsize=12)
        ax.set_xlabel("Jahr"); ax.set_ylabel("Normalisierter Wert")
        ax.set_xticks(np.arange(ymin, ymax+1, 10))
        ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"top10_series_{ymin}_{ymax}_deg{deg}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("Top10_Series", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

def build_tab_compare_tokens_topics(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="Vergleich (Tokens/Topics)")
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
            messagebox.showerror("Fehler", "Spalte 'Wert' in den *_year_value-Dateien nicht gefunden.", parent=root); return

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
        ax.plot(begriffe_df["Jahr"], begriffe_df["norm"], label="Top-50-Topic-Texte Termset (norm.)", linestyle="-", linewidth=1.5)
        ax.plot(tokens_df["year"], tokens_df["norm"], label="Token-Anzahl (norm.)", linestyle="--", linewidth=1.2)
        ax.plot(global_df["Jahr"], global_df["norm"], label="Top-50-Topic-Texte global (norm.)", linestyle="--", linewidth=1.2)
        ax.set_title("Gleitende Mittelwerte: Tokens vs. Top-50-Topic-Texte (global/Termset)", fontsize=12)
        ax.set_xlabel("Jahr"); ax.set_ylabel("Normalisierter Wert (0–1)")
        ax.set_xticks(np.arange(ymin, ymax+1, 10))
        ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"compare_tokens_topics_{ymin}_{ymax}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("Compare_Tokens_Topics", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

def build_tab_top10_value_per_text_topic(nb: ttk.Notebook, root: tk.Tk) -> None:
    """NEU: Anzeige der Datei dtti_topdocs_top10_value_per_text_topic.csv als (text, rank)."""
    frame = ttk.Frame(nb); nb.add(frame, text="Top10 Value pro Text/Topic")
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

        df2 = df.copy()
        text_col = df2.columns[0]
        value_col = df2.columns[1]
        vals = ensure_numeric(df2[value_col]).fillna(0.0)
        # Rang 1 = höchster Wert
        df2["rank"] = (-vals).rank(method="min").astype(int)
        df2 = df2.rename(columns={text_col: "text"})[["text", "rank"]].sort_values("rank", ascending=True)

        tree.delete(*tree.get_children())
        for _, r in df2.iterrows():
            tree.insert("", "end", values=(str(r["text"]), int(r["rank"])))

        out_df = df2
        btn_csv.configure(state="normal", command=lambda: ask_save_df(out_df, "Top10_Value_per_Text_Topic", "text_rank", root))

    ttk.Button(frame, text="Laden/anzeigen", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

# -----------------------------
# Topic-Visualisierungen (Topic-Exploration) – inkl. Initial-Verläufe
# -----------------------------
def build_tab_global_series(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="Global: Topdocs/Jahr (Serie)")
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
        ax.set_title("Global: Top-50-Topic-Texte pro Jahr", fontsize=12)
        ax.set_xlabel("Jahr"); ax.set_ylabel("Normalisierter Wert")
        ax.set_xticks(np.arange(ymin, ymax+1, 10))
        ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"global_series_{ymin}_{ymax}_deg{deg}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("Global_Series", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

def build_tab_global_compare_tokens(nb: ttk.Notebook, root: tk.Tk) -> None:
    frame = ttk.Frame(nb); nb.add(frame, text="Global: Tokens vs. Topdocs")
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

        if "year" not in tokens_df.columns and "Jahr" in tokens_df.columns:
            tokens_df = tokens_df.rename(columns={"Jahr":"year"})
        if "anzahl_tokens" not in tokens_df.columns:
            for c in ["tokens","count_tokens","anzahl"]:
                if c in tokens_df.columns:
                    tokens_df = tokens_df.rename(columns={c:"anzahl_tokens"}); break

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
        ax.plot(tokens_df["year"], tokens_df["norm"], label="Token-Anzahl (norm.)", linestyle="--", linewidth=1.2)
        ax.plot(global_df["Jahr"], global_df["norm"], label="Top-50-Topic-Texte global (norm.)", linestyle="-", linewidth=1.5)
        ax.set_title("Tokens vs. Global Topdocs (normiert)", fontsize=12)
        ax.set_xlabel("Jahr"); ax.set_ylabel("Normalisierter Wert (0–1)")
        ax.set_xticks(np.arange(ymin, ymax+1, 10))
        ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8)
        _apply_layout(fig); plt.show()

        ctx["v"] = f"global_compare_tokens_{ymin}_{ymax}"
        btn_png.configure(state="normal", command=lambda: ask_save_current_figure("Global_Compare_Tokens", ctx["v"], root, fig=fig))

    ttk.Button(frame, text="Plotten", command=run).grid(row=row, column=0, sticky="w", padx=6, pady=6)

def build_tab_topic_trends_from_distribution(nb: ttk.Notebook, root: tk.Tk) -> None:
    """Integriert die Verlaufsfunktion aus deinem Initial-Script (Topics+Metadata)."""
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
    row_pick("Topic-Wörter:",   lambda: DATA.path_topic_words, DATA.set_topic_words)
    row_pick("TF-IDF:",         lambda: DATA.path_tfidf, DATA.set_tfidf)
    row_pick("Ranking:",        lambda: DATA.path_ranks, DATA.set_ranks)
    row_pick("Relevanz:",       lambda: DATA.path_relevance, DATA.set_relevance)
    row_pick("Counts/Jahr:",    lambda: DATA.path_counts_per_year, DATA.set_counts_per_year)
    row_pick("Top10 Year Values:", lambda: DATA.path_top10_year_value, DATA.set_top10_year_value)
    row_pick("Top10 Value per Text/Topic:", lambda: DATA.path_top10_value_per_text_topic, DATA.set_top10_value_per_text_topic)
    # Topic-only
    row_pick("Tokens/Jahr:",    lambda: DATA.path_tokens_year, DATA.set_tokens_year)
    row_pick("Topdocs/Jahr (global):", lambda: DATA.path_global_topdocs_year, DATA.set_global_topdocs_year)
    row_pick("Topics-Distribution:", lambda: DATA.path_topics, DATA.set_topics)
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
            _ok("Termset", DATA.load_termset),
            _ok("Topic-Wörter", DATA.load_topic_words),
            _ok("TF-IDF", DATA.load_tfidf),
            _ok("Ranking", DATA.load_ranks),
            _ok("Relevanz", DATA.load_relevance),
            _ok("Counts per Year", DATA.load_counts_per_year),
            _ok("Top10 Year Values", DATA.load_top10_year_value),
            _ok("Top10 Value per Text/Topic", DATA.load_top10_value_per_text_topic),
            _ok("Tokens/Jahr", DATA.load_tokens_year),
            _ok("Topdocs/Jahr (global)", DATA.load_global_topdocs_year),
            _ok("Topics-Distribution", DATA.load_topics_dist),
            _ok("Metadata", DATA.load_metadata),
        ]
        info.insert(tk.END, "\n".join(msgs))

    ttk.Button(frame, text="Laden & Prüfen", command=load_check).grid(row=row+1, column=0, padx=6, pady=6, sticky="w")

# -----------------------------
# Main – zwei Hauptreiter
# -----------------------------
def main() -> None:
    root = tk.Tk()
    root.title("Topic & Termset–Topic Visualisierungssuite")
    install_safe_exit(root)
    bring_front(root)
    install_focus_minimize(root, enable=True)  # bei Bedarf auf False setzen

    nb_root = ttk.Notebook(root)
    nb_root.pack(fill="both", expand=True)

    # Daten
    build_tab_data(nb_root, root)

    # Hauptreiter 1: Topic-Exploration (nur Topic-/Global-/Tokens-Dateien)
    nb_topics = ttk.Notebook(nb_root); nb_root.add(nb_topics, text="Topic-Exploration")
    build_tab_global_series(nb_topics, root)
    build_tab_global_compare_tokens(nb_topics, root)
    build_tab_topic_trends_from_distribution(nb_topics, root)  # Initial-Verlaufsfunktion integriert

    # Hauptreiter 2: Tag-Topic-Exploration (Mapping)
    nb_t2 = ttk.Notebook(nb_root); nb_root.add(nb_t2, text="Tag-Topic-Exploration")
    build_tab_bubbles_rank_topn(nb_t2, root)
    build_tab_bubbles_ranked_score(nb_t2, root)
    build_tab_tag_relevance(nb_t2, root)
    build_tab_topics_year_stacked(nb_t2, root)
    build_tab_topics_year_poly(nb_t2, root)
    build_tab_series_top10(nb_t2, root)
    build_tab_compare_tokens_topics(nb_t2, root)
    build_tab_top10_value_per_text_topic(nb_t2, root)  # NEU: Text/Rank-Tabelle

    root.mainloop()

if __name__ == "__main__":
    main()
