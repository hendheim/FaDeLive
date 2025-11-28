# file: src/tools_visualisations/export_statistics_plots.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # nur speichern, nicht anzeigen
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm


# =========================
# Projektpfade (wie zuvor)
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

INPUT_DIR = PROJECT_ROOT / "output" / "statistics"
VIS_DIR   = PROJECT_ROOT / "output" / "exploration" / "statistics"
VIS_DIR.mkdir(parents=True, exist_ok=True)


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
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "figure.figsize": (12, 6),
    "savefig.bbox": "tight",
})

BALKENFARBE = "sienna"


# =========================
# Helpers
# =========================

def _exists(p: Path) -> bool:
    if not p.exists():
        print(f"⚠️ Datei nicht gefunden: {p.relative_to(PROJECT_ROOT)}")
        return False
    return True

def _save_fig(name: str) -> Path:
    safe = re.sub(r"[^\w\-.]+", "_", name).strip("_")
    out = VIS_DIR / f"{safe}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ gespeicherte Grafik: {out.relative_to(PROJECT_ROOT)}")
    return out

def _read_csv(filename: str) -> Optional[pd.DataFrame]:
    p = INPUT_DIR / filename
    if not _exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"❌ Fehler beim Lesen von {filename}: {e}")
        return None


# =========================
# Plots nach Dateien
# =========================

def plot_author_statistics() -> None:
    df = _read_csv("author_statistics.csv")
    if df is None: return
    needed = {"author", "percentage", "anzahl_texte"}
    if not needed.issubset(df.columns):
        print("⚠️ Spalten fehlen für author_statistics.csv")
        return
    df = df[df["anzahl_texte"] >= 3].sort_values("percentage")
    plt.figure(figsize=(12, 6))
    plt.bar(df["author"].astype(str), df["percentage"], color=BALKENFARBE)
    plt.title("Texte pro Autor")
    plt.xlabel("Autor*in")
    plt.ylabel("Prozent (%)")
    plt.xticks(rotation=90)
    _save_fig("author_statistics")

def plot_tokens_relative_to_content() -> None:
    df = _read_csv("tokens.csv")
    if df is None: return
    if not {"count", "field"}.issubset(df.columns):
        print("⚠️ Spalte 'count' oder 'field' nicht gefunden (tokens.csv).")
        return
    row = df.loc[df["field"] == "content", "count"]
    if row.empty:
        print("⚠️ Kein Eintrag mit field == 'content' (tokens.csv).")
        return
    ref = float(row.values[0])
    if ref == 0:
        print("⚠️ Referenzwert 'content' = 0 (tokens.csv).")
        return
    df = df.assign(relative_percentage=(df["count"] / ref * 100.0)).sort_values("relative_percentage")
    plt.figure(figsize=(6, 6))
    plt.bar(df["field"].astype(str), df["relative_percentage"], color=BALKENFARBE)
    plt.title("Token-Anzahl relativ zu 'content'")
    plt.xlabel("Feld")
    plt.ylabel("Anteil im Vergleich zu 'content' (%)")
    plt.xticks(rotation=90)
    _save_fig("tokens_relative_to_content")

def plot_tokens_per_textclass() -> None:
    df = _read_csv("tokens_per_textclass.csv")
    if df is None: return
    if not {"textclass", "percentage"}.issubset(df.columns):
        print("⚠️ 'textclass' oder 'percentage' fehlen (tokens_per_textclass.csv).")
        return
    df = df.dropna(subset=["textclass", "percentage"]).sort_values("percentage")
    # Plot 1: Tokens pro Textklasse
    plt.figure(figsize=(6, 6))
    plt.bar(df["textclass"].astype(str), df["percentage"], color=BALKENFARBE)
    plt.title("Tokens pro Textklasse")
    plt.xlabel("Textklasse")
    plt.ylabel("Prozent (%)")
    plt.xticks(rotation=90)
    _save_fig("tokens_per_textclass_tokens")
    # Plot 2: Anteil der Tokenmenge pro Textklasse
    plt.figure(figsize=(6, 4))
    plt.bar(df["textclass"].astype(str), df["percentage"], color=BALKENFARBE)
    plt.title("Anteil der Tokenmenge pro Textklasse")
    plt.xlabel("Textklasse")
    plt.ylabel("Prozent (%)")
    plt.xticks(rotation=90)
    _save_fig("tokens_per_textclass_share")

def plot_textclass_count() -> None:
    df = _read_csv("textclass_count.csv")
    if df is None: return
    if not {"textclass", "percentage"}.issubset(df.columns):
        print("⚠️ 'textclass' oder 'percentage' fehlen (textclass_count.csv).")
        return
    df["textclass"] = df["textclass"].fillna("Unbekannt").astype(str)
    df = df.sort_values("percentage")
    plt.figure(figsize=(6, 4))
    plt.bar(df["textclass"], df["percentage"], color=BALKENFARBE)
    plt.title("Anteil der Textmenge am Gesamtkorpus pro Textklasse")
    plt.xlabel("Textklasse")
    plt.ylabel("Prozent (%)")
    plt.xticks(rotation=90)
    _save_fig("textclass_count_share")

def plot_address_share() -> None:
    df = _read_csv("address.csv")
    if df is None: return
    if not {"address", "percentage"}.issubset(df.columns):
        print("⚠️ 'address' oder 'percentage' fehlen (address.csv).")
        return
    df = df.dropna(subset=["address"]).assign(address=lambda d: d["address"].astype(str))
    df = df.sort_values("percentage")
    plt.figure(figsize=(12, 8))
    plt.bar(df["address"], df["percentage"], color=BALKENFARBE)
    plt.title("Anteil am Gesamtkorpus pro Veröffentlichungsort")
    plt.xlabel("Veröffentlichungsort")
    plt.ylabel("Prozent (%)")
    plt.xticks(rotation=90)
    _save_fig("address_share")

def plot_author_address_share() -> None:
    df = _read_csv("author_address.csv")
    if df is None: return
    if not {"address_author", "percentage"}.issubset(df.columns):
        print("⚠️ 'address_author' oder 'percentage' fehlen (author_address.csv).")
        return
    df["address_author"] = df["address_author"].fillna("Unbekannt").astype(str)
    df = df[df["address_author"] != "Unbekannt"].sort_values("percentage")
    plt.figure(figsize=(12, 6))
    plt.bar(df["address_author"], df["percentage"], color=BALKENFARBE)
    plt.title("Verteilung der Wirkungsorte der Autoren")
    plt.xlabel("Wirkungsort")
    plt.ylabel("Prozent (%)")
    plt.xticks(rotation=90)
    _save_fig("author_address_share")

def plot_source_share() -> None:
    df = _read_csv("source.csv")
    if df is None: return
    if not {"source", "percentage"}.issubset(df.columns):
        print("⚠️ 'source' oder 'percentage' fehlen (source.csv).")
        return
    df = df.dropna(subset=["source"])
    df["source"] = df["source"].astype(str).apply(lambda x: " ".join(x.split()[:5]))
    df = df.sort_values("percentage")
    plt.figure(figsize=(12, 9))
    plt.bar(df["source"], df["percentage"], color=BALKENFARBE)
    plt.title("Verteilung nach Quelle")
    plt.xlabel("Quelle")
    plt.ylabel("Prozent (%)")
    plt.xticks(rotation=90)
    _save_fig("source_share")

def plot_genre_per_source_grouped() -> None:
    df = _read_csv("genre_per_source.csv")
    if df is None: return
    if not {"genre", "source", "percentage"}.issubset(df.columns):
        print("⚠️ Spalten fehlen (genre_per_source.csv).")
        return
    df = df.copy()
    df["source"] = df["source"].astype(str).apply(lambda x: " ".join(x.split()[:5]))
    genres = sorted(df["genre"].astype(str).unique())
    cmap = cm.get_cmap("tab20", len(genres))
    colors = {g: cmap(i) for i, g in enumerate(genres)}
    plt.figure(figsize=(12, 9))
    for g, grp in df.groupby("genre"):
        plt.bar(grp["source"], grp["percentage"], label=str(g), color=colors[str(g)])
    plt.legend(title="Genre", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.title("Genres pro Quelle")
    plt.xlabel("Quelle")
    plt.ylabel("Prozent (%)")
    plt.xticks(rotation=90)
    _save_fig("genre_per_source_grouped")

def plot_tokens_per_author_filtered() -> None:
    df = _read_csv("tokens_per_author.csv")
    if df is None: return
    needed = {"author_surname", "percentage", "token_count"}
    if not needed.issubset(df.columns):
        print("⚠️ Spalten fehlen (tokens_per_author.csv).")
        return
    df = df[df["token_count"] >= 16000].sort_values("percentage")
    plt.figure(figsize=(12, 9))
    plt.bar(df["author_surname"].astype(str), df["percentage"], color=BALKENFARBE)
    # Titel wie in der Vorlage belassen
    plt.title("Anzahl der Texte pro Autor")
    plt.xlabel("Autor*in")
    plt.ylabel("Prozent (%)")
    plt.xticks(rotation=90)
    _save_fig("tokens_per_author_filtered")

def plot_year_count_tokens_range() -> None:
    # 1782–1891 aus Vorlage
    p = INPUT_DIR / "year_count_tokens.csv"
    if not _exists(p): return
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"❌ Fehler beim Lesen von year_count_tokens.csv: {e}")
        return
    if not {"year", "anzahl_tokens"}.issubset(df.columns):
        print("⚠️ 'year' oder 'anzahl_tokens' fehlen (year_count_tokens.csv).")
        return
    # Filter Bereich
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["anzahl_tokens"] = pd.to_numeric(df["anzahl_tokens"], errors="coerce")
    df = df.dropna(subset=["year", "anzahl_tokens"])
    df = df[(df["year"] >= 1782) & (df["year"] <= 1891)].sort_values("year")
    plt.figure(figsize=(14, 6))
    plt.bar(df["year"].astype(int), df["anzahl_tokens"].astype(float), color=BALKENFARBE)
    plt.xlabel("Jahr")
    plt.ylabel("Anzahl Tokens")
    plt.title("Tokens pro Jahr (1782–1891)")
    plt.xticks(rotation=90)
    plt.grid(True, axis="y")
    _save_fig("year_count_tokens_1782_1891")

# =========================
# Normalverteilungen (generisch)
# =========================

def plot_normals_for_all_csvs() -> None:
    csvs = list(INPUT_DIR.glob("*.csv"))
    if not csvs:
        print("⚠️ Keine CSV-Dateien gefunden in", INPUT_DIR.relative_to(PROJECT_ROOT))
        return
    for pfad in csvs:
        try:
            df = pd.read_csv(pfad)
        except Exception:
            print(f"⏭️ Überspringe (kein CSV lesbar): {pfad.name}")
            continue

        mu_col = next((c for c in df.columns if "Mittelwert" in str(c)), None)
        sd_col = next((c for c in df.columns if "Standardabweichung" in str(c)), None)
        if not mu_col or not sd_col:
            print(f"⏭️ Statistikspalten fehlen in {pfad.name}")
            continue

        try:
            mu = float(pd.to_numeric(df[mu_col], errors="coerce").dropna().iloc[-1])
            sigma = float(pd.to_numeric(df[sd_col], errors="coerce").dropna().iloc[-1])
        except Exception:
            print(f"⏭️ Konnte μ/σ nicht extrahieren: {pfad.name}")
            continue

        if sigma <= 0:
            print(f"⏭️ σ ≤ 0 in {pfad.name} → keine Verteilung")
            continue

        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
        y = norm.pdf(x, loc=mu, scale=sigma)

        plt.figure(figsize=(8, 4))
        plt.plot(x, y, color="blue", label=f"μ = {mu:.2f}, σ = {sigma:.2f}")
        plt.title(f"Normalverteilung – {pfad.stem}")
        plt.xlabel("Wert")
        plt.ylabel("Wahrscheinlichkeitsdichte")
        plt.grid(True)
        plt.legend()
        _save_fig(f"{pfad.stem}_normalverteilung")


# =========================
# Main
# =========================

def main() -> None:
    print(f"📁 Eingangsdaten: {INPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"💾 Ausgaben (PNG): {VIS_DIR.relative_to(PROJECT_ROOT)}")

    plot_author_statistics()
    plot_tokens_relative_to_content()
    plot_tokens_per_textclass()
    plot_textclass_count()
    plot_address_share()
    plot_author_address_share()
    plot_source_share()
    plot_genre_per_source_grouped()
    plot_tokens_per_author_filtered()
    plot_year_count_tokens_range()
    plot_normals_for_all_csvs()

if __name__ == "__main__":
    main()
