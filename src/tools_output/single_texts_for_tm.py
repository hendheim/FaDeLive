from pathlib import Path
import pandas as pd
import os


# =========================================
# Projektpfad automatisch bestimmen
# =========================================

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


# =========================================
# Input / Output basierend auf Projektstruktur
# =========================================

# Metadaten-Quelle
METADATA_PATH = PROJECT_ROOT / "output" / "processed_corpus" / "korpus_stop.csv"

# Haupt-Outputordner
OUT_BASE = PROJECT_ROOT / "output" / "processed_corpus" / "txt" / "intervalle"

# Textfeld und Jahrfelder
COL_ID = "_id"
COL_TEXT = "content"
COL_YEAR = "year"
COL_YEAR_FIRST = "year_first"

# Intervalldefinitionen
INTERVALS = [
    (1782, 1852),
    (1853, 1864),
    (1865, 1876),
    (1877, 1891),
]

# =========================================
# Hilfsfunktion: Bestes Jahr bestimmen
# =========================================

def bestimme_jahr(row):
    """Nutze year_first, falls vorhanden, sonst year."""
    if pd.notna(row.get(COL_YEAR_FIRST)):
        return row.get(COL_YEAR_FIRST)
    return row.get(COL_YEAR)


# =========================================
# Daten laden
# =========================================

def lade_metadaten():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadaten-Datei fehlt:\n{METADATA_PATH}")

    print(f"Lade Metadaten aus:\n{METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH, sep=";", dtype=str)

    # Jahrfelder konvertieren
    for col in [COL_YEAR, COL_YEAR_FIRST]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Jahr bestimmen
    df["year_final"] = df.apply(bestimme_jahr, axis=1)
    df["year_final"] = pd.to_numeric(df["year_final"], errors="coerce")

    # Nur gültige Jahre behalten
    df = df.dropna(subset=["year_final"])
    df["year_final"] = df["year_final"].astype(int)

    # Leere Texte entfernen
    df = df[df[COL_TEXT].notna() & df[COL_TEXT].str.strip().astype(bool)]

    print(f"{len(df)} gültige Dokumente geladen.")
    return df


# =========================================
# Ordner erzeugen
# =========================================

def erzeuge_ausgabeordner():
    for start, end in INTERVALS:
        out_dir = OUT_BASE / f"stop_{start}-{end}"
        out_dir.mkdir(parents=True, exist_ok=True)


# =========================================
# Dokumente exportieren
# =========================================

def exportiere_dokumente(df):
    print("Speichere Dokumente …")

    for _, row in df.iterrows():
        jahr = row["year_final"]
        text = str(row[COL_TEXT]).strip()
        doc_id = str(row[COL_ID])

        # passendes Intervall finden
        for start, end in INTERVALS:
            if start <= jahr <= end:
                out_dir = OUT_BASE / f"stop_{start}-{end}"
                filepath = out_dir / f"{doc_id}.txt"

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)

                break

    print("\n✔ Fertig! Alle Dokumente wurden den Intervallen zugewiesen.")
    print(f"→ Ausgabeordner:\n{OUT_BASE}")


# =========================================
# Hauptfunktion
# =========================================

if __name__ == "__main__":
    df = lade_metadaten()
    erzeuge_ausgabeordner()
    exportiere_dokumente(df)
