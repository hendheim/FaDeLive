#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trainiert Word2Vec-Modelle auf Basis eines oder mehrerer vorverarbeiteter
Korpora 'korpus_gen*.csv' (Gensim-Preprocessing, stopwortbereinigt, lemmatisiert).

ÄNDERUNG v5 (Portable Modelle):
- FIX: Speichert nur KeyedVectors statt vollständiges Modell → portabel zwischen NumPy/Gensim-Versionen
- FIX: Fester seed-Parameter für Reproduzierbarkeit
- NEU: Speichert Versions-Informationen und Trainingsparameter als JSON
- NEU: Warnung über Kompatibilitätsprobleme bei vollständiger Modell-Speicherung

ÄNDERUNG v4 (Vollständige Flexibilisierung):
- Automatische Content-Spalten-Erkennung (content_gen > content_stop > content_lem > content_min > content)
- Flexible Metadaten-Handhabung: Alle nicht-Content-Spalten werden automatisch erkannt
- Automatische Jahr-Spalten-Erkennung (year_first, year, Jahr_final, etc.)
- Automatische ID-Spalten-Erkennung (doc_id, _id, id, filename)
- Automatische Intervall-Erkennung aus Dateinamen
- Adaptive Parameter: Passt Word2Vec-Parameter automatisch an Korpusgröße an
- Unterschiedliche Parameter für Gesamtkorpus vs. Intervalle

Erwartetes Eingabeformat:
    - CSV-Dateien mit mindestens einer Content-Spalte
    - Gesamtkorpus: korpus_gen.csv
    - Intervalle: korpus_gen_1784-1796.csv, korpus_gen_1797-1810.csv, etc.

Für jede Eingabedatei wird:
    1) die Content-Spalte automatisch erkannt,
    2) die Korpusgröße ermittelt,
    3) Parameter automatisch angepasst,
    4) der Inhalt in Sätze und Tokens zerlegt (NLTK, Deutsch),
    5) ein Word2Vec-Modell trainiert,
    6) PORTABEL gespeichert: Nur KeyedVectors (.wordvectors) + Metadaten (.json)

Adaptive Parameter (basierend auf Token-Anzahl):
    - Klein (<100k Tokens): vector_size=100, window=3, min_count=3, negative=3, epochs=30
    - Mittel (100k-1M): vector_size=150, window=5, min_count=8, negative=5, epochs=20
    - Groß (>1M): vector_size=175, window=10, min_count=8, negative=10, epochs=20

Beispielaufruf:

    python s07_word_vector_model_v5.py \
        --input-dir output/processed_corpus \
        --pattern "korpus_gen*.csv" \
        --output-dir output/word2vec_models
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import pandas as pd
from gensim.models import Word2Vec

try:
    import gensim
    import numpy as np
    GENSIM_VERSION = gensim.__version__
    NUMPY_VERSION = np.__version__
except ImportError:
    GENSIM_VERSION = "unknown"
    NUMPY_VERSION = "unknown"


# =============================================================================
# FLEXIBLE METADATEN-ERKENNUNG
# =============================================================================

KNOWN_METADATA_NAMES = {
    "_id", "id", "doc_id", "filename",
    "author", "author_prename", "author_surname", "author_surname_norm", "author_address", "author_address_geo",
    "editor_prename", "editor_surname",
    "title", "title_norm", "title_addition",
    "source", "journal", "magazine",
    "year", "year_first", "year_final", "Jahr_final",
    "volume", "edition", "issue", "pages", "pages_exzerpt",
    "textclass", "genre", "address", "address_geo",
    "lang", "language", "note", "archive",
    "female_education",
}

def identify_content_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identifiziert die Content-Spalte flexibel.
    
    Priorität: content_gen > content_stop > content_lem > content_min > content > text > clean_text
    """
    candidates = [
        "content_gen", "content_stop", "content_lem", "content_min",
        "content", "text", "clean_text"
    ]
    lower_map = {str(c).lower(): c for c in df.columns}
    
    for cand in candidates:
        if cand in df.columns:
            return cand
        lc = str(cand).lower()
        if lc in lower_map:
            return lower_map[lc]
    
    return None


def identify_doc_id_column(df: pd.DataFrame) -> Optional[str]:
    """Identifiziert die Dokument-ID-Spalte flexibel."""
    candidates = ["doc_id", "_id", "id", "filename"]
    lower_map = {str(c).lower(): c for c in df.columns}
    
    for cand in candidates:
        if cand in df.columns and df[cand].notna().any():
            return cand
        lc = str(cand).lower()
        if lc in lower_map and df[lower_map[lc]].notna().any():
            return lower_map[lc]
    
    return None


def identify_year_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Identifiziert Jahr-Spalten flexibel.
    
    Returns:
        (year_first_column, year_column)
    """
    year_first_candidates = ["year_first", "Jahr_first"]
    year_candidates = ["year", "jahr", "Jahr", "year_final", "Jahr_final"]
    
    lower_map = {str(c).lower(): c for c in df.columns}
    
    year_first = None
    for cand in year_first_candidates:
        if cand in df.columns:
            year_first = cand
            break
        lc = str(cand).lower()
        if lc in lower_map:
            year_first = lower_map[lc]
            break
    
    year = None
    for cand in year_candidates:
        if cand in df.columns:
            year = cand
            break
        lc = str(cand).lower()
        if lc in lower_map:
            year = lower_map[lc]
            break
    
    return year_first, year


def is_metadata_column(col_name: str) -> bool:
    """Prüft, ob eine Spalte eine Metadaten-Spalte ist."""
    col_lower = str(col_name).strip().lower()
    if col_name in KNOWN_METADATA_NAMES or col_lower in {n.lower() for n in KNOWN_METADATA_NAMES}:
        return True
    return False


def identify_metadata_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifiziert alle Metadaten-Spalten (= alles außer Content).
    
    Gibt eine Liste aller Spalten zurück, die NICHT die Content-Spalte sind.
    """
    content_col = identify_content_column(df)
    
    metadata_cols = []
    for col in df.columns:
        if content_col and col == content_col:
            continue
        metadata_cols.append(col)
    
    return metadata_cols


# =============================================================================
# Adaptive Parameter-Berechnung
# =============================================================================

def calculate_word2vec_params(num_tokens: int) -> Dict[str, int]:
    """
    Passt Word2Vec-Parameter automatisch an Korpusgröße an.
    
    Kategorien:
    - Klein (<100k Tokens): Konservative Parameter für begrenzte Daten
    - Mittel (100k-1M): Standard-Parameter für Intervalle
    - Groß (>1M): Aggressive Parameter für umfangreiches Korpus
    
    Args:
        num_tokens: Geschätzte Anzahl Tokens im Korpus
    
    Returns:
        Dict mit Word2Vec-Parametern
    """
    
    # Feste Parameter (unabhängig von Größe)
    base_params = {
        'workers': 4,
        'sg': 1,           # Skip-gram
        'hs': 0,           # Negative Sampling
        'sample': 1e-4,
        'seed': 42,        # Fester seed für Reproduzierbarkeit
    }
    
    # Größenabhängige Parameter
    if num_tokens < 100_000:  # Klein (kleine Intervalle)
        size_params = {
            'vector_size': 100,
            'window': 3,
            'min_count': 3,
            'negative': 3,
            'epochs': 30,
        }
        category = "KLEIN"
    elif num_tokens < 1_000_000:  # Mittel (größere Intervalle)
        size_params = {
            'vector_size': 150,
            'window': 5,
            'min_count': 8,
            'negative': 5,
            'epochs': 20,
        }
        category = "MITTEL"
    else:  # Groß (Gesamtkorpus, >1 Mio. Tokens)
        size_params = {
            'vector_size': 175,
            'window': 10,
            'min_count': 13,
            'negative': 10,
            'epochs': 20,
        }
        category = "GROSS"
    
    params = {**base_params, **size_params}
    params['_category'] = category  # Nur für Logging
    
    return params


def estimate_token_count(texts: pd.Series) -> int:
    """
    Schätzt die Token-Anzahl durch Stichprobennahme.
    Zählt Tokens in den ersten 100 Dokumenten und hochrechnet.
    """
    sample_size = min(100, len(texts))
    sample_texts = texts.head(sample_size)
    
    total_tokens = 0
    for text in sample_texts:
        if isinstance(text, str):
            total_tokens += len(text.split())
    
    # Hochrechnung auf Gesamtkorpus
    if sample_size > 0:
        avg_tokens_per_doc = total_tokens / sample_size
        estimated_total = int(avg_tokens_per_doc * len(texts))
    else:
        estimated_total = 0
    
    return estimated_total


# =============================================================================
# Intervall-Erkennung
# =============================================================================

def is_interval_file(filename: str) -> bool:
    """
    Prüft, ob eine Datei ein Intervall-Korpus ist.
    Beispiel: korpus_gen_1784-1796.csv → True
              korpus_gen.csv → False
    """
    # Pattern: korpus_*_YYYY-YYYY.csv (flexibel für verschiedene Präfixe)
    pattern = r'korpus_\w+_\d{4}-\d{4}\.csv$'
    return bool(re.search(pattern, filename))


def categorize_files(files: List[Path]) -> Tuple[Optional[Path], List[Path]]:
    """
    Trennt Gesamtkorpus von Intervall-Dateien.
    
    Returns:
        (gesamtkorpus_file, interval_files)
    """
    gesamtkorpus = None
    intervals = []
    
    for f in files:
        if is_interval_file(f.name):
            intervals.append(f)
        elif f.stem in ["korpus_gen", "korpus_stop", "korpus_lem", "korpus_min"]:
            # Verschiedene Korpus-Typen akzeptieren
            gesamtkorpus = f
        # Andere Dateien ignorieren
    
    return gesamtkorpus, intervals


# =============================================================================
# Hilfsfunktionen
# =============================================================================

def find_input_files(input_dir: Path, pattern: str) -> List[Path]:
    """Sucht alle Dateien im input_dir, die zum Pattern passen (kein rekursives Suchen)."""
    return sorted(input_dir.glob(pattern))


def tokenize_corpus(texts: pd.Series) -> List[List[str]]:
    """
    Nimmt eine Serie von Texten (Strings) und gibt eine Liste von Sätzen,
    wobei jeder Satz eine Tokenliste ist.
    
    KEINE Satzzeichen-Filterung - Tokens werden so verwendet wie von NLTK geliefert.
    """
    sentences = []

    for raw_text in texts.dropna():
        raw_text = str(raw_text).strip()
        if not raw_text:
            continue

        # In Sätze zerlegen (Deutsch)
        try:
            sents = nltk.sent_tokenize(raw_text, language="german")
        except LookupError as e:
            raise RuntimeError(
                "NLTK 'punkt' Tokenizer für Deutsch fehlt.\n"
                "Bitte einmalig ausführen:\n"
                "    import nltk\n"
                '    nltk.download("punkt")'
            ) from e

        # Jeden Satz tokenisieren
        for sent in sents:
            tokens = nltk.word_tokenize(sent, language="german")
            # Leersätze überspringen
            if tokens:
                sentences.append(tokens)

    return sentences


def train_word2vec(sentences: List[List[str]], params: Dict) -> Word2Vec:
    """
    Trainiert ein Word2Vec-Modell mit den gegebenen Parametern.
    
    WICHTIG: Der seed-Parameter ist bereits in params enthalten für Reproduzierbarkeit.
    """
    
    # Entferne _category (nur für Logging)
    training_params = {k: v for k, v in params.items() if k != '_category'}
    
    # Stelle sicher, dass seed gesetzt ist (Fallback)
    if 'seed' not in training_params:
        training_params['seed'] = 42
    
    model = Word2Vec(sentences, **training_params)
    return model


def save_model_portable(
    model: Word2Vec, 
    output_path: Path, 
    params: Dict, 
    metadata: Dict
) -> None:
    """
    Speichert das Modell PORTABEL zwischen verschiedenen NumPy/Gensim-Versionen.
    
    Speichert:
    1. KeyedVectors (.wordvectors) - PORTABEL, enthält nur die Word-Vektoren
    2. Metadaten (.json) - Trainingsparameter, Versionen, Statistiken
    
    Args:
        model: Trainiertes Word2Vec-Modell
        output_path: Basis-Pfad (ohne Extension)
        params: Verwendete Trainingsparameter
        metadata: Zusätzliche Metadaten (Korpus-Info, etc.)
    """
    
    # 1. Speichere KeyedVectors (PORTABEL!)
    vectors_path = output_path.with_suffix('.wordvectors')
    model.wv.save(str(vectors_path))
    print(f"   ✓ KeyedVectors gespeichert (portabel): {vectors_path.name}")
    
    # 2. Speichere Metadaten als JSON
    meta_info = {
        'created_at': datetime.now().isoformat(),
        'gensim_version': GENSIM_VERSION,
        'numpy_version': NUMPY_VERSION,
        'training_params': {k: v for k, v in params.items() if k != '_category'},
        'vocabulary_size': len(model.wv),
        'vector_size': model.wv.vector_size,
        **metadata
    }
    
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Metadaten gespeichert: {json_path.name}")
    
    # 3. Optional: Vollständiges Modell (nur für EXAKT gleiche Umgebung nutzbar)
    # WARNUNG: Nicht portabel zwischen verschiedenen NumPy/Gensim-Versionen!
    full_model_path = output_path.with_suffix('.model')
    model.save(str(full_model_path))
    print(f"   ⚠️  Vollmodell gespeichert (nur für NumPy {NUMPY_VERSION} + Gensim {GENSIM_VERSION}): {full_model_path.name}")


def process_file(
    input_file: Path, 
    output_dir: Path, 
    delimiter: str = ";", 
    file_type: str = "unknown"
) -> None:
    """Lädt eine CSV-Datei, tokenisiert die Texte und trainiert ein Word2Vec-Modell."""
    print(f"\n📄 Verarbeite Datei: {input_file}")
    print(f"   🏷️  Typ: {file_type}")

    if not input_file.exists():
        print(f"⚠️ Datei existiert nicht, übersprungen: {input_file}")
        return

    try:
        df = pd.read_csv(input_file, encoding="utf-8", sep=delimiter)
    except Exception as e:
        print(f"⚠️ Fehler beim Einlesen von {input_file}: {e}")
        return

    # Content-Spalte flexibel erkennen
    content_col = identify_content_column(df)
    
    if content_col is None:
        print(f"⚠️ Keine Content-Spalte gefunden in {input_file}, übersprungen.")
        print(f"   Verfügbare Spalten: {', '.join(df.columns[:10])}")
        return
    
    print(f"   📝 Content-Spalte: {content_col}")

    # Metadaten erkennen (nur zur Info)
    metadata_cols = identify_metadata_columns(df)
    if metadata_cols:
        print(f"   📋 Erkannte Metadaten: {', '.join(metadata_cols[:5])}{'...' if len(metadata_cols) > 5 else ''}")
    
    # ID-Spalte erkennen (nur zur Info)
    id_col = identify_doc_id_column(df)
    if id_col:
        print(f"   🔑 ID-Spalte: {id_col}")
    
    # Jahr-Spalten erkennen (nur zur Info)
    year_first, year = identify_year_columns(df)
    if year_first or year:
        print(f"   📅 Jahr-Spalten: year_first={year_first or '—'}, year={year or '—'}")

    texts = df[content_col].astype(str)
    if texts.str.strip().eq("").all():
        print(f"⚠️ Alle Einträge in '{content_col}' sind leer in {input_file}, übersprungen.")
        return

    # Token-Anzahl schätzen
    estimated_tokens = estimate_token_count(texts)
    print(f"   📊 Geschätzte Token-Anzahl: ~{estimated_tokens:,}")

    # Parameter berechnen
    params = calculate_word2vec_params(estimated_tokens)
    category = params.get('_category', 'UNKNOWN')
    print(f"   ⚙️  Kategorie: {category}")
    print(f"   ⚙️  Parameter: vector_size={params['vector_size']}, window={params['window']}, "
          f"min_count={params['min_count']}, epochs={params['epochs']}, seed={params['seed']}")

    print("   ➜ Tokenisiere Korpus …")
    sentences = tokenize_corpus(texts)

    if not sentences:
        print(f"⚠️ Keine Sätze nach Tokenisierung in {input_file}, übersprungen.")
        return

    print(f"   ➜ Trainiere Word2Vec-Modell ({len(sentences):,} Sätze) …")
    try:
        model = train_word2vec(sentences, params)
    except Exception as e:
        print(f"❌ Fehler beim Training des Modells für {input_file}: {e}")
        return

    # Modell PORTABEL speichern
    output_dir.mkdir(parents=True, exist_ok=True)
    model_base_path = output_dir / input_file.stem
    
    corpus_metadata = {
        'source_file': input_file.name,
        'file_type': file_type,
        'content_column': content_col,
        'num_documents': len(df),
        'num_sentences': len(sentences),
        'estimated_tokens': estimated_tokens,
        'category': category,
    }

    save_model_portable(model, model_base_path, params, corpus_metadata)
    print(f"   📈 Vokabulargröße: {len(model.wv):,} Wörter")


# =============================================================================
# run-Funktion für Pipeline
# =============================================================================

def run(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "korpus_gen*.csv",
    delimiter: str = ";",
) -> None:
    """Trainiert Word2Vec-Modelle auf Basis von korpus_gen-CSV-Dateien."""

    print(f"📂 Eingabeordner: {input_dir}")
    print(f"📂 Ausgabeordner: {output_dir}")
    print(f"🔎 Dateipattern:  {pattern}")
    print(f"📦 Gensim Version: {GENSIM_VERSION}")
    print(f"🔢 NumPy Version:  {NUMPY_VERSION}")

    input_files = find_input_files(input_dir, pattern)
    if not input_files:
        print("⚠️ Keine passenden Eingabedateien gefunden.")
        return

    print(f"✓ {len(input_files)} Datei(en) gefunden.")

    # Dateien kategorisieren
    gesamtkorpus, intervals = categorize_files(input_files)

    # Gesamtkorpus verarbeiten
    if gesamtkorpus:
        print("\n" + "="*60)
        print("🌍 GESAMTKORPUS")
        print("="*60)
        process_file(gesamtkorpus, output_dir, delimiter, file_type="GESAMTKORPUS")
    else:
        print("\n⚠️ Kein Gesamtkorpus gefunden.")

    # Intervalle verarbeiten
    if intervals:
        print("\n" + "="*60)
        print(f"📅 INTERVALLE ({len(intervals)} Dateien)")
        print("="*60)
        for interval_file in intervals:
            process_file(interval_file, output_dir, delimiter, file_type="INTERVALL")
    else:
        print("\n⚠️ Keine Intervall-Dateien gefunden.")

    print("\n" + "="*60)
    print("✅ Alle Modelle wurden verarbeitet.")
    print("="*60)
    print("\n💡 WICHTIG: Zum Laden der Modelle:")
    print("   - Verwende .wordvectors Dateien (portabel zwischen Versionen)")
    print("   - Beispiel: KeyedVectors.load('modell.wordvectors')")
    print(f"   - .model Dateien funktionieren NUR mit NumPy {NUMPY_VERSION} + Gensim {GENSIM_VERSION}")


# =============================================================================
# Argumentparser (akzeptiert optional argv)
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trainiert Word2Vec-Modelle mit adaptiven Parametern auf Basis von korpus_gen-CSV-Dateien.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FLEXIBILISIERUNG v5 (PORTABLE MODELLE):
  - FIX: Speichert KeyedVectors (.wordvectors) - portabel zwischen NumPy/Gensim-Versionen
  - FIX: Fester seed=42 für Reproduzierbarkeit
  - NEU: JSON-Metadaten mit Versionen und Parametern
  - WARNUNG: .model Dateien nur für exakt gleiche Bibliotheksversionen nutzbar

FLEXIBILISIERUNG v4:
  - Automatische Content-Spalten-Erkennung (content_gen > content_stop > ...)
  - Flexible Metadaten-Handhabung (automatische Erkennung aller Nicht-Content-Spalten)
  - Automatische Jahr-Spalten-Erkennung (year_first, year, Jahr_final, etc.)
  - Automatische ID-Spalten-Erkennung (doc_id, _id, id, filename)

Parameter-Kategorien (automatisch):
  - KLEIN   (<100k Tokens):  Konservative Parameter für kleine Intervalle
  - MITTEL  (100k-1M):       Standard-Parameter für größere Intervalle
  - GROSS   (>1M):           Optimierte Parameter für Gesamtkorpus

Ausgabedateien:
  - korpus_gen.wordvectors   → PORTABEL, Word-Vektoren (empfohlen zum Laden)
  - korpus_gen.json          → Metadaten, Parameter, Versionen
  - korpus_gen.model         → Vollmodell (nur für exakte NumPy+Gensim Version)

Beispiele:
  korpus_gen.csv              → GROSS  (Gesamtkorpus, ~2 Mio. Tokens)
  korpus_gen_1784-1796.csv    → MITTEL (Intervall)
  korpus_stop.csv             → GROSS  (alternative Content-Spalte)
        """
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Ordner mit den Eingabe-CSV-Dateien (z. B. output/processed_corpus).",
    )
    parser.add_argument(
        "--pattern",
        default="korpus_gen*.csv",
        help="Dateinamen-Pattern für Eingabedateien (Standard: 'korpus_gen*.csv').",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Zielordner für gespeicherte Word2Vec-Modelle.",
    )
    parser.add_argument(
        "--delimiter",
        default=";",
        help="CSV-Delimiter (Standard: ';').",
    )
    return parser.parse_args(argv)


# =============================================================================
# Main (CLI-Wrapper)
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        delimiter=args.delimiter,
    )


if __name__ == "__main__":
    main()
