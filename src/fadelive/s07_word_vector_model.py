#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trainiert Word2Vec-Modelle auf Basis eines oder mehrerer vorverarbeiteter
Korpora 'korpus_gen*.csv' (Gensim-Preprocessing, stopwortbereinigt, lemmatisiert).

ÄNDERUNG v3:
- Automatische Delimiter-Erkennung (Fallback: ";")
- Verwendet gemeinsame pipeline_utils
- Portable Modelle: Speichert nur KeyedVectors
- Adaptive Parameter basierend auf Korpusgröße
- Konsistent mit Pipeline v3

Beispielaufruf:

    python s07_word_vector_model.py \\
        --input-dir output/processed_corpus \\
        --pattern "korpus_gen*.csv" \\
        --output-dir output/word2vec_models \\
        --delimiter auto
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


def _ensure_nltk_punkt():
    """Stellt sicher, dass NLTK punkt-Tokenizer verfügbar ist."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)


# Import der gemeinsamen Utils
try:
    from .pipeline_utils import (
        detect_delimiter,
        identify_content_column,
        identify_metadata_columns,
        identify_id_column,
        identify_year_columns
    )
except ImportError:
    from pipeline_utils import (
        detect_delimiter,
        identify_content_column,
        identify_metadata_columns,
        identify_id_column,
        identify_year_columns
    )

try:
    import gensim
    import numpy as np
    GENSIM_VERSION = gensim.__version__
    NUMPY_VERSION = np.__version__
except ImportError:
    GENSIM_VERSION = "unknown"
    NUMPY_VERSION = "unknown"


# =============================================================================
# Adaptive Parameter-Berechnung
# =============================================================================

def calculate_word2vec_params(num_tokens: int) -> Dict[str, int]:
    """Passt Word2Vec-Parameter automatisch an Korpusgröße an."""
    base_params = {
        'workers': 4,
        'sg': 1,
        'hs': 0,
        'sample': 1e-4,
        'seed': 42,
    }
    
    if num_tokens < 100_000:
        size_params = {'vector_size': 100, 'window': 3, 'min_count': 3, 'negative': 3, 'epochs': 30}
        category = "KLEIN"
    elif num_tokens < 1_000_000:
        size_params = {'vector_size': 150, 'window': 5, 'min_count': 8, 'negative': 5, 'epochs': 20}
        category = "MITTEL"
    else:
        size_params = {'vector_size': 175, 'window': 10, 'min_count': 8, 'negative': 10, 'epochs': 20}
        category = "GROSS"
    
    all_params = {**base_params, **size_params, '_category': category}
    return all_params


# =============================================================================
# Tokenisierung
# =============================================================================

def tokenize_corpus(texts: pd.Series) -> List[List[str]]:
    """Zerlegt Texte in Sätze und Tokens."""
    _ensure_nltk_punkt()  # Stelle sicher, dass NLTK-Ressourcen verfügbar sind
    sentences = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            sents = nltk.sent_tokenize(text, language='german')
            for sent in sents:
                tokens = nltk.word_tokenize(sent, language='german')
                tokens_clean = [t.lower() for t in tokens if t.isalpha()]
                if tokens_clean:
                    sentences.append(tokens_clean)
        except Exception:
            tokens = text.lower().split()
            tokens_clean = [t for t in tokens if t.isalpha()]
            if tokens_clean:
                sentences.append(tokens_clean)
    return sentences


def estimate_token_count(texts: pd.Series) -> int:
    """Schätzt die Token-Anzahl im Korpus."""
    sample_size = min(100, len(texts))
    sample = texts.sample(n=sample_size, random_state=42) if len(texts) > sample_size else texts
    avg_tokens = sample.astype(str).apply(lambda x: len(x.split())).mean()
    return int(avg_tokens * len(texts))


def train_word2vec(sentences: List[List[str]], params: Dict) -> Word2Vec:
    """Trainiert ein Word2Vec-Modell."""
    clean_params = {k: v for k, v in params.items() if not k.startswith('_')}
    return Word2Vec(sentences, **clean_params)


# =============================================================================
# Modell speichern
# =============================================================================

def save_model_portable(model: Word2Vec, base_path: Path, params: Dict, corpus_metadata: Dict) -> None:
    """Speichert das Modell portabel."""
    wv_path = Path(str(base_path) + ".wordvectors")
    model.wv.save(str(wv_path))
    print(f"   ✔ Word-Vektoren: {wv_path.name}")
    
    metadata = {
        'created_at': datetime.now().isoformat(),
        'gensim_version': GENSIM_VERSION,
        'numpy_version': NUMPY_VERSION,
        'parameters': {k: v for k, v in params.items() if not k.startswith('_')},
        'category': params.get('_category', 'UNKNOWN'),
        'vocabulary_size': len(model.wv),
        **corpus_metadata
    }
    
    json_path = Path(str(base_path) + ".json")
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✔ Metadaten: {json_path.name}")
    
    model_path = Path(str(base_path) + ".model")
    model.save(str(model_path))
    print(f"   ✔ Vollmodell: {model_path.name}")


# =============================================================================
# Datei-Handling
# =============================================================================

def find_input_files(input_dir: Path, pattern: str) -> List[Path]:
    """Findet alle Dateien, die dem Pattern entsprechen."""
    return sorted(list(input_dir.glob(pattern)))


def categorize_files(files: List[Path]) -> Tuple[Optional[Path], List[Path]]:
    """Kategorisiert Dateien in Gesamtkorpus und Intervalle."""
    gesamtkorpus = None
    intervals = []
    interval_pattern = re.compile(r'_\d{4}-\d{4}')
    
    for f in files:
        if interval_pattern.search(f.stem):
            intervals.append(f)
        else:
            gesamtkorpus = f
    
    return gesamtkorpus, sorted(intervals)


def process_file(input_file: Path, output_dir: Path, delimiter: str = "auto", file_type: str = "unknown") -> None:
    """Verarbeitet eine CSV-Datei und trainiert ein Word2Vec-Modell."""
    print(f"\n📄 Verarbeite Datei: {input_file}")
    print(f"   🏷️  Typ: {file_type}")

    if not input_file.exists():
        print(f"⚠️ Datei existiert nicht, übersprungen.")
        return

    if delimiter == "auto":
        delimiter = detect_delimiter(input_file)

    try:
        df = pd.read_csv(input_file, encoding="utf-8", sep=delimiter)
    except Exception as e:
        print(f"⚠️ Fehler beim Einlesen: {e}")
        return

    content_col = identify_content_column(df)
    if content_col is None:
        print(f"⚠️ Keine Content-Spalte gefunden, übersprungen.")
        return
    
    print(f"   📝 Content-Spalte: {content_col}")

    texts = df[content_col].astype(str)
    if texts.str.strip().eq("").all():
        print(f"⚠️ Alle Einträge leer, übersprungen.")
        return

    estimated_tokens = estimate_token_count(texts)
    print(f"   📊 Geschätzte Token-Anzahl: ~{estimated_tokens:,}")

    params = calculate_word2vec_params(estimated_tokens)
    category = params.get('_category', 'UNKNOWN')
    print(f"   ⚙️  Kategorie: {category}")

    print("   ➜ Tokenisiere Korpus ...")
    sentences = tokenize_corpus(texts)

    if not sentences:
        print(f"⚠️ Keine Sätze nach Tokenisierung, übersprungen.")
        return

    print(f"   ➜ Trainiere Word2Vec-Modell ({len(sentences):,} Sätze) ...")
    try:
        model = train_word2vec(sentences, params)
    except Exception as e:
        print(f"âŒ Fehler beim Training: {e}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model_base_path = output_dir / input_file.stem
    
    corpus_metadata = {
        'source_file': input_file.name,
        'file_type': file_type,
        'content_column': content_col,
        'num_documents': len(df),
        'num_sentences': len(sentences),
        'estimated_tokens': estimated_tokens,
    }

    save_model_portable(model, model_base_path, params, corpus_metadata)
    print(f"   📈 Vokabulargröße: {len(model.wv):,} Wörter")


# =============================================================================
# run-Funktion
# =============================================================================

def run(input_dir: Path, output_dir: Path, pattern: str = "korpus_gen*.csv", delimiter: str = "auto") -> None:
    """Trainiert Word2Vec-Modelle."""
    print(f"📚 Eingabeordner: {input_dir}")
    print(f"📚 Ausgabeordner: {output_dir}")
    print(f"🔽 Dateipattern:  {pattern}")

    input_files = find_input_files(input_dir, pattern)
    if not input_files:
        print("⚠️ Keine passenden Eingabedateien gefunden.")
        return

    print(f"✔ {len(input_files)} Datei(en) gefunden.")

    gesamtkorpus, intervals = categorize_files(input_files)

    if gesamtkorpus:
        print("\n" + "="*60)
        print("🌐 GESAMTKORPUS")
        print("="*60)
        process_file(gesamtkorpus, output_dir, delimiter, file_type="GESAMTKORPUS")

    if intervals:
        print("\n" + "="*60)
        print(f"📅 INTERVALLE ({len(intervals)} Dateien)")
        print("="*60)
        for interval_file in intervals:
            process_file(interval_file, output_dir, delimiter, file_type="INTERVALL")

    print("\n" + "="*60)
    print("✅ Alle Modelle wurden verarbeitet.")
    print("="*60)


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainiert Word2Vec-Modelle.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Ordner mit Eingabe-CSV-Dateien.")
    parser.add_argument("--pattern", default="korpus_gen*.csv", help="Dateinamen-Pattern.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Zielordner für Modelle.")
    parser.add_argument("--delimiter", default="auto", help="CSV-Delimiter ('auto' für automatische Erkennung).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run(input_dir=args.input_dir, output_dir=args.output_dir, pattern=args.pattern, delimiter=args.delimiter)


if __name__ == "__main__":
    main()
