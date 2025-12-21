#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tag-Topic-Explorer
==================

Eine GUI-Anwendung zur Visualisierung und Exploration von Topic-Modeling-Ergebnissen.

Entwickelt für die Analyse historischer Textkorpora im Kontext der Digital Humanities
und Korpuslinguistik. Ermöglicht die interaktive Exploration von Tag-Topic-Beziehungen,
Zeitreihenanalysen und Relevanz-Visualisierungen.

Funktionen:
-----------
- Automatische Datei-Erkennung (Auto-Discovery) für Projektstrukturen
- Bubble-Charts für Tag-Topic-Relevanz (TF-IDF-basiert)
- Zeitreihenanalysen mit Glättung und Polynomregression
- Stacked Bar Charts für Term-Topic-Verteilungen über Zeit
- Vergleichsvisualisierungen (Tokens vs. Topics vs. Texte)

Projektstruktur (erwartet):
---------------------------
project/
├── resources/
│   ├── termsets/           # Termset-Definitionen (*_suffix.csv)
│   └── topic-models/       # Topic-Model-Ausgaben
│       └── topics*/        # Topic-Ordner
├── output/
│   ├── processed_termset/  # Verarbeitete Termset-Daten
│   ├── processed_topics/   # Verarbeitete Topic-Daten
│   ├── statistics/         # Korpus-Statistiken
│   └── exploration/        # Export-Ordner (wird erstellt)
└── data/
    └── raw/
        └── metadata.csv    # Dokument-Metadaten

Autor: Hendrick Heimböckel

"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt


# =============================================================================
# KONFIGURATION
# =============================================================================

def _detect_project_root() -> Path:
    """
    Ermittelt das Projektstammverzeichnis.
    
    Sucht nach einem Verzeichnis mit 'output/' und 'resources/' Unterordnern,
    ausgehend vom Skript-Speicherort oder dem aktuellen Arbeitsverzeichnis.
    
    Returns:
        Path: Projektstammverzeichnis
    """
    try:
        candidate = Path(__file__).resolve().parents[2]
    except NameError:
        candidate = Path.cwd()
    
    # Prüfe Kandidat und dessen Elternverzeichnis
    for path in [candidate, candidate.parent]:
        if (path / "output").exists() and (path / "resources").exists():
            return path
    
    return candidate


# Globale Pfade
PROJECT_ROOT = _detect_project_root()
OUTPUT_DIR = PROJECT_ROOT / "output"
RESOURCES_DIR = PROJECT_ROOT / "resources"
EXPLORATION_DIR = OUTPUT_DIR / "exploration"
EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DYNAMISCHE METADATEN-ERKENNUNG
# =============================================================================

class MetadataDetector:
    """
    Dynamische Erkennung von Metadaten-Spalten in DataFrames.
    
    Verwendet mehrere Heuristiken:
    1. Bekannte Spaltennamen (konfigurierbar)
    2. Datentyp-Analyse (String-Spalten sind oft Metadaten)
    3. Kardinalitäts-Analyse (wenige unique Werte = wahrscheinlich Metadaten)
    4. Namens-Patterns (id, name, title, year, etc.)
    
    Attributes:
        known_metadata_names: Set bekannter Metadaten-Spaltennamen
        metadata_patterns: Regex-Patterns für Metadaten-Erkennung
    
    Example:
        >>> detector = MetadataDetector()
        >>> meta_cols = detector.detect(df)
        >>> term_cols = detector.get_term_columns(df)
    """
    
    # Standard-Metadaten-Namen (lowercase für Vergleich)
    DEFAULT_METADATA_NAMES = {
        # IDs
        "_id", "id", "doc_id", "document_id", "filename", "file", "name",
        # Autoren
        "author", "author_surname", "author_firstname", "author_name",
        "autor", "verfasser", "author_surname_norm",
        # Titel
        "title", "titel", "title_norm", "heading", "headline",
        # Quellen
        "source", "quelle", "journal", "magazine", "publication",
        "publisher", "verlag", "zeitschrift",
        # Zeit
        "year", "jahr", "date", "datum", "year_first", "year_final",
        "jahr_final", "publication_year", "erscheinungsjahr",
        # Klassifikation
        "textclass", "category", "kategorie", "genre", "type", "typ",
        "class", "klasse", "classification",
        # Ort
        "address", "adresse", "location", "ort", "place",
        "address_author", "city", "stadt", "country", "land",
        # Sprache
        "lang", "language", "sprache",
        # Sonstige
        "url", "path", "pfad", "notes", "bemerkungen", "comments",
    }
    
    # Patterns für automatische Erkennung (case-insensitive)
    DEFAULT_PATTERNS = [
        r"^_",           # Beginnt mit Unterstrich
        r"_?id$",        # Endet mit 'id'
        r"_?name$",      # Endet mit 'name'
        r"_?date$",      # Endet mit 'date'
        r"_?year$",      # Endet mit 'year'
        r"_?jahr$",      # Endet mit 'jahr'
        r"_?title$",     # Endet mit 'title'
        r"_?autor",      # Enthält 'autor'
        r"_?author",     # Enthält 'author'
        r"_norm$",       # Endet mit '_norm' (normalisierte Felder)
    ]
    
    def __init__(
        self,
        known_names: Optional[set] = None,
        patterns: Optional[List[str]] = None,
        max_cardinality_ratio: float = 0.1,
        min_numeric_ratio: float = 0.8
    ):
        """
        Initialisiert den Metadaten-Detektor.
        
        Args:
            known_names: Bekannte Metadaten-Spaltennamen (zusätzlich zu Defaults)
            patterns: Regex-Patterns für Erkennung (zusätzlich zu Defaults)
            max_cardinality_ratio: Max. Verhältnis unique/total für Metadaten
            min_numeric_ratio: Min. Anteil numerischer Werte für Term-Spalten
        """
        self.known_metadata_names = self.DEFAULT_METADATA_NAMES.copy()
        if known_names:
            self.known_metadata_names.update(n.lower() for n in known_names)
        
        self.metadata_patterns = [re.compile(p, re.IGNORECASE) 
                                   for p in (patterns or []) + self.DEFAULT_PATTERNS]
        
        self.max_cardinality_ratio = max_cardinality_ratio
        self.min_numeric_ratio = min_numeric_ratio
    
    def is_metadata_column(self, df: pd.DataFrame, col: str) -> bool:
        """
        Prüft, ob eine Spalte wahrscheinlich Metadaten enthält.
        
        Args:
            df: DataFrame
            col: Spaltenname
            
        Returns:
            True wenn wahrscheinlich Metadaten
        """
        col_lower = col.lower().strip()
        
        # 1. Bekannter Name?
        if col_lower in self.known_metadata_names:
            return True
        
        # 2. Pattern-Match?
        for pattern in self.metadata_patterns:
            if pattern.search(col_lower):
                return True
        
        # 3. Datentyp-Analyse
        series = df[col]
        
        # String-Spalten mit niedriger Kardinalität sind oft Metadaten
        if series.dtype == 'object':
            # Prüfe ob es Text ist (nicht numerisch konvertierbar)
            numeric = pd.to_numeric(series, errors='coerce')
            non_numeric_ratio = numeric.isna().sum() / len(series)
            
            if non_numeric_ratio > 0.5:  # Mehr als 50% nicht-numerisch
                return True
        
        # 4. Kardinalitäts-Check für kategoriale Daten
        if len(df) > 0:
            cardinality_ratio = series.nunique() / len(df)
            # Sehr niedrige Kardinalität = wahrscheinlich Kategorie/Metadaten
            if cardinality_ratio < self.max_cardinality_ratio and series.dtype == 'object':
                return True
        
        return False
    
    def detect(self, df: pd.DataFrame) -> List[str]:
        """
        Erkennt alle Metadaten-Spalten in einem DataFrame.
        
        Args:
            df: DataFrame zur Analyse
            
        Returns:
            Liste der Metadaten-Spaltennamen
        """
        return [col for col in df.columns if self.is_metadata_column(df, col)]
    
    def get_term_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Ermittelt alle Term-Spalten (numerisch, keine Metadaten).
        
        Args:
            df: DataFrame zur Analyse
            
        Returns:
            Liste der Term-Spaltennamen
        """
        result = []
        metadata_cols = set(self.detect(df))
        
        for col in df.columns:
            if col in metadata_cols:
                continue
            
            series = df[col]
            
            # Bereits numerisch?
            if pd.api.types.is_numeric_dtype(series):
                result.append(col)
                continue
            
            # Konvertierbar zu numerisch?
            numeric = pd.to_numeric(series, errors='coerce')
            numeric_ratio = numeric.notna().sum() / len(series) if len(series) > 0 else 0
            
            if numeric_ratio >= self.min_numeric_ratio:
                result.append(col)
        
        return result
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Erstellt detaillierte Analyse aller Spalten.
        
        Args:
            df: DataFrame zur Analyse
            
        Returns:
            Dictionary mit Spaltenname -> Analyse-Details
        """
        analysis = {}
        
        for col in df.columns:
            series = df[col]
            is_meta = self.is_metadata_column(df, col)
            
            # Numerisch-Ratio berechnen
            if pd.api.types.is_numeric_dtype(series):
                numeric_ratio = 1.0
            else:
                numeric = pd.to_numeric(series, errors='coerce')
                numeric_ratio = numeric.notna().sum() / len(series) if len(series) > 0 else 0
            
            analysis[col] = {
                'dtype': str(series.dtype),
                'is_metadata': is_meta,
                'unique_count': series.nunique(),
                'null_count': series.isna().sum(),
                'numeric_ratio': round(numeric_ratio, 3),
                'sample_values': series.dropna().head(3).tolist()
            }
        
        return analysis


# Globale Instanz des Metadaten-Detektors
METADATA_DETECTOR = MetadataDetector()


# =============================================================================
# MATPLOTLIB-KONFIGURATION
# =============================================================================

plt.rcParams.update({
    # Schriftarten
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    
    # Achsen
    "axes.titlesize": 12,
    "axes.titleweight": "regular",
    "axes.labelsize": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    
    # Gitter
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    
    # Tick-Labels
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    
    # Legende
    "legend.fontsize": 8,
    "legend.frameon": False,
    
    # Figur
    "figure.figsize": (12, 6),
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


# =============================================================================
# DATEI-ERKENNUNG (FILE DISCOVERY)
# =============================================================================

class FileDiscovery:
    """
    Automatische Erkennung von Projektdateien basierend auf Ordnerstruktur.
    
    Unterstützt flexible Namenskonventionen:
    - termsets/: Dateien mit festem Suffix (z.B. "*_2.3.csv")
    - topic-models/: Feste Dateinamen in variablen Unterordnern
    - processed_termset/: Keyword-basierte Suche
    
    Attributes:
        base_dir: Basis-Projektverzeichnis
        termset_suffix: Suffix für Termset-Dateien (z.B. "_2.3.csv")
        found_files: Dictionary mit gefundenen Dateipfaden
    
    Example:
        >>> discovery = FileDiscovery(Path("/projekt"), "_2.3.csv")
        >>> files = discovery.scan_project()
        >>> print(files['termset'])
        /projekt/resources/termsets/Begriffe_2.3.csv
    """
    
    def __init__(self, base_dir: Path, termset_suffix: str = "_2.3.csv"):
        self.base_dir = base_dir
        self.termset_suffix = termset_suffix
        self.found_files: Dict[str, Optional[Path]] = {}
    
    def scan_project(self) -> Dict[str, Optional[Path]]:
        """
        Scannt das Projekt und findet alle relevanten Dateien.
        
        Returns:
            Dictionary mit Datei-Keys und gefundenen Pfaden (None wenn nicht gefunden)
        """
        self.found_files = {
            # Basis-Dateien
            'termset': self._find_in_termsets(),
            'topic_words': self._find_topic_words(),
            'tfidf': self._find_tfidf(),
            
            # Verarbeitete Termset-Daten
            'ranks': self._find_in_processed('tag_', 'topic', 'rank'),
            'relevance': self._find_in_processed('relevance'),
            'counts_per_year': self._find_in_processed('counts', 'year'),
            'top10_year_value': self._find_in_processed('year', 'value'),
            'top10_value_per_text': self._find_in_processed('value', 'text'),
            
            # Topic-Model-Ausgaben
            'topics_dist': self._find_topics_distribution(),
            
            # Statistiken
            'tokens_year': self._find_tokens_per_year(),
            'global_topdocs': self._find_global_topdocs(),
            
            # Metadaten
            'metadata': self._find_metadata(),
        }
        return self.found_files
    
    def _find_in_termsets(self) -> Optional[Path]:
        """Findet Termset-Datei in resources/termsets/."""
        search_dir = self.base_dir / "resources" / "termsets"
        if not search_dir.exists():
            return None
        
        for path in search_dir.glob(f"*{self.termset_suffix}"):
            return path
        return None
    
    def _find_topic_words(self) -> Optional[Path]:
        """Findet Topic-Words-Datei in resources/topic-models/topics*/."""
        search_dir = self.base_dir / "resources" / "topic-models"
        if not search_dir.exists():
            return None
        
        for topics_dir in search_dir.glob("topics*"):
            if topics_dir.is_dir():
                for path in topics_dir.glob("*words*tag*.csv"):
                    return path
        return None
    
    def _find_tfidf(self) -> Optional[Path]:
        """Findet TF-IDF-Datei in output/dtm_tfidf*/."""
        search_dir = self.base_dir / "output"
        if not search_dir.exists():
            return None
        
        for tfidf_dir in search_dir.glob("dtm_tfidf*"):
            if tfidf_dir.is_dir():
                for path in tfidf_dir.glob("tfidf*.csv"):
                    return path
        return None
    
    def _find_in_processed(self, *keywords: str) -> Optional[Path]:
        """
        Findet Datei in output/processed_termset/ basierend auf Keywords.
        
        Args:
            *keywords: Keywords, die im Dateinamen enthalten sein müssen
            
        Returns:
            Pfad zur gefundenen Datei oder None
        """
        search_dir = self.base_dir / "output" / "processed_termset"
        if not search_dir.exists():
            return None
        
        # Suffix bereinigen: "_2.3.csv" → "2.3"
        suffix_clean = self.termset_suffix.lstrip('_').replace('.csv', '')
        
        for termset_dir in search_dir.glob(f"Termset*{suffix_clean}"):
            if not termset_dir.is_dir():
                continue
            
            # Suche nach Datei mit allen Keywords
            pattern = "*" + "*".join(keywords) + "*.csv"
            for path in termset_dir.glob(pattern):
                return path
        
        return None
    
    def _find_topics_distribution(self) -> Optional[Path]:
        """Findet Document-Topics-Distribution in topic-models/."""
        search_dir = self.base_dir / "resources" / "topic-models"
        if not search_dir.exists():
            return None
        
        for topics_dir in search_dir.glob("topics*"):
            if topics_dir.is_dir():
                target = topics_dir / "document-topics-distribution_tag.csv"
                if target.exists():
                    return target
        return None
    
    def _find_tokens_per_year(self) -> Optional[Path]:
        """Findet Tokens-per-Year-Statistik."""
        search_dir = self.base_dir / "output" / "statistics"
        if not search_dir.exists():
            return None
        
        for path in search_dir.glob("*tokens*.csv"):
            if 'year' in path.name.lower():
                return path
        return None
    
    def _find_global_topdocs(self) -> Optional[Path]:
        """Findet globale TopDocs-Statistik."""
        search_dir = self.base_dir / "output" / "processed_topics"
        if not search_dir.exists():
            return None
        
        for path in search_dir.iterdir():
            if path.is_file() and path.suffix == '.csv':
                name_lower = path.name.lower()
                if "year" in name_lower and "value" in name_lower:
                    return path
        return None
    
    def _find_metadata(self) -> Optional[Path]:
        """Findet Metadaten-Datei an Standardorten."""
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
        """Erstellt einen formatierten Status-Report."""
        lines = [
            f"📁 Arbeitsordner: {self.base_dir}",
            f"🔖 Termset-Suffix: {self.termset_suffix}",
            ""
        ]
        
        categories = {
            "Basis-Dateien": ['termset', 'topic_words', 'tfidf'],
            "Verarbeitete Termset-Daten": [
                'ranks', 'relevance', 'counts_per_year', 
                'top10_year_value', 'top10_value_per_text'
            ],
            "Topic-Model-Ausgaben": ['topics_dist'],
            "Statistiken": ['tokens_year', 'global_topdocs'],
            "Metadaten": ['metadata'],
        }
        
        for category, keys in categories.items():
            lines.append(f"{category}:")
            for key in keys:
                path = self.found_files.get(key)
                if path:
                    try:
                        # Beide Pfade zu absoluten Pfaden auflösen
                        abs_path = Path(path).resolve()
                        abs_base = Path(self.base_dir).resolve()
                        rel_path = abs_path.relative_to(abs_base)
                    except ValueError:
                        # Fallback: nur Dateinamen anzeigen
                        rel_path = Path(path).name
                    lines.append(f"  ✅ {key}: {rel_path}")
                else:
                    lines.append(f"  ❌ {key}: nicht gefunden")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_missing_files(self) -> List[str]:
        """Gibt Liste der nicht gefundenen Dateien zurück."""
        return [key for key, path in self.found_files.items() if path is None]


# =============================================================================
# DATENMANAGER
# =============================================================================

class DataManager:
    """
    Zentrale Datenverwaltung mit Lazy Loading und Caching.
    
    Verwaltet alle Dateipfade und geladenen DataFrames. Verwendet Caching,
    um wiederholtes Laden zu vermeiden. Cache wird automatisch invalidiert,
    wenn sich Pfade ändern.
    
    Attributes:
        current_base_dir: Aktuelles Basis-Projektverzeichnis
        current_termset_suffix: Aktuelles Termset-Suffix
    """
    
    def __init__(self):
        # Dateipfade (Defaults)
        self._init_default_paths()
        
        # Gecachte DataFrames
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}
        
        # Aktuelle Konfiguration
        self.current_base_dir: Path = PROJECT_ROOT
        self.current_termset_suffix: str = "_2.3.csv"
    
    def _init_default_paths(self):
        """Initialisiert Standard-Dateipfade."""
        base = OUTPUT_DIR / "processed_termset" / "Termset_Begriffe_2.3"
        
        self.path_termset = RESOURCES_DIR / "termsets" / "Termset_Begriffe_2.3.csv"
        self.path_topic_words = RESOURCES_DIR / "topic-models" / "topics_v3" / "topic_words_tag.csv"
        self.path_tfidf = OUTPUT_DIR / "dtm_tfidf_stop" / "tfidf-2000.csv"
        self.path_ranks = base / "Termset_Begriffe_2.3_tag_topic_rank.csv"
        self.path_relevance = base / "Termset_Begriffe_2.3_tag_topic_relevance.csv"
        self.path_counts_per_year = base / "Termset_Begriffe_2.3_dtti_topdocs_topic_counts_per_year.csv"
        self.path_top10_year_value = base / "Termset_Begriffe_2.3_dtti_topdocs_top10_year_value.csv"
        self.path_top10_value_per_text = base / "Termset_Begriffe_2.3_dtti_topdocs_top10_value_per_text_topic.csv"
        self.path_tokens_year = OUTPUT_DIR / "statistics" / "year_count_tokens.csv"
        self.path_global_topdocs = OUTPUT_DIR / "processed_topics" / "document-topics-distribution_tag_topdocs_year_value.csv"
        self.path_topics_dist = RESOURCES_DIR / "topic-models" / "topics_v3" / "document-topics-distribution_tag.csv"
        self.path_metadata = PROJECT_ROOT / "data" / "raw" / "metadata.csv"
    
    def invalidate_cache(self):
        """Invalidiert alle gecachten DataFrames."""
        self._cache.clear()
    
    def auto_discover(self, base_dir: Path, suffix: str) -> Dict[str, Optional[Path]]:
        """
        Führt Auto-Discovery durch und aktualisiert Pfade.
        
        Args:
            base_dir: Basis-Projektverzeichnis
            suffix: Termset-Suffix (z.B. "_2.3.csv")
            
        Returns:
            Dictionary mit gefundenen Dateipfaden
        """
        self.current_base_dir = base_dir
        self.current_termset_suffix = suffix
        self.invalidate_cache()
        
        discovery = FileDiscovery(base_dir, suffix)
        found = discovery.scan_project()
        
        # Aktualisiere Pfade für gefundene Dateien
        path_mapping = {
            'termset': 'path_termset',
            'topic_words': 'path_topic_words',
            'tfidf': 'path_tfidf',
            'ranks': 'path_ranks',
            'relevance': 'path_relevance',
            'counts_per_year': 'path_counts_per_year',
            'top10_year_value': 'path_top10_year_value',
            'top10_value_per_text': 'path_top10_value_per_text',
            'tokens_year': 'path_tokens_year',
            'global_topdocs': 'path_global_topdocs',
            'topics_dist': 'path_topics_dist',
            'metadata': 'path_metadata',
        }
        
        for key, attr in path_mapping.items():
            if found.get(key):
                setattr(self, attr, found[key])
        
        return found
    
    def get_discovery_report(self) -> str:
        """Erstellt Status-Report basierend auf aktuellen Pfaden."""
        discovery = FileDiscovery(self.current_base_dir, self.current_termset_suffix)
        
        def path_if_exists(p: Path) -> Optional[Path]:
            """Prüft ob Pfad existiert, behandelt Fehler."""
            try:
                return p if p and Path(p).exists() else None
            except (OSError, ValueError):
                return None
        
        # Fülle found_files mit aktuellen Pfaden
        discovery.found_files = {
            'termset': path_if_exists(self.path_termset),
            'topic_words': path_if_exists(self.path_topic_words),
            'tfidf': path_if_exists(self.path_tfidf),
            'ranks': path_if_exists(self.path_ranks),
            'relevance': path_if_exists(self.path_relevance),
            'counts_per_year': path_if_exists(self.path_counts_per_year),
            'top10_year_value': path_if_exists(self.path_top10_year_value),
            'top10_value_per_text': path_if_exists(self.path_top10_value_per_text),
            'topics_dist': path_if_exists(self.path_topics_dist),
            'tokens_year': path_if_exists(self.path_tokens_year),
            'global_topdocs': path_if_exists(self.path_global_topdocs),
            'metadata': path_if_exists(self.path_metadata),
        }
        
        return discovery.get_status_report()
    
    # --- Loader-Methoden ---
    
    def _load_csv(self, path: Path, cache_key: str, **kwargs) -> pd.DataFrame:
        """Generischer CSV-Loader mit Caching."""
        if cache_key in self._cache and self._cache[cache_key] is not None:
            return self._cache[cache_key]
        
        if not path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {path}")
        
        df = pd.read_csv(path, **kwargs)
        self._cache[cache_key] = df
        return df
    
    def load_termset(self) -> pd.DataFrame:
        """Lädt Termset-Definitionen (lowercase normalisiert)."""
        df = self._load_csv(self.path_termset, 'termset')
        # Normalisiere zu lowercase
        return df.apply(lambda col: col.map(
            lambda x: str(x).strip().lower() if pd.notna(x) else x
        ))
    
    def load_topic_words(self) -> pd.DataFrame:
        """Lädt Topic-Words-Matrix."""
        df = self._load_csv(self.path_topic_words, 'topic_words', index_col=0)
        return df.apply(lambda col: col.map(
            lambda x: str(x).strip().lower() if pd.notna(x) else x
        ))
    
    def load_tfidf(self) -> pd.DataFrame:
        """Lädt TF-IDF-Matrix."""
        return self._load_csv(self.path_tfidf, 'tfidf')
    
    def load_ranks(self) -> pd.DataFrame:
        """Lädt Tag-Topic-Rankings."""
        return self._load_csv(self.path_ranks, 'ranks')
    
    def load_relevance(self) -> pd.DataFrame:
        """Lädt Tag-Topic-Relevanzscores."""
        return self._load_csv(self.path_relevance, 'relevance')
    
    def load_counts_per_year(self) -> pd.DataFrame:
        """Lädt Topic-Counts pro Jahr."""
        return self._load_csv(self.path_counts_per_year, 'counts_per_year', index_col=0)
    
    def load_top10_year_value(self) -> pd.DataFrame:
        """Lädt Top-10-Werte pro Jahr."""
        return self._load_csv(self.path_top10_year_value, 'top10_year_value')
    
    def load_top10_value_per_text(self) -> pd.DataFrame:
        """Lädt Top-10-Werte pro Text."""
        return self._load_csv(self.path_top10_value_per_text, 'top10_value_per_text')
    
    def load_tokens_year(self) -> pd.DataFrame:
        """Lädt Token-Statistik pro Jahr."""
        return self._load_csv(self.path_tokens_year, 'tokens_year')
    
    def load_global_topdocs(self) -> pd.DataFrame:
        """Lädt globale TopDocs-Statistik."""
        return self._load_csv(self.path_global_topdocs, 'global_topdocs')
    
    def load_topics_dist(self) -> pd.DataFrame:
        """Lädt Document-Topics-Distribution."""
        df = self._load_csv(self.path_topics_dist, 'topics_dist', index_col=0)
        # Entferne .txt aus Index
        df.index = df.index.astype(str).str.replace(".txt", "", regex=False)
        return df
    
    def load_metadata(self) -> pd.DataFrame:
        """Lädt Dokument-Metadaten."""
        if 'metadata' in self._cache and self._cache['metadata'] is not None:
            return self._cache['metadata']
        
        if not self.path_metadata.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {self.path_metadata}")
        
        df = pd.read_csv(self.path_metadata, sep=";")
        df["_id"] = df["_id"].astype(str)
        
        # Jahr-Koaleszenz: year_first hat Vorrang vor year
        df = self._coalesce_years(df)
        
        self._cache['metadata'] = df
        return df
    
    @staticmethod
    def _coalesce_years(df: pd.DataFrame) -> pd.DataFrame:
        """Kombiniert year_first und year zu Jahr_final."""
        year_first = pd.to_numeric(df.get('year_first', pd.Series(dtype='float64')), errors='coerce')
        year = pd.to_numeric(df.get('year', pd.Series(dtype='float64')), errors='coerce')
        df['Jahr_final'] = year_first.where(year_first.notna(), year)
        return df


# Globale DataManager-Instanz
DATA = DataManager()


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def normalize_filename(s: str) -> str:
    """Normalisiert String für Dateinamen (entfernt Sonderzeichen)."""
    return re.sub(r"[^\w\-.,+]+", "_", s).strip("_") or "export"


def parse_year_range(default_min: int, default_max: int, raw: str) -> Tuple[int, int]:
    """
    Parst Jahresbereich aus String (Format: "YYYY-YYYY").
    
    Args:
        default_min: Standard-Minimum bei Parse-Fehler
        default_max: Standard-Maximum bei Parse-Fehler
        raw: Eingabe-String
        
    Returns:
        Tuple (min_year, max_year)
    """
    raw = (raw or "").strip()
    if "-" in raw:
        parts = raw.split("-", 1)
        try:
            lo, hi = int(parts[0].strip()), int(parts[1].strip())
            if lo <= hi:
                return lo, hi
        except ValueError:
            pass
    return default_min, default_max


def get_term_columns(df: pd.DataFrame) -> List[str]:
    """
    Ermittelt Term-Spalten (numerisch, keine Metadaten).
    
    Verwendet den globalen MetadataDetector für dynamische Erkennung.
    
    Args:
        df: DataFrame mit potenziellen Term-Spalten
        
    Returns:
        Liste der Term-Spaltennamen
    """
    return METADATA_DETECTOR.get_term_columns(df)


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Findet erste vorhandene Spalte aus Kandidatenliste."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def normalize_document_id(doc_id: str) -> str:
    """
    Normalisiert Dokument-ID für Matching.
    
    Entfernt Pfade, Erweiterungen und Präfixe.
    """
    doc_id = str(doc_id)
    doc_id = Path(doc_id).stem  # Entferne Pfad und Erweiterung
    doc_id = re.sub(r'^(doc_?|text_?|id_?)', '', doc_id, flags=re.IGNORECASE)
    return doc_id.strip()


# =============================================================================
# SPEICHER-FUNKTIONEN
# =============================================================================

def save_dataframe(df: Optional[pd.DataFrame], tab: str, context: str, parent: tk.Tk) -> None:
    """Speichert DataFrame als CSV mit Datei-Dialog."""
    if df is None or df.empty:
        messagebox.showinfo("Info", "Keine Daten zum Speichern.", parent=parent)
        return
    
    try:
        save_dir = EXPLORATION_DIR / tab
        save_dir.mkdir(parents=True, exist_ok=True)
        
        path = filedialog.asksaveasfilename(
            parent=parent,
            title="Als CSV speichern",
            initialdir=str(save_dir),
            initialfile=f"{normalize_filename(context)}.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Alle Dateien", "*.*")]
        )
        
        if path:
            df.to_csv(path, index=False)
            messagebox.showinfo("Gespeichert", path, parent=parent)
    
    except Exception as e:
        messagebox.showerror("Fehler", str(e), parent=parent)


def save_figure(fig: plt.Figure, tab: str, context: str, parent: tk.Tk, dpi: int = 300) -> None:
    """Speichert Figure als PNG mit Datei-Dialog."""
    try:
        save_dir = EXPLORATION_DIR / tab
        save_dir.mkdir(parents=True, exist_ok=True)
        
        path = filedialog.asksaveasfilename(
            parent=parent,
            title="Als PNG speichern",
            initialdir=str(save_dir),
            initialfile=f"{normalize_filename(context)}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("Alle Dateien", "*.*")]
        )
        
        if path:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            messagebox.showinfo("Gespeichert", path, parent=parent)
    
    except Exception as e:
        messagebox.showerror("Fehler", str(e), parent=parent)


# =============================================================================
# TKINTER-HILFSFUNKTIONEN
# =============================================================================

def create_entry(parent, **kwargs) -> ttk.Entry:
    """Erstellt ein konfiguriertes Entry-Widget."""
    entry = ttk.Entry(parent, **kwargs)
    entry.configure(state="normal", takefocus=True)
    return entry


def setup_window(root: tk.Tk) -> None:
    """Konfiguriert Hauptfenster mit sicherem Beenden."""
    def safe_exit():
        try:
            for widget in root.winfo_children():
                widget.destroy()
            root.quit()
        except Exception:
            pass
        finally:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", safe_exit)
    root.bind("<Escape>", lambda e: safe_exit())


def apply_figure_layout(fig: plt.Figure) -> None:
    """Wendet optimales Layout auf Figure an."""
    try:
        fig.set_constrained_layout(True)
        fig.canvas.draw_idle()
    except Exception:
        try:
            fig.tight_layout()
        except Exception:
            pass


# =============================================================================
# DATENVERARBEITUNG FÜR VISUALISIERUNGEN
# =============================================================================

def compute_tfidf_sums() -> pd.Series:
    """
    Berechnet TF-IDF-Summen pro Term.
    
    Returns:
        Series mit Term als Index und TF-IDF-Summe als Wert
    """
    df = DATA.load_tfidf()
    term_cols = get_term_columns(df)
    
    if not term_cols:
        raise ValueError("Keine Term-Spalten in TF-IDF-Datei gefunden.")
    
    numeric_df = df[term_cols].apply(pd.to_numeric, errors='coerce')
    sums = numeric_df.sum(skipna=True)
    sums.index = sums.index.astype(str).str.strip().str.lower()
    
    return sums


def get_tag_dict(df_tags: pd.DataFrame) -> Dict[str, List[str]]:
    """Erstellt Dictionary: Tag -> Liste von Ausdrücken."""
    return {
        col: df_tags[col].dropna().astype(str).str.strip().tolist()
        for col in df_tags.columns
    }


def get_topic_word_map(df_topics: pd.DataFrame) -> Dict[str, List[str]]:
    """Erstellt Dictionary: Topic -> Liste von Wörtern."""
    return {
        str(topic): df_topics.loc[topic].dropna().astype(str).str.strip().tolist()
        for topic in df_topics.index
    }


def get_ranked_topics_for_counts(df_counts: pd.DataFrame, df_ranks: pd.DataFrame, top_n: int) -> List[str]:
    """
    Findet Top-N Topics aus Rankings die in Counts-DataFrame vorkommen.
    
    Verwendet flexible Matching-Strategien für verschiedene Namensformate.
    
    Args:
        df_counts: DataFrame mit Topic-Spalten
        df_ranks: DataFrame mit Rankings
        top_n: Anzahl gewünschter Topics
        
    Returns:
        Liste der gematchten Topic-Spaltennamen
    """
    topics_in_df = set(df_counts.columns.astype(str))
    
    # Finde Rang-Spalte flexibel
    rank_col = None
    for col in ['TFIDF-Positions-Rang', 'Rang', 'rank', 'Rank']:
        if col in df_ranks.columns:
            rank_col = col
            break
    if rank_col is None:
        for col in df_ranks.columns:
            if 'rang' in col.lower() or 'rank' in col.lower():
                rank_col = col
                break
    
    # Finde Topic-Spalte flexibel
    topic_col = None
    for col in ['Topic', 'topic', 'TOPIC']:
        if col in df_ranks.columns:
            topic_col = col
            break
    
    if not rank_col or not topic_col:
        # Fallback: Top-N nach Summe
        df_numeric = df_counts.apply(pd.to_numeric, errors='coerce')
        counts = df_numeric.sum(axis=0, skipna=True).sort_values(ascending=False)
        return counts.head(top_n).index.tolist()
    
    r = df_ranks.copy()
    r['_topic'] = r[topic_col].astype(str)
    r['_rank'] = pd.to_numeric(r[rank_col], errors='coerce')
    r = r.dropna(subset=['_rank']).sort_values('_rank')
    
    ranked = []
    for topic in r['_topic'].tolist():
        if len(ranked) >= top_n:
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
        
        # Strategie 3: Ohne "Topic " Präfix
        if topic_str.startswith("Topic "):
            topic_without = topic_str.replace("Topic ", "", 1)
            if topic_without in topics_in_df:
                ranked.append(topic_without)
                continue
        
        # Strategie 4: Nur Nummer extrahieren
        match = re.search(r'\d+', topic_str)
        if match:
            num = match.group()
            for variant in [f"Topic {num}", num, f"topic_{num}", f"topic{num}"]:
                if variant in topics_in_df:
                    ranked.append(variant)
                    break
    
    # Fallback wenn nichts gefunden
    if not ranked:
        df_numeric = df_counts.apply(pd.to_numeric, errors='coerce')
        counts = df_numeric.sum(axis=0, skipna=True).sort_values(ascending=False)
        ranked = counts.head(top_n).index.tolist()
    
    return ranked


def match_text_to_metadata(text_value: str, metadata_df: pd.DataFrame) -> Optional[str]:
    """
    Mappt einen Text-Wert auf eine _id in den Metadaten.
    
    Prüft mehrere Strategien:
    1. Direkte _id Übereinstimmung
    2. author_surname + title + year Matching
    3. Lockereres Matching (2 von 3)
    4. Normalisierte ID-Übereinstimmung
    
    Args:
        text_value: Der Text-Wert aus der Daten-Tabelle
        metadata_df: DataFrame mit Metadaten
    
    Returns:
        Die gefundene _id oder None
    """
    text_str = str(text_value).strip().lower()
    
    if not text_str or text_str == "nan":
        return None
    
    # Strategie 1: Direkte _id Übereinstimmung
    if "_id" in metadata_df.columns:
        for _, row in metadata_df.iterrows():
            row_id = str(row["_id"]).strip()
            if row_id.lower() == text_str or row_id == text_value:
                return row_id
    
    # Strategie 2: author_surname + title + year
    for _, row in metadata_df.iterrows():
        author = str(row.get("author_surname", "")).strip().lower() if pd.notna(row.get("author_surname")) else ""
        title = str(row.get("title", "")).strip().lower() if pd.notna(row.get("title")) else ""
        year = str(int(row.get("year"))) if pd.notna(row.get("year")) else ""
        year_first = str(int(row.get("year_first"))) if pd.notna(row.get("year_first")) else ""
        
        author_match = author and len(author) >= 3 and author in text_str
        title_match = title and len(title) >= 5 and (title in text_str or title[:20] in text_str)
        year_match = (year and year in text_str) or (year_first and year_first in text_str)
        
        if author_match and title_match and year_match:
            return str(row["_id"])
    
    # Strategie 3: Lockereres Matching (2 von 3)
    for _, row in metadata_df.iterrows():
        author = str(row.get("author_surname", "")).strip().lower() if pd.notna(row.get("author_surname")) else ""
        title = str(row.get("title", "")).strip().lower() if pd.notna(row.get("title")) else ""
        year = str(int(row.get("year"))) if pd.notna(row.get("year")) else ""
        year_first = str(int(row.get("year_first"))) if pd.notna(row.get("year_first")) else ""
        
        author_match = author and len(author) >= 3 and author in text_str
        title_match = title and len(title) >= 5 and title[:15] in text_str
        year_match = (year and year in text_str) or (year_first and year_first in text_str)
        
        if author_match and title_match:
            return str(row["_id"])
        if author_match and year_match and len(author) >= 5:
            return str(row["_id"])
    
    # Strategie 4: Normalisierte ID
    normalized_text = re.sub(r'^(doc|text|document|file)[-_\s]*', '', text_str, flags=re.IGNORECASE)
    normalized_text = re.sub(r'\.(txt|csv|pdf)$', '', normalized_text, flags=re.IGNORECASE)
    normalized_text = normalized_text.lstrip('0') or '0'
    
    if "_id" in metadata_df.columns:
        for _, row in metadata_df.iterrows():
            row_id = str(row["_id"]).strip()
            normalized_id = re.sub(r'^(doc|text|document|file)[-_\s]*', '', row_id.lower(), flags=re.IGNORECASE)
            normalized_id = normalized_id.lstrip('0') or '0'
            if normalized_text == normalized_id:
                return row_id
    
    return None


def get_ranked_topics(df_ranks: pd.DataFrame, available_topics: set, top_n: int) -> List[str]:
    """
    Ermittelt Top-N Topics nach Rang.
    
    Args:
        df_ranks: DataFrame mit Rankings
        available_topics: Set verfügbarer Topic-Namen
        top_n: Anzahl gewünschter Topics
        
    Returns:
        Liste der Top-N Topic-Namen in Rangfolge
    """
    df = df_ranks.copy()
    
    # Finde Topic-Spalte (flexibel)
    topic_col = None
    for col in ['Topic', 'topic', 'TOPIC', 'topic_id', 'Topic_ID']:
        if col in df.columns:
            topic_col = col
            break
    
    if topic_col is None:
        # Erste Spalte als Fallback
        topic_col = df.columns[0] if len(df.columns) > 0 else None
    
    if topic_col is None:
        return []
    
    # Finde Rang-Spalte (flexibel)
    rank_col = None
    for col in ['TFIDF-Positions-Rang', 'Rang', 'rank', 'Rank', 'position']:
        if col in df.columns:
            rank_col = col
            break
    
    if rank_col is None:
        # Suche nach Spalte mit "rang" oder "rank" im Namen
        for col in df.columns:
            if 'rang' in col.lower() or 'rank' in col.lower():
                rank_col = col
                break
    
    if rank_col is None:
        return []
    
    df['_topic_str'] = df[topic_col].astype(str).str.strip()
    df['_rank'] = pd.to_numeric(df[rank_col], errors='coerce')
    df = df.dropna(subset=['_rank']).sort_values('_rank')
    
    # Normalisiere available_topics für Vergleich
    available_normalized = {str(t).strip(): str(t).strip() for t in available_topics}
    
    # Versuche direktes Matching
    result = []
    for topic in df['_topic_str'].tolist():
        if topic in available_normalized:
            result.append(topic)
        # Auch ohne Klammern versuchen (Topic "5 (xyz)" -> "5")
        elif topic.split('(')[0].strip() in available_normalized:
            result.append(topic.split('(')[0].strip())
        # Auch mit nur der Nummer versuchen
        elif topic.split()[0] in available_normalized:
            result.append(topic.split()[0])
        
        if len(result) >= top_n:
            break
    
    return result[:top_n]


# =============================================================================
# VISUALISIERUNGS-TABS
# =============================================================================

def build_tab_workspace(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """
    Tab: Arbeitsordner-Verwaltung
    
    Ermöglicht die Auswahl des Projektverzeichnisses und führt
    Auto-Discovery für Projektdateien durch.
    """
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="📁 Arbeitsordner")
    
    # Header
    ttk.Label(
        frame, 
        text="Arbeitsordner-Verwaltung", 
        font=("TkDefaultFont", 12, "bold")
    ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(6, 10))
    
    # Arbeitsordner-Auswahl
    ttk.Label(frame, text="Arbeitsordner:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    workspace_var = tk.StringVar(value=str(DATA.current_base_dir))
    create_entry(frame, width=70, textvariable=workspace_var).grid(
        row=1, column=1, sticky="we", padx=6, pady=4
    )
    frame.columnconfigure(1, weight=1)
    
    def browse_workspace():
        path = filedialog.askdirectory(parent=root, initialdir=str(DATA.current_base_dir))
        if path:
            workspace_var.set(path)
    
    ttk.Button(frame, text="📁 Wählen...", command=browse_workspace).grid(
        row=1, column=2, sticky="w", padx=4, pady=4
    )
    
    # Termset-Suffix
    ttk.Label(frame, text="Termset-Suffix:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
    suffix_var = tk.StringVar(value=DATA.current_termset_suffix)
    create_entry(frame, width=20, textvariable=suffix_var).grid(
        row=2, column=1, sticky="w", padx=6, pady=4
    )
    ttk.Label(frame, text="z.B. _2.3.csv", foreground="gray").grid(
        row=2, column=2, sticky="w", padx=4, pady=4
    )
    
    # Info-Box
    info_frame = ttk.LabelFrame(frame, text="ℹ️ Hinweis", padding=8)
    info_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=8)
    ttk.Label(
        info_frame,
        text="Das Tool sucht automatisch nach Dateien basierend auf:\n"
             "• termsets/: Dateien mit dem angegebenen Suffix\n"
             "• topic-models/: Feste Dateinamen in topics*/-Ordnern\n"
             "• processed_termset/: Keyword-basierte Suche",
        justify="left"
    ).pack(anchor="w")
    
    # Status-Anzeige
    ttk.Label(frame, text="Gefundene Dateien:").grid(row=4, column=0, sticky="nw", padx=6, pady=4)
    
    status_text = tk.Text(frame, height=18, width=100, wrap="word", font=("Courier", 9))
    status_text.grid(row=4, column=1, columnspan=2, sticky="nsew", padx=6, pady=4)
    frame.rowconfigure(4, weight=1)
    
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=status_text.yview)
    status_text.configure(yscrollcommand=scrollbar.set)
    scrollbar.grid(row=4, column=3, sticky="ns")
    
    def scan_files():
        """Führt Datei-Scan durch."""
        status_text.delete(1.0, tk.END)
        status_text.insert(tk.END, "🔄 Scanne Dateien...\n\n")
        root.update_idletasks()
        
        try:
            base_dir = Path(workspace_var.get())
            suffix = suffix_var.get().strip()
            
            if not base_dir.exists():
                messagebox.showerror("Fehler", f"Ordner existiert nicht:\n{base_dir}", parent=root)
                return
            
            found = DATA.auto_discover(base_dir, suffix)
            
            status_text.delete(1.0, tk.END)
            status_text.insert(tk.END, DATA.get_discovery_report())
            
            # Zusammenfassung
            missing = [k for k, v in found.items() if v is None]
            found_count = len(found) - len(missing)
            
            if missing:
                status_text.insert(tk.END, f"\n⚠️ {len(missing)} Datei(en) nicht gefunden\n")
            else:
                status_text.insert(tk.END, "\n✅ Alle Dateien gefunden!\n")
            
            status_text.insert(tk.END, "\n💡 Wechseln Sie zum Tab 'Daten' zum Laden.\n")
            
            messagebox.showinfo(
                "Scan abgeschlossen",
                f"Gefunden: {found_count}/{len(found)} Dateien",
                parent=root
            )
        
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            status_text.insert(tk.END, f"\n❌ Fehler: {e}\n")
    
    def show_status():
        """Zeigt aktuellen Status."""
        status_text.delete(1.0, tk.END)
        status_text.insert(tk.END, DATA.get_discovery_report())
    
    # Buttons
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=8)
    
    ttk.Button(btn_frame, text="🔄 Dateien scannen", command=scan_files).pack(side="left", padx=(0, 10))
    ttk.Button(btn_frame, text="📊 Status anzeigen", command=show_status).pack(side="left")
    
    # Initial: Status anzeigen
    show_status()


def build_tab_data(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """
    Tab: Datenverwaltung
    
    Zeigt alle Dateipfade und ermöglicht manuelles Laden/Prüfen.
    """
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="📊 Daten")
    
    path_vars: Dict[str, tk.StringVar] = {}
    row = 0
    
    def add_path_row(label: str, key: str, getter, setter):
        nonlocal row
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        
        var = tk.StringVar(value=str(getter()))
        path_vars[key] = var
        
        create_entry(frame, width=80, textvariable=var).grid(
            row=row, column=1, sticky="we", padx=6, pady=3
        )
        
        def browse():
            path = filedialog.askopenfilename(
                parent=root,
                initialdir=str(Path(var.get()).parent),
                filetypes=[("CSV", "*.csv"), ("Alle", "*.*")]
            )
            if path:
                var.set(path)
                setter(Path(path))
        
        ttk.Button(frame, text="...", width=3, command=browse).grid(
            row=row, column=2, sticky="w", padx=4
        )
        
        frame.columnconfigure(1, weight=1)
        row += 1
    
    # Dateipfade
    add_path_row("Termset:", "termset", lambda: DATA.path_termset, lambda p: setattr(DATA, 'path_termset', p))
    add_path_row("Topic-Words:", "topic_words", lambda: DATA.path_topic_words, lambda p: setattr(DATA, 'path_topic_words', p))
    add_path_row("TF-IDF:", "tfidf", lambda: DATA.path_tfidf, lambda p: setattr(DATA, 'path_tfidf', p))
    add_path_row("Rankings:", "ranks", lambda: DATA.path_ranks, lambda p: setattr(DATA, 'path_ranks', p))
    add_path_row("Relevanz:", "relevance", lambda: DATA.path_relevance, lambda p: setattr(DATA, 'path_relevance', p))
    add_path_row("Counts/Jahr:", "counts_year", lambda: DATA.path_counts_per_year, lambda p: setattr(DATA, 'path_counts_per_year', p))
    add_path_row("Top10/Jahr:", "top10_year", lambda: DATA.path_top10_year_value, lambda p: setattr(DATA, 'path_top10_year_value', p))
    add_path_row("Top10/Text:", "top10_text", lambda: DATA.path_top10_value_per_text, lambda p: setattr(DATA, 'path_top10_value_per_text', p))
    add_path_row("Tokens/Jahr:", "tokens", lambda: DATA.path_tokens_year, lambda p: setattr(DATA, 'path_tokens_year', p))
    add_path_row("TopDocs:", "topdocs", lambda: DATA.path_global_topdocs, lambda p: setattr(DATA, 'path_global_topdocs', p))
    add_path_row("Topics-Dist:", "topics", lambda: DATA.path_topics_dist, lambda p: setattr(DATA, 'path_topics_dist', p))
    add_path_row("Metadata:", "metadata", lambda: DATA.path_metadata, lambda p: setattr(DATA, 'path_metadata', p))
    
    # Status-Anzeige
    row += 1
    info = tk.Text(frame, height=10, width=100, font=("Courier", 9))
    info.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    def check_status(name: str, loader) -> str:
        try:
            df = loader()
            return f"✅ {name}: {df.shape[0]:,} × {df.shape[1]}"
        except Exception as e:
            return f"❌ {name}: {e}"
    
    def load_and_check():
        info.delete(1.0, tk.END)
        info.insert(tk.END, "🔄 Lade Dateien...\n\n")
        root.update_idletasks()
        
        DATA.invalidate_cache()
        
        results = [
            check_status("Termset", DATA.load_termset),
            check_status("Topic-Words", DATA.load_topic_words),
            check_status("TF-IDF", DATA.load_tfidf),
            check_status("Rankings", DATA.load_ranks),
            check_status("Relevanz", DATA.load_relevance),
            check_status("Counts/Jahr", DATA.load_counts_per_year),
            check_status("Top10/Jahr", DATA.load_top10_year_value),
            check_status("Top10/Text", DATA.load_top10_value_per_text),
            check_status("Tokens/Jahr", DATA.load_tokens_year),
            check_status("TopDocs", DATA.load_global_topdocs),
            check_status("Topics-Dist", DATA.load_topics_dist),
            check_status("Metadata", DATA.load_metadata),
        ]
        
        info.delete(1.0, tk.END)
        info.insert(tk.END, "\n".join(results))
        
        ok = sum(1 for r in results if r.startswith("✅"))
        info.insert(tk.END, f"\n\n{'─'*40}\n")
        info.insert(tk.END, f"Zusammenfassung: {ok}/{len(results)} geladen\n")
        
        # Zeige Metadaten-Analyse für TF-IDF wenn geladen
        try:
            df_tfidf = DATA.load_tfidf()
            meta_cols = METADATA_DETECTOR.detect(df_tfidf)
            term_cols = METADATA_DETECTOR.get_term_columns(df_tfidf)
            info.insert(tk.END, f"\n{'─'*40}\n")
            info.insert(tk.END, f"TF-IDF Spaltenanalyse:\n")
            info.insert(tk.END, f"  • Metadaten: {len(meta_cols)} Spalten\n")
            info.insert(tk.END, f"  • Terme: {len(term_cols)} Spalten\n")
        except Exception:
            pass
    
    def sync_paths():
        """Synchronisiert Entry-Felder mit DataManager."""
        path_vars["termset"].set(str(DATA.path_termset))
        path_vars["topic_words"].set(str(DATA.path_topic_words))
        path_vars["tfidf"].set(str(DATA.path_tfidf))
        path_vars["ranks"].set(str(DATA.path_ranks))
        path_vars["relevance"].set(str(DATA.path_relevance))
        path_vars["counts_year"].set(str(DATA.path_counts_per_year))
        path_vars["top10_year"].set(str(DATA.path_top10_year_value))
        path_vars["top10_text"].set(str(DATA.path_top10_value_per_text))
        path_vars["tokens"].set(str(DATA.path_tokens_year))
        path_vars["topdocs"].set(str(DATA.path_global_topdocs))
        path_vars["topics"].set(str(DATA.path_topics_dist))
        path_vars["metadata"].set(str(DATA.path_metadata))
        
        info.delete(1.0, tk.END)
        info.insert(tk.END, "🔄 Pfade synchronisiert.\n\nKlicken Sie 'Laden & Prüfen'.\n")
    
    # Buttons
    row += 1
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=6)
    
    ttk.Button(btn_frame, text="↻ Pfade synchronisieren", command=sync_paths).pack(side="left", padx=(0, 10))
    ttk.Button(btn_frame, text="Laden & Prüfen", command=load_and_check).pack(side="left")


def build_tab_column_analysis(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """
    Tab: Spaltenanalyse
    
    Zeigt detaillierte Analyse der Spalten einer ausgewählten Datei,
    inkl. automatischer Metadaten-Erkennung.
    """
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="🔍 Spaltenanalyse")
    
    # Datei-Auswahl
    row = 0
    ttk.Label(frame, text="Datei auswählen:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    
    file_options = [
        ("TF-IDF", "tfidf"),
        ("Metadata", "metadata"),
        ("Termset", "termset"),
        ("Rankings", "ranks"),
        ("Counts/Jahr", "counts_year"),
        ("Topics-Dist", "topics_dist"),
    ]
    
    file_var = tk.StringVar(value="tfidf")
    file_combo = ttk.Combobox(
        frame, 
        textvariable=file_var, 
        values=[opt[0] for opt in file_options],
        state="readonly",
        width=20
    )
    file_combo.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    file_combo.current(0)
    
    row += 1
    
    # Ergebnis-Anzeige
    result_frame = ttk.LabelFrame(frame, text="Spaltenanalyse", padding=8)
    result_frame.grid(row=row, column=0, columnspan=4, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    frame.columnconfigure(1, weight=1)
    
    # Treeview für Spalten
    columns = ("Spalte", "Typ", "Kategorie", "Unique", "Nulls", "Num%", "Beispiele")
    tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=15)
    
    col_widths = [150, 80, 100, 70, 70, 70, 250]
    for col, width in zip(columns, col_widths):
        tree.heading(col, text=col)
        tree.column(col, width=width, anchor="w")
    
    tree.pack(side="left", fill="both", expand=True)
    
    scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    
    row += 1
    
    # Zusammenfassung
    summary_var = tk.StringVar(value="Wählen Sie eine Datei und klicken Sie 'Analysieren'.")
    ttk.Label(frame, textvariable=summary_var, wraplength=600).grid(
        row=row, column=0, columnspan=4, sticky="w", padx=6, pady=4
    )
    
    def analyze():
        # Treeview leeren
        for item in tree.get_children():
            tree.delete(item)
        
        # Datei-Mapping
        file_map = {
            "TF-IDF": DATA.load_tfidf,
            "Metadata": DATA.load_metadata,
            "Termset": DATA.load_termset,
            "Rankings": DATA.load_ranks,
            "Counts/Jahr": DATA.load_counts_per_year,
            "Topics-Dist": DATA.load_topics_dist,
        }
        
        selected = file_var.get()
        loader = file_map.get(selected)
        
        if not loader:
            messagebox.showerror("Fehler", "Unbekannte Datei.", parent=root)
            return
        
        try:
            df = loader()
        except Exception as e:
            messagebox.showerror("Fehler", f"Laden fehlgeschlagen:\n{e}", parent=root)
            return
        
        # Analyse durchführen
        analysis = METADATA_DETECTOR.analyze(df)
        
        meta_count = 0
        term_count = 0
        
        for col, info in analysis.items():
            category = "Metadaten" if info['is_metadata'] else "Term"
            if info['is_metadata']:
                meta_count += 1
            else:
                term_count += 1
            
            examples = ", ".join(str(v)[:20] for v in info['sample_values'][:3])
            
            tree.insert("", "end", values=(
                col,
                info['dtype'],
                category,
                info['unique_count'],
                info['null_count'],
                f"{info['numeric_ratio']*100:.0f}%",
                examples
            ))
        
        summary_var.set(
            f"Analysiert: {len(df)} Zeilen × {len(df.columns)} Spalten | "
            f"Metadaten: {meta_count} | Terme: {term_count}"
        )
    
    row += 1
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=6, pady=6)
    
    ttk.Button(btn_frame, text="🔍 Analysieren", command=analyze).pack(side="left", padx=(0, 10))
    
    # Export der Analyse
    def export_analysis():
        # Sammle Daten aus Treeview
        rows = []
        for item in tree.get_children():
            rows.append(tree.item(item)['values'])
        
        if not rows:
            messagebox.showinfo("Info", "Keine Daten zum Exportieren.", parent=root)
            return
        
        df_export = pd.DataFrame(rows, columns=columns)
        save_dataframe(df_export, "Analysis", f"spaltenanalyse_{file_var.get()}", root)
    
    ttk.Button(btn_frame, text="📥 Als CSV exportieren", command=export_analysis).pack(side="left")


def build_tab_bubbles(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """
    Tab: Tag-Topic-Relevanz (Bubble-Chart)
    
    Visualisiert die Überlappung zwischen Tags und Topics basierend auf
    TF-IDF-Werten der gemeinsamen Terme.
    """
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="TT-Relevanz (Bubbles)")
    
    # Parameter
    row = 0
    ttk.Label(frame, text="Top-N Topics:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_topn = create_entry(frame, width=8)
    ent_topn.insert(0, "10")
    ent_topn.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    ttk.Label(frame, text="Max. Bubble-Größe:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_maxsize = create_entry(frame, width=8)
    ent_maxsize.insert(0, "auto")
    ent_maxsize.grid(row=row, column=3, sticky="w", padx=6, pady=4)
    
    row += 1
    ttk.Label(frame, text="Min. Bubble-Größe:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_minsize = create_entry(frame, width=8)
    ent_minsize.insert(0, "20")
    ent_minsize.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    show_values_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame, text="Werte anzeigen", variable=show_values_var).grid(
        row=row, column=2, sticky="w", padx=6, pady=4
    )
    
    # Ergebnis-Speicher
    result_data = {"df": None, "fig": None, "ctx": ""}
    
    row += 1
    btn_csv = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled")
    btn_csv.grid(row=row, column=2, sticky="e", padx=6, pady=6)
    btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            df_tags = DATA.load_termset()
            df_topics = DATA.load_topic_words()
            df_ranks = DATA.load_ranks()
            tfidf_sums = compute_tfidf_sums()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        # Parameter parsen
        try:
            top_n = max(1, int(ent_topn.get() or "10"))
            min_size = float(ent_minsize.get() or "20")
            max_size_str = (ent_maxsize.get() or "auto").strip().lower()
        except ValueError:
            messagebox.showerror("Fehler", "Ungültige Parameter", parent=root)
            return
        
        # Top-N Topics ermitteln
        available_topics = set(df_topics.index.astype(str))
        top_topics = get_ranked_topics(df_ranks, available_topics, top_n)
        
        if not top_topics:
            # Debug-Info sammeln
            rank_topics = set(df_ranks['Topic'].astype(str).str.strip().tolist()) if 'Topic' in df_ranks.columns else set()
            common = available_topics.intersection(rank_topics)
            
            debug_msg = (
                f"Keine Topics gefunden.\n\n"
                f"Debug-Info:\n"
                f"• Topics in topic_words: {len(available_topics)}\n"
                f"• Topics in rankings: {len(rank_topics)}\n"
                f"• Gemeinsame Topics: {len(common)}\n\n"
                f"Beispiele topic_words: {list(available_topics)[:3]}\n"
                f"Beispiele rankings: {list(rank_topics)[:3]}"
            )
            messagebox.showerror("Fehler", debug_msg, parent=root)
            return
        
        # Daten berechnen
        tag_dict = get_tag_dict(df_tags)
        topic_map = get_topic_word_map(df_topics)
        tfidf_dict = tfidf_sums.to_dict()
        
        rows = []
        for topic in top_topics:
            topic_words = set(topic_map[topic])
            for tag, expressions in tag_dict.items():
                common = topic_words.intersection(expressions)
                value = sum(tfidf_dict.get(w, 0.0) for w in common)
                rows.append({"Topic": topic, "Tag": tag, "value": value})
        
        df_result = pd.DataFrame(rows)
        
        # Topic-Labels bereinigen
        df_result["Topic_Label"] = df_result["Topic"].str.split("(", n=1).str[0].str.strip()
        
        # Anzahl Topics/Tags für Größenberechnung
        n_topics = len(top_topics)
        n_tags = len(tag_dict)
        
        # Automatische Max-Größe
        if max_size_str == "auto":
            fig_width = max(10, n_topics * 1.2)
            fig_height = max(6, n_tags * 0.4)
            cell_size = min(fig_width / n_topics, fig_height / n_tags) * 72 * 0.6
            max_size = min(max(min_size * 2, cell_size ** 2), 2000)
        else:
            max_size = float(max_size_str)
            fig_width = max(10, n_topics * 1.2)
            fig_height = max(6, n_tags * 0.4)
        
        # Bubble-Größen normalisieren
        values = df_result["value"].clip(lower=0).values
        if values.max() > values.min():
            normalized = (values - values.min()) / (values.max() - values.min())
        else:
            normalized = np.zeros_like(values)
        
        sizes = min_size + normalized * (max_size - min_size)
        sizes = np.where(values == 0, min_size * 0.3, sizes)
        
        # Plot erstellen
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        scatter = ax.scatter(
            df_result["Topic_Label"],
            df_result["Tag"],
            s=sizes,
            c=values,
            cmap="YlOrRd",
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5
        )
        
        plt.colorbar(scatter, ax=ax, shrink=0.8, label="TF-IDF Summe")
        
        if show_values_var.get():
            for x, y, v in zip(df_result["Topic_Label"], df_result["Tag"], values):
                if v > 0:
                    ax.annotate(f"{v:.0f}", (x, y), ha='center', va='center', fontsize=6)
        
        ax.set_xlabel("Topic")
        ax.set_ylabel("Tag")
        ax.set_title(f"Top {n_topics} Topics – Tag-Topic-Relevanz")
        
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Abstand an den Rändern erhöhen, damit Bubbles nicht abgeschnitten werden
        ax.set_xlim(-0.5, n_topics - 0.5)
        ax.set_ylim(-0.5, n_tags - 0.5)
        ax.margins(x=0.08, y=0.08)  # 8% zusätzlicher Rand
        
        apply_figure_layout(fig)
        plt.show()
        
        # Ergebnisse speichern
        result_data["df"] = df_result
        result_data["fig"] = fig
        result_data["ctx"] = f"bubbles_top{n_topics}"
        
        btn_csv.configure(state="normal")
        btn_png.configure(state="normal")
    
    def save_csv():
        save_dataframe(result_data["df"], "Bubbles", result_data["ctx"], root)
    
    def save_png():
        if result_data["fig"]:
            save_figure(result_data["fig"], "Bubbles", result_data["ctx"], root)
    
    btn_csv.configure(command=save_csv)
    btn_png.configure(command=save_png)
    
    ttk.Button(frame, text="Berechnen", command=compute).grid(
        row=row, column=0, sticky="w", padx=6, pady=6
    )


def build_tab_stacked(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """
    Tab: Topic-Texte pro Jahr (Stacked Bar Chart)
    
    Zeigt die Anzahl der Texte pro Topic über die Zeit als gestapeltes
    Balkendiagramm.
    """
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Topics/Jahr (Stacked)")
    
    row = 0
    ttk.Label(frame, text="Zeitraum (YYYY-YYYY):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_years = create_entry(frame, width=12)
    ent_years.insert(0, "1780-1900")
    ent_years.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    ttk.Label(frame, text="Top-N Topics:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_topn = create_entry(frame, width=8)
    ent_topn.insert(0, "10")
    ent_topn.grid(row=row, column=3, sticky="w", padx=6, pady=4)
    
    result_data = {"fig": None, "ctx": ""}
    
    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled")
    btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            df_counts = DATA.load_counts_per_year()
            df_ranks = DATA.load_ranks()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        # Parameter
        try:
            top_n = max(1, int(ent_topn.get() or "10"))
        except ValueError:
            top_n = 10
        
        # Daten vorbereiten
        df = df_counts.drop(columns=["Anzahl Topics"], errors="ignore")
        df.index = df.index.astype(int)
        
        year_min, year_max = parse_year_range(
            int(df.index.min()), 
            int(df.index.max()), 
            ent_years.get()
        )
        
        df = df.reindex(range(year_min, year_max + 1), fill_value=0)
        
        # Top-N Topics ermitteln
        available = set(df.columns.astype(str))
        topics = get_ranked_topics(df_ranks, available, top_n)
        
        if not topics:
            messagebox.showerror("Fehler", "Keine Topics gefunden.", parent=root)
            return
        
        df_plot = df[topics]
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(topics)))
        
        df_plot.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.85)
        
        # Gleitender Mittelwert
        rolling = df_plot.sum(axis=1).rolling(window=5, center=True, min_periods=1).mean()
        ax.plot(range(len(df_plot)), rolling.values, color="black", linestyle="--", 
                linewidth=1.5, alpha=0.7, label="Gleitender MW (5J)")
        
        # X-Achse: nur jedes 10. Jahr beschriften
        tick_positions = [i for i, y in enumerate(df_plot.index) if y % 10 == 0]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([df_plot.index[i] for i in tick_positions], rotation=45, ha='right')
        
        ax.set_xlabel("Jahr")
        ax.set_ylabel("Anzahl Texte")
        ax.set_title(f"Top {len(topics)} Topics – Texte pro Jahr")
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8)
        
        apply_figure_layout(fig)
        plt.show()
        
        result_data["fig"] = fig
        result_data["ctx"] = f"stacked_{year_min}_{year_max}_top{len(topics)}"
        btn_png.configure(state="normal")
    
    def save_png():
        if result_data["fig"]:
            save_figure(result_data["fig"], "Stacked", result_data["ctx"], root)
    
    btn_png.configure(command=save_png)
    
    ttk.Button(frame, text="Plotten", command=compute).grid(
        row=row, column=0, sticky="w", padx=6, pady=6
    )


def build_tab_trends(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """
    Tab: Topic-Verläufe
    
    Zeigt zeitliche Entwicklung ausgewählter Topics mit verschiedenen
    Visualisierungsoptionen (absolut, geglättet, Polynom).
    """
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Topic-Verläufe")
    
    row = 0
    
    # Parameter
    ttk.Label(frame, text="Cosinus-Schwelle:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_threshold = create_entry(frame, width=8)
    ent_threshold.insert(0, "0.2")
    ent_threshold.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    ttk.Label(frame, text="MA-Fenster:").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_ma = create_entry(frame, width=8)
    ent_ma.insert(0, "3")
    ent_ma.grid(row=row, column=3, sticky="w", padx=6, pady=4)
    
    ttk.Label(frame, text="Polynom-Grad:").grid(row=row, column=4, sticky="w", padx=6, pady=4)
    ent_degree = create_entry(frame, width=8)
    ent_degree.insert(0, "3")
    ent_degree.grid(row=row, column=5, sticky="w", padx=6, pady=4)
    
    row += 1
    
    # Checkboxen
    var_absolute = tk.BooleanVar(value=False)
    var_smooth = tk.BooleanVar(value=True)
    var_poly = tk.BooleanVar(value=True)
    
    ttk.Checkbutton(frame, text="Absolut", variable=var_absolute).grid(
        row=row, column=0, sticky="w", padx=6, pady=2
    )
    ttk.Checkbutton(frame, text="Geglättet", variable=var_smooth).grid(
        row=row, column=1, sticky="w", padx=6, pady=2
    )
    ttk.Checkbutton(frame, text="Polynom", variable=var_poly).grid(
        row=row, column=2, sticky="w", padx=6, pady=2
    )
    
    row += 1
    
    # Topic-Auswahl
    ttk.Label(frame, text="Topics auswählen:").grid(row=row, column=0, sticky="nw", padx=6, pady=4)
    
    listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, width=60, height=12, exportselection=False)
    listbox.grid(row=row, column=1, columnspan=5, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    scrollbar.grid(row=row, column=6, sticky="ns")
    
    figures: Dict[str, plt.Figure] = {}
    
    def load_topics(show_error: bool = True):
        listbox.delete(0, tk.END)
        try:
            df = DATA.load_topics_dist()
            for col in sorted(df.columns.tolist()):
                listbox.insert(tk.END, col)
        except Exception as e:
            if show_error:
                messagebox.showerror("Fehler", str(e), parent=root)
            else:
                listbox.insert(tk.END, "(Bitte Dateien laden)")
    
    load_topics(show_error=False)
    
    def compute():
        figures.clear()
        
        try:
            df_topics = DATA.load_topics_dist()
            df_metadata = DATA.load_metadata()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        # Parameter
        try:
            threshold = float(ent_threshold.get() or "0.2")
            ma_window = max(1, int(ent_ma.get() or "3"))
            degree = max(1, int(ent_degree.get() or "3"))
        except ValueError:
            messagebox.showerror("Fehler", "Ungültige Parameter", parent=root)
            return
        
        # Ausgewählte Topics
        selected_indices = listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Fehler", "Keine Topics ausgewählt.", parent=root)
            return
        
        selected_topics = [listbox.get(i) for i in selected_indices]
        
        # Jahr-Mapping
        df_metadata['_id_norm'] = df_metadata['_id'].map(normalize_document_id)
        
        df = df_topics.copy()
        df['_id_norm'] = df.index.astype(str).map(normalize_document_id)
        
        year_map = dict(zip(df_metadata['_id_norm'], df_metadata['Jahr_final']))
        df['Jahr'] = df['_id_norm'].map(year_map)
        
        df = df.dropna(subset=['Jahr'])
        df['Jahr'] = df['Jahr'].astype(int)
        df = df[df['Jahr'] >= 1800]
        
        if df.empty:
            messagebox.showerror("Fehler", "Keine Daten nach Jahr-Mapping.", parent=root)
            return
        
        # Aggregieren
        df_grouped = df.groupby('Jahr')[selected_topics].mean().fillna(0)
        
        # Plots
        if var_absolute.get():
            fig, ax = plt.subplots(figsize=(12, 6))
            for topic in selected_topics:
                ax.plot(df_grouped.index, df_grouped[topic], label=topic)
            ax.set_xlabel('Jahr')
            ax.set_ylabel('Durchschnittliche Cosinus-Ähnlichkeit')
            ax.set_title('Absolute Topic-Verläufe')
            ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8)
            apply_figure_layout(fig)
            plt.show()
            figures["absolut"] = fig
        
        if var_smooth.get():
            fig, ax = plt.subplots(figsize=(12, 6))
            for topic in selected_topics:
                smoothed = df_grouped[topic].rolling(window=ma_window, center=True, min_periods=1).mean()
                ax.plot(df_grouped.index, smoothed, label=topic)
            ax.set_xlabel('Jahr')
            ax.set_ylabel('Durchschnittliche Cosinus-Ähnlichkeit')
            ax.set_title(f'Geglättete Topic-Verläufe (MA={ma_window})')
            ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8)
            apply_figure_layout(fig)
            plt.show()
            figures["smooth"] = fig
        
        # Schwellen-Plot (immer)
        fig, ax = plt.subplots(figsize=(12, 6))
        for topic in selected_topics:
            counts = df.groupby('Jahr')[topic].apply(lambda x: (x >= threshold).sum())
            ax.plot(counts.index, counts.values, label=topic)
        ax.set_xlabel('Jahr')
        ax.set_ylabel(f'Anzahl Texte (Cosinus ≥ {threshold})')
        ax.set_title(f'Relevante Dokumente pro Jahr (Schwelle={threshold})')
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8)
        apply_figure_layout(fig)
        plt.show()
        figures["threshold"] = fig
        
        if var_poly.get():
            fig, ax = plt.subplots(figsize=(12, 6))
            for topic in selected_topics:
                years = df_grouped.index.values.astype(float)
                values = df_grouped[topic].values
                if len(years) > degree:
                    coeffs = np.polyfit(years, values, degree)
                    poly_values = np.polyval(coeffs, years)
                    ax.plot(years, poly_values, label=topic)
            ax.set_xlabel('Jahr')
            ax.set_ylabel('Durchschnittliche Cosinus-Ähnlichkeit')
            ax.set_title(f'Polynomiale Regression (Grad={degree})')
            ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8)
            apply_figure_layout(fig)
            plt.show()
            figures["poly"] = fig
        
        btn_save.configure(state="normal")
    
    def save_all():
        if not figures:
            messagebox.showinfo("Info", "Keine Plots vorhanden.", parent=root)
            return
        
        save_dir = EXPLORATION_DIR / "Trends"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved = []
        for name, fig in figures.items():
            path = save_dir / f"trend_{name}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            saved.append(path.name)
        
        messagebox.showinfo("Gespeichert", f"{len(saved)} Dateien in:\n{save_dir}", parent=root)
    
    row += 1
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=row, column=0, columnspan=6, sticky="w", padx=6, pady=8)
    
    ttk.Button(btn_frame, text="Berechnen", command=compute).pack(side="left", padx=(0, 10))
    ttk.Button(btn_frame, text="🔄 Topics laden", command=lambda: load_topics(True)).pack(side="left", padx=(0, 10))
    btn_save = ttk.Button(btn_frame, text="Alle speichern", state="disabled", command=save_all)
    btn_save.pack(side="left")


def build_tab_comparison(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """
    Tab: Vergleich Tokens vs. Topics
    
    Vergleicht normalisierte Zeitreihen von Token-Anzahl und
    Topic-Verteilung.
    """
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Tokens vs. Topics")
    
    row = 0
    ttk.Label(frame, text="Zeitraum (YYYY-YYYY):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_years = create_entry(frame, width=12)
    ent_years.insert(0, "1780-1900")
    ent_years.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    result_data = {"fig": None}
    
    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled")
    btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    
    def normalize(series: pd.Series) -> pd.Series:
        """Min-Max-Normalisierung."""
        s = pd.to_numeric(series, errors='coerce').fillna(0)
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return s * 0
    
    def compute():
        try:
            df_tokens = DATA.load_tokens_year()
            df_topdocs = DATA.load_global_topdocs()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        # Spalten finden
        year_col_t = find_column(df_tokens, ["year", "Jahr", "Year"])
        tokens_col = find_column(df_tokens, ["anzahl_tokens", "tokens", "count"])
        year_col_d = find_column(df_topdocs, ["Jahr", "year", "Year"])
        value_col = find_column(df_topdocs, ["Wert", "value", "Value"])
        
        if not all([year_col_t, tokens_col, year_col_d, value_col]):
            messagebox.showerror("Fehler", "Erforderliche Spalten nicht gefunden.", parent=root)
            return
        
        # Normalisieren
        df_tokens = df_tokens.rename(columns={year_col_t: "year", tokens_col: "tokens"})
        df_topdocs = df_topdocs.rename(columns={year_col_d: "year", value_col: "value"})
        
        df_tokens = df_tokens.sort_values("year")
        df_topdocs = df_topdocs.sort_values("year")
        
        # Glättung
        df_tokens["smooth"] = df_tokens["tokens"].rolling(5, center=True, min_periods=1).mean()
        df_topdocs["smooth"] = df_topdocs["value"].rolling(5, center=True, min_periods=1).mean()
        
        df_tokens["norm"] = normalize(df_tokens["smooth"])
        df_topdocs["norm"] = normalize(df_topdocs["smooth"])
        
        # Zeitraum
        year_min, year_max = parse_year_range(
            int(min(df_tokens["year"].min(), df_topdocs["year"].min())),
            int(max(df_tokens["year"].max(), df_topdocs["year"].max())),
            ent_years.get()
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(df_tokens["year"], df_tokens["norm"], label="Tokens (normalisiert)", 
                linestyle="--", linewidth=1.2)
        ax.plot(df_topdocs["year"], df_topdocs["norm"], label="TopDocs (normalisiert)", 
                linestyle="-", linewidth=1.5)
        
        ax.set_xlabel("Jahr")
        ax.set_ylabel("Normalisierter Wert (0–1)")
        ax.set_title("Vergleich: Tokens vs. Topic-Verteilung")
        ax.set_xticks(np.arange(year_min, year_max + 1, 10))
        ax.legend()
        
        apply_figure_layout(fig)
        plt.show()
        
        result_data["fig"] = fig
        btn_png.configure(state="normal")
    
    def save_png():
        if result_data["fig"]:
            save_figure(result_data["fig"], "Comparison", "tokens_vs_topics", root)
    
    btn_png.configure(command=save_png)
    
    ttk.Button(frame, text="Plotten", command=compute).grid(
        row=row, column=0, sticky="w", padx=6, pady=6
    )


def build_tab_tt_texts_poly(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """
    Tab: TT-Texts/Jahr (Polynomiale Regression)
    
    Zeigt Topic-Counts pro Jahr als polynomiale Trendlinien für
    die Top-N Topics nach TFIDF-Positions-Rang.
    """
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="TT-Texts/Jahr (Poly)")
    
    row = 0
    ttk.Label(frame, text="Polynom-Grad:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ent_deg = create_entry(frame, width=8)
    ent_deg.insert(0, "6")
    ent_deg.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    
    ttk.Label(frame, text="Top-N (Rang):").grid(row=row, column=2, sticky="w", padx=6, pady=4)
    ent_topn = create_entry(frame, width=8)
    ent_topn.insert(0, "10")
    ent_topn.grid(row=row, column=3, sticky="w", padx=6, pady=4)
    
    result_data = {"fig": None, "ctx": ""}
    
    row += 1
    btn_png = ttk.Button(frame, text="PNG speichern", state="disabled")
    btn_png.grid(row=row, column=3, sticky="e", padx=6, pady=6)
    
    def compute():
        try:
            df_counts = DATA.load_counts_per_year()
            df_ranks = DATA.load_ranks()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        # Daten vorbereiten
        df = df_counts.drop(columns=["Anzahl Topics"], errors="ignore")
        df.index = df.index.astype(int)
        df = df.reindex(range(df.index.min(), df.index.max() + 1), fill_value=0)
        
        # Parameter
        try:
            top_n = max(1, int(ent_topn.get() or "10"))
        except ValueError:
            top_n = 10
        
        # Top-N Topics ermitteln
        ranked_topics = get_ranked_topics_for_counts(df, df_ranks, top_n)
        
        if not ranked_topics:
            messagebox.showerror("Fehler", "Keine Topics gefunden.", parent=root)
            return
        
        x = df.index.values.astype(float)
        
        try:
            deg = int(ent_deg.get() or "6")
            deg = max(1, min(deg, len(x) - 1))
        except ValueError:
            deg = 6
        
        # Plot
        colors = plt.cm.tab10(np.linspace(0, 1, len(ranked_topics)))
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, topic in enumerate(ranked_topics):
            if topic not in df.columns:
                continue
            y = df[topic].values.astype(float)
            if len(np.unique(y)) < 2:
                continue
            
            coeffs = np.polyfit(x, y, deg)
            y_poly = np.polyval(coeffs, x)
            
            label = topic.replace("(", "\n(") if "(" in topic else topic
            ax.plot(x, y_poly, label=label, color=colors[i], linewidth=1.2)
        
        ax.set_xticks([int(j) for j in x if int(j) % 10 == 0])
        ax.set_xlabel("Jahr")
        ax.set_ylabel("Anzahl TT-Texts pro Topic")
        ax.set_title(f"TT-Texts/Jahr (Polynom Grad {deg}) – Top-{len(ranked_topics)} nach Rang")
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8)
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        
        apply_figure_layout(fig)
        plt.show()
        
        result_data["fig"] = fig
        result_data["ctx"] = f"tt_texts_poly_deg{deg}_top{len(ranked_topics)}"
        btn_png.configure(state="normal")
    
    def save_png():
        if result_data["fig"]:
            save_figure(result_data["fig"], "TT_Texts_Poly", result_data["ctx"], root)
    
    btn_png.configure(command=save_png)
    
    ttk.Button(frame, text="Plotten", command=compute).grid(
        row=row, column=0, sticky="w", padx=6, pady=6
    )


def build_tab_tt_texts_rank(notebook: ttk.Notebook, root: tk.Tk) -> None:
    """
    Tab: TT-Texts-Rang
    
    Zeigt Top-Texte pro Topic mit Rang und optionalem Metadata-Mapping.
    """
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="TT-Texts-Rang")
    
    row = 0
    
    # Treeview für Ergebnisse
    cols = ("_id", "text", "rank")
    col_widths = [100, 550, 80]
    
    tree = ttk.Treeview(frame, columns=cols, show="headings", height=20)
    for col, width in zip(cols, col_widths):
        tree.heading(col, text=col)
        tree.column(col, width=width, anchor="w")
    
    tree.grid(row=row, column=0, columnspan=5, sticky="nsew", padx=6, pady=6)
    frame.rowconfigure(row, weight=1)
    frame.columnconfigure(0, weight=1)
    
    # Scrollbars
    h_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(xscrollcommand=h_scroll.set)
    h_scroll.grid(row=row+1, column=0, columnspan=5, sticky="ew")
    
    v_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=v_scroll.set)
    v_scroll.grid(row=row, column=5, sticky="ns")
    
    result_data = {"df": None}
    
    row += 2
    btn_csv = ttk.Button(frame, text="CSV speichern", state="disabled")
    btn_csv.grid(row=row, column=4, sticky="e", padx=6, pady=6)
    
    # Info-Label
    info_var = tk.StringVar(value="")
    ttk.Label(frame, textvariable=info_var).grid(row=row, column=1, columnspan=2, sticky="w", padx=6)
    
    def compute():
        try:
            df = DATA.load_top10_value_per_text()
        except Exception as e:
            messagebox.showerror("Fehler", str(e), parent=root)
            return
        
        # Metadata optional laden
        try:
            metadata_df = DATA.load_metadata()
            has_metadata = True
        except Exception:
            metadata_df = None
            has_metadata = False
        
        if df.shape[1] < 2:
            messagebox.showerror("Fehler", "Datei benötigt mind. 2 Spalten.", parent=root)
            return
        
        # Spalten identifizieren
        text_col = df.columns[0]
        value_col = df.columns[1]
        
        topic_col = None
        for cand in ["Topic", "topic", "topic_label"]:
            if cand in df.columns:
                topic_col = cand
                break
        
        work = df.copy()
        work[value_col] = pd.to_numeric(work[value_col], errors='coerce').fillna(0.0)
        
        # Top-30 pro Topic oder global
        if topic_col:
            work = (work.sort_values([topic_col, value_col], ascending=[True, False])
                       .groupby(topic_col, group_keys=False)
                       .head(30))
            work["rank"] = work.groupby(topic_col)[value_col].rank(method="min", ascending=False).astype(int)
        else:
            work = work.sort_values(value_col, ascending=False).head(30)
            work["rank"] = work[value_col].rank(method="min", ascending=False).astype(int)
        
        # Metadata-Mapping
        work["_id"] = ""
        matched_count = 0
        
        if has_metadata and metadata_df is not None:
            for idx, row_data in work.iterrows():
                text_value = str(row_data[text_col])
                matched_id = match_text_to_metadata(text_value, metadata_df)
                if matched_id:
                    work.at[idx, "_id"] = matched_id
                    matched_count += 1
        
        df_result = work.rename(columns={text_col: "text"})[["_id", "text", "rank"]]
        df_result = df_result.sort_values(["rank", "text"])
        
        # Treeview aktualisieren
        tree.delete(*tree.get_children())
        for _, r in df_result.iterrows():
            tree.insert("", "end", values=(
                str(r["_id"]),
                str(r["text"]),
                int(r["rank"])
            ))
        
        result_data["df"] = df_result
        btn_csv.configure(state="normal")
        
        # Info anzeigen
        info_var.set(f"Geladen: {len(df_result)} Einträge | Metadata-Mapping: {matched_count}/{len(df_result)}")
    
    def save_csv():
        if result_data["df"] is not None:
            save_dataframe(result_data["df"], "TT_Texts_Rang", "text_rank_top30", root)
    
    btn_csv.configure(command=save_csv)
    
    ttk.Button(frame, text="Laden/Anzeigen", command=compute).grid(
        row=row, column=0, sticky="w", padx=6, pady=6
    )


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def main():
    """Startet die Anwendung."""
    root = tk.Tk()
    root.title("Tag-Topic-Explorer")
    root.geometry("1200x700")
    
    setup_window(root)
    
    # Hauptnotebook
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=5, pady=5)
    
    # Verwaltungs-Tabs
    build_tab_workspace(notebook, root)
    build_tab_data(notebook, root)
    build_tab_column_analysis(notebook, root)
    
    # Visualisierungs-Tabs in eigenem Notebook
    viz_frame = ttk.Frame(notebook)
    notebook.add(viz_frame, text="📈 Visualisierungen")
    
    viz_notebook = ttk.Notebook(viz_frame)
    viz_notebook.pack(fill="both", expand=True)
    
    build_tab_bubbles(viz_notebook, root)
    build_tab_stacked(viz_notebook, root)
    build_tab_tt_texts_poly(viz_notebook, root)
    build_tab_trends(viz_notebook, root)
    build_tab_comparison(viz_notebook, root)
    build_tab_tt_texts_rank(viz_notebook, root)
    
    root.mainloop()


if __name__ == "__main__":
    main()
