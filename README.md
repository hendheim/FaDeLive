# Projekttitel: FaDeLive (1782-1891)

Herausgeber: Hendrick Heimböckel, [ORCID-ID: 0000-0002-4211-9769](https://orcid.org/0000-0002-4211-9769)
<br><br>
## 1. Projektübersicht
_Fachlichkeit des Deutschunterrichts: Literaturvermittlung 1782-1891_ (FaDeLive: 1782-1891) ist ein Korpus mit 1022 Texten zu Literaturvermittlung an höheren Schulen in Preußen im 19. Jahrhundert. Es ist die Grundlage des Projektes *"Lesen heißt diese Übung" – und dann? Begriffe und intendierte Praktiken der Vermittlung deutschsprachiger Literatur an höheren Schulen 1796–1890*. 

Das Korpus setzt sich aus fotomechanisch gesannten Texten zusammen, die mit OCR (<https://github.com/OCR4all>, <https://github.com/tesseract-ocr/tesseract>) verarbeitet und in Hinblick auf OCR-Fehler gesäubert wurden. Diese Verarbeitungsstufe *TXT (content)* ist die Grundlage für die weitere Prozessierung der Daten. Siehe zur Vorverarbeitung und Verarbeitung des Korpus folgenden Workflow: <https://doi.org/10.48693/730>

Die Funktionen des Repositorium sind Reproduzierbarkeit, Nachnutzung und Weiterentwicklung des Korpus, seiner Datafizierung sowie Visualisierung. 
 
Es umfasst 
- das Korpus mit textspezifischen bibliographischen Angaben,
- die Pipeline, Tools und Ressourcen zur Vorverarbeitung sowie für die Verarbeitung des Korpus,
- ein GUI-Tool zur Exploration und Visualisierung des verarbeiteten Korpus. 

Die Pipeline und weiteren Tools zur Verarbeitung, Exploration und Visualisierung können mit Anpassungen auch für andere Korpora verwendet werden. 

**Schlagwörter**: 
- Computerlinguistik
- Didaktik des Deutschunterrichts
- Literaturunterricht
- Historische Bildungsforschung
- Text Mining
- Computational Discourse Analyses
<br><br>
## 2. Datengrundlage
Die Pipeline erwartet ein Korpus in folgendem Format: <br>
**Datei**: `data/raw/korpus.csv`<br>
**Encoding**: utf-8<br>
**Trennzeichen**: Semikolon<br>
**Notwendige Metadaten in _korpus.csv_ für die Verarbeitung der Texte in**: 
- `content`: maschinenlesbarer Text in der Verarbeitungsstufe _TXT (content)_
- `_id`: ID der Dokumente	
- `author_surname`: Nachname der Verfsser, der Verfasserin
- `genre`: Genre eines Textes
- `textclass`: allgemeine Gattung eines Textes
- `year`: Veröffentlichungsjahr der Zeitschrift oder des Bandes, in dem der Text veröffentlicht wurde, oder Veröffentlichungsjahr der Auflage
- `title`: Titel des Textes
- `year_first`: Veröffentlichungsjahr der Erstausgabe eines Textes
- `source`: Name der Zeitschrift oder des Bandes in dem der Text veröffentlicht wurde
- `year_first`: Veröffentlichungsjahr der Erstausgabe eines Textes
  <br><br>
**Weitere Metadaten der Texte**: 
- `editor_prename`: Vorname des Herausgebers, von dem die Neuauflage herausgegeben wurde oder der die Zeitschrift bzw. den Band herausgegeben hat, in dem der Text veröffentlicht wurde (die Textsammlungen des Korpus wurden ausschließlich von Männern herausgegeben)
- `editor_surname`: Nachname des Herausgebers, von dem die Neuauflage herausgegeben wurde oder der die Zeitschrift bzw. den Band herausgegeben hat, in dem der Text veröffentlicht wurde
- `volume`: Nummerierung von Zeitschriftenbänden
- `title_addition`: ausführlicher Titel der Zeitschrift
- `edition`: Nummer der Auflage bei mehr als einer Auflage
- `issue`: Zeitschriftennummer
- `pages`: Seitenzahlen bei vollständigen Texten aus Zeitschriften, Sammelbänden etc.
- `pages_exzerpt`: Seitenzahlen bei Auszügen aus Texten
- `archive`: digitales Archiv oder analoges Archiv, aus dem der Text stammt
- `author_address`: Wirkungsstätte des Verfassers, der Verfasserin
- `address`: Veröffentlichungsort des Textes
- `note`: Notiz
- `female_education`: Markierung, ob ein Text
- `author_address_geo`: normierter Name der Wirkungsstätte des Verfassers, der Verfasserin
- `address_geo`: normierter Name des Veröffentlichungsortes des Textes
<br><br>
## 3. Ordnerstruktur
```
fadelive/
├───config/
├───data/
│   └───raw/
├───resources/
│   ├───stop_pos_tag/
│   ├───termsets/
│   └───topic-models/
│       ├───compare/
│       ├───topics_exp_v1/
│       ├───topics_exp_v2/
│       ├───topics_exp_v3/
│       └───topics_v3/
└───src/
    ├───fadelive/
    ├───procession_termsets_topics
    ├───tools_output
    └───tools_visualisations
``` 
<br>

## 4. Module
1. Grundlegende NLP-Pipeline: `src/fadelive/`

2. Skript zur Ausgabe aller Dokumente in einer beliebigen Verarbeitungsstufe mit IDs: `src/tools_output/single_texts_for_tm.py`

3. Skripte zur weiteren Verarbeitung der getaggten Ausdrücke und Topics: `src/procession_termsets_topics/`

4. Skripte für Visualisierungen und Erkundung des Korpus in einem GUI-Tool: `src/tools_visualisations/`
<br><br>
### Installation 
Die Installation des Korpus setzt eine Umgebung mit `python 3.11` voraus.
Die Installation der benötigten Pakete ist ausgelegt für Anaconda und `python`:
- mit Anaconda anhand von `environment.yml` in `main` (installiert auch python 3.11)<br>
`conda env create -f environment.yml`<br>
    - im heruntergeladenen Programmordner von _fadelive_ einen Anaconda-Terminal öffnen und die Zeilen ausführen:<br>
      `conda activate fadelive`<br>
      `pip install .`

- mit `python` anhand von `pyproject.toml` in `main` (`python 3.11` muss vorher installiert worden sein)
    - im heruntergeladenen Programmordner von _fadelive_ einen Terminal öffnen und die Zeilen ausführen:<br>
`pip install .`
<br><br>
### Modul 1: Grundlegende NLP-Pipeline

1. Vorverarbeitung von `data/raw/korpus.csv` und Erzeugung der Verarbeitungsstufen _TXT (min)_, _TXT (lem)_, _TXT (stop)_ in `output/processed_corpus`

    - für _TXT (min)_: Säuberung von OCR-Fehlern mit
  
        - `resources/ocr_post-correction_dictionary_v1.txt` oder `resources/ocr_post-correction_dictionary_v2.txt`
        - Normalisierung des Vokabulars mit `resources/replacements_v1.json`, `resources/replacements_v2.json` oder `resources/replacements_v3.json`

    - für _TXT (lem)_: Lemmatisierung mit `resources/morphmodel_ger.pgz` (HanoverTagger)

    - für _TXT (stop)_: Entfernung von Stoppwörtern mit `resources/stopwords_v1.txt` oder `resources/stopwords_v2.txt`

2. Erzeugung des Vokabulars in `output/vocabular`

    - für `genre`, `textclass`

    - für `intervals`:
  
        - für 4 Intervalle 1782-1852, 1853-1864, 1865-1876, 1877-1891
        - für 3 Intervalle 1782-1856, 1857-1872, 1873-1891
        - für 2 Intervalle 1782-1864, 1865-1891

    - für die drei Verarbeitungsstufen

3. Erzeugung von Statistiken anhand von `data/raw/korpus.csv` und des erzeugten Vokabulars in `output/statistics`

4. POS-Tagging der Top-5000 Ausdrücke des maximal vorverarbeiteten Korpus mit `spacy`

5. Erzeugung der Vorverarbeitungsstufe _TXT (gen)_ für die Verarbeitung zu Wort-Vektor-Modellen in `output/preprocessed_corpus/`

6. Erzeugung der dtm- und tfidf-Matrizen mit Metadaten in `output/dtm_tfidf_stop/`

7. Erzeugung der Kosinus-Matrizen `output/cosine/`

8. Schritt 7 und 8 für Intervalle `output/intervals/`

9. Erzeugung der tfidf-Ranglisten des Vokabulars und der Texte in `output/tfidf_rank/`

10. Erzeugung des Wort-Vektor-Modells des Korpus in `output/word2vec_models`
<br><br>
#### Konfiguration der Pipeline
Bei dem Korpus handelt es sich um eine Sammlung historischer Texte, deren adäquate Verarbeitung angesichts der Normalisierung und der Entfernung von Stoppwörtern herausfordernd ist. Im Lauf des Projektes wurde die Vorverarbeitung kontinuierlich verbessert, um das Vokabular zu vereinheitlichen. Um auch die Reproduzierbarkeit der Verbesserung des Korpus im Laufe erster Experimente zu ermöglichen, kann die Pipeline `src/fadelive/pipeline_config.py` mit drei unterschiedlichen Konfigurationen ausgeführt werden:

- `config/fadelive_v1.toml` mit `ocr_post-correction_dictionary_v1.txt`, `resources/replacements_v1.json`, `resources/stopwords_v1.txt`

- `config/fadelive_v2.toml` mit `ocr_post-correction_dictionary_v2.txt`, `resources/replacements_v2.json`, `resources/stopwords_v1.txt`

- `config/fadelive_v3.toml` mit `ocr_post-correction_dictionary_v2.txt`, `resources/replacements_v3.json`, `resources/stopwords_v2.txt`

Das Korpus, das mit `config/fadelive_v3.toml` vorverarbeitet wird, ist die Version, in der die Verarbeitung des Vokabulars am besten normalisiert ist und Fehler bei der Ersetzung und der Entfernung von Stoppwörtern beseitigt wurden. 

Die mit `config/fadelive_v3.toml` erzeugten Ausgaben finden sich in <https://doi.org/10.25625/APN9VH>

Neben den unterschiedlichen Pipeline-Konfigurationen gibt es eine allgemeine Pipeline, die genauso wie die Konfigurationen angepasst werden kann: `src/fadelive/pipeline/`.
<br><br>
#### Start der Pipelines
Start einer konfigurierten Pipeline `fadelive_v3.toml` mit `src/fadelive/pipeline_config.py` im Terminal des Ordners von _fadelive_:<br> 
`python -m fadelive.pipeline_config --config config/fadelive_v3.toml`

Start der Pipeline `src/fadelive/pipeline` im Terminal des Ordners von _fadelive_:<br>
`python -m fadelive.main`
<br><br>
### Modul 2: Ausgabe des Vokabulars in einer beliebigen Verarbeitungsstufe

Ein Topic-Modell ist grundlegend für die Verarbeitungen in Modul 3 und für topic-spezifische Visualisierungen, die mit Modul 4 erzeugt werden. Für die Modellierung der Topics wurde die ohtm-Pipeline von Bayerschmidt mit MALLET verwendet: <https://github.com/bayerschphi/ohtm_pipeline>

Um das Topic-Modelling durchzuführen, sind intervalle des Diskurszeitraums ideal. Mit dem Skript `src/tools_output/single_texts_for_tm.py` werden für das Topic-Modelling alle Dokumente in einer beliebigen Verarbeitungsstufe mit IDs und intervallspezifisch geordnet in `src/tools_output/single_texts_for_tm.py` ausgegeben.
<br><br>
### Modul 3: Skripte zur Verarbeitung der getaggten Ausdrücke und Topics
Mit diesem Modul können kontrollierte Vokabulare und Topicmodelle  verarbeitet werden. Der hier verwendete _Dokument-Term-Topic-Index_ relativiert die Topics und Texte in Hinblick auf das angewendete Vokabular.

**Funktionen der Skripte**:

`src/procession_termsets_topics/s01_process_stop_pos_tag.py`:<br>
Verarbeitung einer semantisch getaggten POS-Liste und Ausgabe einer Pivot-Tabelle der getaggten Ausdrücke in `output/processed_tag/`

- **Voraussetzung** in `resources/stop_pos_tag/[Vokabular].csv`:<br>
    - semantisch getaggte POS-Liste mit den Spalten `word`, `tag1`, `tag2`, `tag3`

`src/procession_termsets_topics/s02_process_topics.py`:<br> 
Verarbeitung von Topics in `output/processed_topics`

- **Voraussetzung** in `resources/topic-models/[Modell]/`:<br>
    - Document-Topics-Distribution-Matrix als `.csv` mit Topics als Spaltenindex und Text-IDs als Zeilenindex<br>
    - Top-100-Word-Topic-Matrix `.csv` mit SpaltenIndex `Word 0` bis `Word 99` und Zeilenindex Topicbezeichnungen

`src/procession_termsets_topics/s03_termset-topics_dtti.py`:<br> 
Berechnung der Termset-Topic-Text-Verhältnisse in `output/processed_termset/[Termset]/`
- **Voraussetzung**: <br>
    - kontrolliertes Vokabular als Pivottabelle in `resources/termsets/[Termset].csv`<br>
    - verarbeitete Topics in `output/processed_topics/`

`s04_process_termset-topics_dtti.py`:<br>
Verarbeitung der Termset-Topic-Text-Verhältnisse in `output/processed_termset/[Termset]/`
-  **Voraussetzung**:<br>
    - berechnete Termset-Topic-Text-Verhältnisse in `output/processed_termset/`

**Funktionen der Verarbeitung des getaggten Vokabulars, der Termsets und Topics**: 

Eines der grundlegenden Ziele des übergeordneten Projektes ist die Ermittlung von konstitutiven Ausdrücken der Fachlichkeit der Literaturvermittlung im 19. Jahrhundert. Hierzu wurden zunächst zwei Verfahren separat angewendet und dann kombiniert: qualitatives, semantisches Taggen sowie algorithmische Modellierung von Topics und semantisches Taggen der Topics.

Entsprechend der Entwicklung der Verarbeitung des Korpus gibt es drei experimentelle Stufen, in denen auch jeweils eine semantisch getaggte Ausdrucksliste und ein semantisch getaggtes Topic-Modell erzeugt wurde. 
- Taglisten: `resources/stop_pos_tag/`
- Topic-Modelle: `resources/topic-models/`

**Exemplarische Anwendung im Kontext des Projektes**: 
- es wurde eine POS-getaggte Vokabelliste semantisch getaggt
    - indem der POS-Tag-Liste `output/vocabular/vocab_top5000_stop_pos.csv` die Spalten `tag1`, `tag2` und `tag3` hinzugefügt,
    - die Ausdrücke mit bis zu drei Kategorien abstrahiert wurde,
- daraus wurde anhand von `src/procession_termsets_topics/s01_process_stop_pos_tag.py` eine Pivot-Tabelle erzeugt,
- anhand der Tag-Statistiken wurden relevante Tags ausgewählt und die Master-Termsets _Begriffe_, _Gegenstände_ und _Praktiken_ gebildet,
- damit konnten Topics und Texte in Hinblick auf das Termset relativiert werden
<br><br>
### Modul 4: Skripte für Visualisierungen und Erkundung des Korpus in einem GUI-Tool

Mit den beiden rudimentären GUI-Tools `src/tools_visualisations/gui_corpus-explorer.py` und`src/tools_visualisations/gui_tag-topic-explorer.py` können das Korpus erkundet und einzelne Daten erzeugt werden.

#### Korpus-Explorer
Der Korpus-Explorer ermöglicht gängige korpusanalytische Abfragen. Mit den Ausgaben aus der Verarbeitung des Korpus mit der Pipeline und mit einem Termset aus `resourcen/termsets` und einem Topic-Modell aus `resourcen/topic-models` sind alle Abfragen möglich. Alle Dateien sind voreingestellt und können geändert werden. 

`gui_corpus-explorer.py` benötigt
- Document-Term-Matrix:`output/dtm_tfidf_stop/dtm_minfreq6.csv`
- Document-Topic-Verteilung: `resources/topic-models/topics_v3/document-topics-distribution_tag.csv`
- Metadaten der Texte: `data/raw/metadata.csv`
- tfidf-Matrix: `output/dtm_tfidf_stop/tfidf-2000.csv`

Im Tab *Daten* müssen zunächst die Dateien geladen und geprüft werden, ob sie die Voraussetzungen zur Abfrage und Visualisierung erfüllen. 

Neben dem Tab *Daten* gibt es vier Kategorien mit Abfragen und Visualisierungen: 

   ***Ausdrücke***
   - Frequenzliste des Vokabulars (Abfrage)
   - tfidf-Relevanzliste des Vokabulars (Abfrage)
   - Frequenz von gesuchten Ausdrücken in Dokumenten (Abfrage)
   - tfidf-Werte von gesuchten Ausdrücken in Dokumenten (Abfrage)
   - Konkordanzliste zu gesuchtem Ausdruck (Abfrage)
   - Kollokationsliste zu gesuchtem Ausdruck mit Fokussierung ausgewählter Kollokationen (Abfrage)
   - Wortverlaufskurven zu gesuchten Ausdrücken (Visualisierung)
  
   ***Wort-Vektor-Modell***
   - Embeddings und Kosinus-Werte zu gesuchtem Ausdruck (Abfrage)
   - Top-Embeddings von Ausdrücken und Kosinus-Werte vergleichen (Abfrage)
   - Netzwerk von Ausdrücken anhand von Embeddinganzahl und Ähnlichkeitsschwelle (Abfrage)

   ***Termset***, 
   - Streudiagramm eines Termsets anhand des Wort-Vektor-Modells mit dem Dimensionsreduktionsalgorithmus _UMAP_ und dem hierarchischen Clusteralgorithmus von `scikit-learn` (Visualisierung)<br>
       Parameter:
        - Clusteranzahl
        - Tags als Marker im Plot indizieren
   - Wortwolke eines Termsets erstellen (Visualisierung)
   - Dendrogramme eines Termsets anhand des Wort-Vektor-Modells mit dem hierarchischen Clusteralgorithmus von `scikit-learn` (Visualisierung)<br>
       Parameter:
         - Clusteranzahl

   ***Topics***
   - diachrone Topicverläufe an ausgewählten Topics (Visualisierung)


#### Tag-Topic-Explorer
Der Tag-Topic-Explorer visualisiert die Topics diachron und die Tag-Topic-Verhältnisse sowohl synchron als auch diachron. Für die Abfragen werden die Verarbeitungen aus Modul 3 benötigt. Die Tag-Topic-Verhältnisse ermöglichen eine auf das jeweilige Termset relativierte Abfrage der Topics. Alle benötigten Dateien sind voreingestellt und können geändert werden. 

`gui_tag-topic-explorer.py` benötigt
- Termset als Pivot-Tabelle: `resources\termsets\Termset_Begriffe_2.3.csv`
- Top-100-Topics-Word-Matrix: `resources\topic-models\topics_v3\fadelive_mallet_stop_topic_words_100_words_tag.csv`
- tfidf-Matrix: `output\dtm_tfidf_stop\tfidf-2000.csv`
- Rangliste der Topics relativ zum Termset: `output\processed_termset\Termset_Begriffe_2.3\Termset_Begriffe_2.3_tag_topic_rank.csv`
- Relevanzscore der Topics relativ zum Termset:`output\processed_termset\Termset_Begriffe_2.3\Termset_Begriffe_2.3_tag_topic_relevance.csv`
- Verteilung der Topics auf Jahre relativ zum Termset und zu den Top-30 Texten `output\processed_termset\Termset_Begriffe_2.3\Termset_Begriffe_2.3_dtti_topdocs_topic_counts_per_year.csv`
- Summierter Relevanzscore der Termset-Text-Topic-Verhältnisse pro Jahr: `output\processed_termset\Termset_Begriffe_2.3\Termset_Begriffe_2.3_dtti_topdocs_top10_year_value.csv`
- Summierter Relevanzscore der Termset-Text-Topic-Verhältnisse pro Text`output\processed_termset\Termset_Begriffe_2.3\Termset_Begriffe_2.3_dtti_topdocs_top10_value_per_text_topic.csv`
- Verteilung der Tokens pro Jahr: `output\statistics\year_count_tokens.csv`
- Summierter Relevanzscore der Text-Topic-Verhältnisse pro Jahr`output\processed_topics\document-topics-distribution_tag_topdocs_year_value.csv`
- Document-Topic-Verteilung:`resources\topic-models\topics_v3\document-topics-distribution_tag.csv`
- Metadaten der Texte: `data\raw\metadata.csv`

Im Tab *Daten* müssen zunächst die Dateien geladen und geprüft werden, ob sie die Voraussetzungen zur Abfrage und Visualisierung erfüllen. 

Neben dem Tab *Daten* gibt es zwei Kategorien für Abfragen und Visualisierungen: 

   ***Topic-Exploration***
   - Globale Verteilung der Top-[30] Dokumente auf die Topics (Visualisierung)
   - Globale Verteilung der Top-[30] Dokumente auf die Topics im Verhältnis zur Tokenverteilung (Visualisierung)
   - diachrone Topicverläufe an ausgewählten Topics (Visualisierung)

   ***Tag-Topic-Exploration***
   - Blasen-Diagramm zur Verteilung der Tags auf die Topics (Visualisierung)
   - Balkendiagramm zum Relevanzscore der Tags pro Topic und der tfidf-Werte der Tags (Visualisierung)
   - Verteilung der Top-[30] Texte pro Topic auf Jahre, differenziert nach Topics und relativ zum Termset (Visualisierung)
   - Tendenzkurven zur Entwicklung der Topics relativ zum Termset und zu den Topics Top-[30] Texten (Visualisierung)
   - globale Verlaufskurve der Termset-Topic-Verhältnisse (Visualisierung)
   - Vergleich des Verlaufs von Tokens, Topics und Termset-Topic-Text-Verhältnisse (Visualisierung)
   - Rangliste der Texte relativ zu den Topics und zum Termset (Visualisierung)
<br><br>
### Projektstruktur
```
fadelive
│   environment.yml
│   pyproject.toml
│   README.txt
│   requirements.txt
│
├───config
│       fadelive_v1.toml
│       fadelive_v2.toml
│       fadelive_v3.toml
│
├───data
│       raw
│       korpus.csv
│       metadata.csv
│
├───resources
│   │   morphmodel_ger.pgz
│   │   ocr_post-correction_dictionary_v1.txt
│   │   ocr_post-correction_dictionary_v2.txt
│   │   replacements_v1.json
│   │   replacements_v2.json
│   │   replacements_v3.json
│   │   stopwords_v1.txt
│   │   stopwords_v2.txt
│   │   stopwords_v3.txt
│   │
│   ├───stop_pos_tag
│   │       vocab_top5000_stop_pos_tag_exp_v1.csv
│   │       vocab_top5000_stop_pos_tag_exp_v2.csv
│   │       vocab_top5000_stop_pos_tag_exp_v3.csv
│   │       vocab_top5000_stop_pos_tag_v3.csv
│   │
│   ├───termsets
│   │       Termset_Begriffe_2.2.csv
│   │       Termset_Begriffe_2.3.csv
│   │       Termset_Begriffe_2.csv
│   │       Termset_Drama_1.csv
│   │       Termset_Gegenstände_1.2.csv
│   │       Termset_Lyrik_1.csv
│   │       Termset_Praktiken_1.4.3.csv
│   │       Termset_Praktiken_diff_1.csv
│   │       Termset_Prosa_1.csv
│   │
│   └───topic-models
│       │   Parameter.txt
│       │
│       ├───compare
│       ├───topics_exp_v1
│       │       document-topic-distribution_tag.csv
│       │       topic_100_words_tag.csv
│       │
│       ├───topics_exp_v2
│       │       document-topics-distribution_tag.csv
│       │       topic_100-words_tag.csv
│       │
│       ├───topics_exp_v3
│       │       document-topics-distribution_tag.csv
│       │       topics_100-words_tag.csv
│       │
│       └───topics_v3
│               document-topics-distribution_tag.csv
│               fadelive_mallet_stop_topic_words_100_words_tag.csv
│
└───src
    ├───fadelive
    │   │   main.py
    │   │   pipeline.py
    │   │   pipeline_config.py
    │   │   s01_1_preprocessing.py
    │   │   s01_2_vocabular.py
    │   │   s01_3_statistics.py
    │   │   s01_4_pos_tag.py
    │   │   s02_preprocessing_gensim.py
    │   │   s03_dtm_tfidf.py
    │   │   s04_cosine.py
    │   │   s05_dtm_tfidf_cos_intervals.py
    │   │   s06_tfidf_rank.py
    │   └─s07_word_vector_model.py
    │
    ├───procession_termsets_topics
    │       s01_process_stop_pos_tag.py
    │       s02_process_topics.py
    │       s03_termset-topics_dtti.py
    │       s04_process_termset-topics_dtti.py
    │
    ├───tools_output
    │       single_texts_for_tm.py
    │
    └───tools_visualisations
            gui_corpus-explorer.py
            gui_tag-topics-explorer.py
            statistics.py
```


## Lizenz und Zitation 
- **Zitation:** empfohlene Zitierweise steht in `CITATION.cff`
- **Lizenz:** Rechte zur Nutzung und Weiterverarbeitung stehen in `LICENSE.md`