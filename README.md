# Enkidu

Pair of Python scripts for procedural text generation. This will come in handy on a later Salvage Crew novel. 


## Scripts

-   `enkidu-poetry.py`: Generates poetic text from a source corpus using a Markov chain, with multi-stage filtering and an optional rewrite pass using a local LLM. Right now it's the Epic of Gilgamesh and Stapledon's Star Maker (in public domain since 2021 outside the US).
-   `enkidu-speech.py`: Translates text files by replacing content words (nouns, verbs, adjectives, adverbs) with their primary WordNet gloss. Test case is Gilgamesh.

## Setup

### Dependencies

The scripts require Python 3 and several third-party libraries. Install them via pip:

```bash
pip install numpy spacy requests rich nltk tqdm
```

### First-Time Run

On the first execution, the scripts will automatically download the necessary SpaCy and NLTK models (`en_core_web_sm`, `punkt`, `wordnet`, etc.) if they are not already installed.

## Usage

### Directory Structure

The scripts are designed to work with a specific directory structure. Create the following input folders before running:

-   `enkidu-poetry-input/`: Place `.txt` corpus files here for the poetry generator.
-   `enkidu-speech-input/`: Place `.txt` files you want to translate here.

The scripts will automatically create output directories as needed.

### `enkidu-poetry.py`

This script builds a Markov model from all `.txt` files in the specified input directory. It generates text, filters it for coherence, and saves the results.

**Command:**

```bash
python3 enkidu-poetry.py --input-dir enkidu-poetry-input/ --output-dir enkidu-poetry-output/ [OPTIONS]
```

**Arguments:**

-   `--input-dir`: (Required) Path to the directory containing your corpus `.txt` files.
-   `--output-dir`: (Required) Path where the output subdirectories (`candidates/`, `final_verses/`) will be created.
-   `-c, --count`: Target number of coherent documents to generate.
-   `-b, --batch`: Number of text seeds to generate per cycle.
-   `-n, --n-gram`: N-gram order for the Markov model (default: 3).
-   `--rewrite`: If set, the script will attempt to use a local LLM to rewrite pairs of generated texts into final verses.

**LLM Rewrite Feature:**

The `--rewrite` option requires a local LLM server compatible with the OpenAI Chat Completions API, running at `http://localhost:1234/v1`. This is the default address for LM Studio. If the server is not found, this step is skipped.

### `enkidu-speech.py`

This script processes each file in the input directory, replacing content words with their WordNet definitions.

**Command:**

```bash
python3 enkidu-speech.py --input-dir enkidu-speech-input/ --output-dir enkidu-speech-output/
```

**Arguments:**

-   `--input-dir`: (Required) Path to the directory containing `.txt` files to be translated.
-   `--output-dir`: (Required) Path where the translated output files will be saved.
