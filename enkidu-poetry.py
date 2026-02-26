"""
ENKIDU
"""

import os
import sys
import time
import json
import argparse
import random
import glob
from collections import defaultdict
from datetime import datetime
from typing import List

# --- Third Party Imports ---
try:
    import numpy as np
    import spacy
    import requests
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console
    from rich import box
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
except ImportError as e:
    sys.exit(f"CRITICAL ERROR: Missing dependency. {e}\nPlease run: pip install numpy spacy requests rich nltk")

# --- NLTK setup ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' model...")
    nltk.download('punkt', quiet=True)
    print("'punkt' downloaded.")

# --- SpaCy Setup ---
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    if 'sentencizer' not in nlp.pipe_names:
        nlp.add_pipe('sentencizer')
except OSError:
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe('sentencizer')

# ==============================================================================
# CONFIGURATION & THRESHOLDS
# ==============================================================================

THRESHOLDS = {
    "entropy_min": 2.0,
    "entropy_max": 9.5,
    "topic_cohesion_min": 0.15 
}

# ==============================================================================
# DATA EXPORTER & VERSE FORMATTER
# ==============================================================================

class DataExporter:
    @staticmethod
    def format_as_verse(text, words_per_line=(5, 9), lines_per_stanza=4):
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            hit_punctuation = word[-1] in ".,!?;:"
            is_long_enough = len(current_line) >= words_per_line[0]
            is_max_length = len(current_line) >= words_per_line[1]
            
            if (hit_punctuation and is_long_enough) or is_max_length:
                lines.append(" ".join(current_line))
                current_line = []
                
        if current_line:
            lines.append(" ".join(current_line))
            
        stanzas = []
        for i in range(0, len(lines), lines_per_stanza):
            stanzas.append("\n".join(lines[i:i+lines_per_stanza]))
            
        return "\n\n".join(stanzas)

    @staticmethod
    def save(generator, encoded_text, metrics, directory, fname, full_length, seed_length):
        raw_text = generator.decode_batch(np.array([encoded_text]))[0]
        poetic_text = DataExporter.format_as_verse(raw_text)
        
        filepath = os.path.join(directory, fname)
        header = "\n".join([f"{k.capitalize()}: {v}" for k, v in metrics.items()])
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{header}\n---\n\n{poetic_text}")


# ==============================================================================
# CORE MODULES
# ==============================================================================

class MarkovGenerator:
    def __init__(self, dashboard, input_dir, n=3):
        self.n = n
        self.input_dir = input_dir
        self.word_to_idx = {}
        self.idx_to_word = []
        self.chain = defaultdict(list)
        self.full_size = 0
        self.dashboard = dashboard
        
        self.dashboard.log(f"Initializing MarkovGenerator with n={self.n}", "magenta")
        self._build_model()
        self.dashboard.log("Markov model built successfully.", "green")

    def _build_model(self):
        corpus_files = glob.glob(os.path.join(self.input_dir, "*.txt"))

        if not corpus_files:
             sys.exit(f"CRITICAL ERROR: No .txt corpus files found in '{self.input_dir}'.")

        self.dashboard.log(f"Building model from {len(corpus_files)} corpus file(s)...", "yellow")
        
        full_text = ""
        for file_path in corpus_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text += f.read().lower() + "\n"

        if not full_text.strip():
            sys.exit("CRITICAL ERROR: Combined corpus text is empty.")

        tokens = [word for sent in sent_tokenize(full_text) for word in word_tokenize(sent)]
        
        if len(tokens) < self.n:
            sys.exit(f"CRITICAL ERROR: Not enough tokens ({len(tokens)}) for order {self.n}.")

        vocab = sorted(list(set(tokens)))
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}
        self.idx_to_word = vocab
        self.full_size = len(vocab)
        
        indexed_tokens = [self.word_to_idx[token] for token in tokens]

        for i in range(len(indexed_tokens) - self.n):
            state = tuple(indexed_tokens[i:i+self.n])
            next_word_idx = indexed_tokens[i+self.n]
            self.chain[state].append(next_word_idx)

        if not self.chain:
            sys.exit("CRITICAL ERROR: Markov chain is empty.")

    def generate_batch(self, num_seeds, seed_length):
        start_states = list(self.chain.keys())
        if not start_states:
             return np.array([], dtype=np.int32)
             
        batch = []
        for _ in range(num_seeds):
            current_state = random.choice(start_states)
            sequence = list(current_state)
            
            for _ in range(seed_length - self.n):
                if current_state in self.chain:
                    next_word_idx = random.choice(self.chain[current_state])
                    sequence.append(next_word_idx)
                    current_state = current_state[1:] + (next_word_idx,)
                else:
                    break
            
            while len(sequence) < seed_length:
                sequence.append(random.randint(0, self.full_size - 1))
            batch.append(sequence[:seed_length])
            
        return np.array(batch, dtype=np.int32)
        
    def decode_batch(self, batch: np.ndarray) -> List[str]:
        decoded = []
        for row in batch:
            text = " ".join([self.idx_to_word[idx] for idx in row if idx < self.full_size])
            text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
            decoded.append(text)
        return decoded


class MultiStageFilter:
    def __init__(self, text_generator: MarkovGenerator, dashboard, lexical_density_threshold=0.3):
        self.text_generator = text_generator
        self.dashboard = dashboard
        self.lexical_density_threshold = lexical_density_threshold

    def filter_and_archive(self, batch: np.ndarray, args, dir_candidates):
        for i, encoded_text in enumerate(batch):
            if self.dashboard.stats["coherent"] >= args.count:
                break

            L = len(encoded_text)
            if L == 0:
                self.dashboard.stats["vaporized"] += 1
                continue

            # Stage 1: Math
            counts = np.bincount(encoded_text, minlength=self.text_generator.full_size).astype(np.float32)
            probs = counts[counts > 0] / L
            entropy = -np.sum(probs * np.log2(probs))

            if not (THRESHOLDS["entropy_min"] <= entropy <= THRESHOLDS["entropy_max"]):
                self.dashboard.stats["vaporized"] += 1
                continue

            # Stage 2: Linguistic
            text = self.text_generator.decode_batch(np.array([encoded_text]))[0]
            doc = nlp(text)

            words = [token for token in doc if token.is_alpha]
            num_sentences = len(list(doc.sents))
            
            if len(words) == 0 or num_sentences == 0:
                self.dashboard.stats["vaporized"] += 1
                continue

            lexical_words = sum(1 for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"])
            density = lexical_words / len(doc)
            
            if density <= self.lexical_density_threshold:
                self.dashboard.stats["vaporized"] += 1
                continue

            # Stage 3: Topic Cohesion
            sentences = list(doc.sents)
            if len(sentences) < 2:
                self.dashboard.stats["vaporized"] += 1
                continue

            overlaps = []
            for j in range(len(sentences) - 1):
                s1_words = set(t.lemma_ for t in sentences[j] if t.is_alpha and not t.is_stop)
                s2_words = set(t.lemma_ for t in sentences[j+1] if t.is_alpha and not t.is_stop)
                
                if not s1_words or not s2_words:
                    overlaps.append(0.0)
                    continue
                    
                intersection = len(s1_words.intersection(s2_words))
                union = len(s1_words.union(s2_words))
                overlaps.append(intersection / union)
            
            avg_similarity = sum(overlaps) / len(overlaps) if overlaps else 0.0
            
            if avg_similarity < THRESHOLDS["topic_cohesion_min"]:
                self.dashboard.stats["vaporized"] += 1
                continue

            # Survivor
            metrics = {
                "entropy": round(entropy, 2),
                "lexical_density": round(density, 2),
                "topic_cohesion": round(avg_similarity, 2)
            }
            
            self.dashboard.stats["coherent"] += 1
            self.dashboard.log(f"COHERENT Salvaged! Cohesion: {metrics['topic_cohesion']:.2f}", "bold green")
            
            if args.save_coherent:
                fname = f"enkidu_{int(time.time())}_{i}.txt"
                DataExporter.save(self.text_generator, encoded_text, metrics, dir_candidates, fname, args.full, args.seed)


# ==============================================================================
# DASHBOARD UI 
# ==============================================================================

class enkiduDashboard:
    def __init__(self, target_count):
        self.target_count = target_count
        self.stats = {"gen": 0, "vaporized": 0, "coherent": 0, "start_time": time.time()}
        self.console = Console()
        
    def log(self, msg, style="white"):
        """Prints standard logs directly to the console, pushing the Live panel down automatically."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim][{ts}][/dim] {msg}", style=style)

    def render_stats(self):
        """Returns a single clean panel containing the metrics."""
        table = Table(box=box.SIMPLE, expand=False, show_header=False)
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Value", style="green", justify="right")
        
        elapsed = time.time() - self.stats["start_time"]
        rate = int(self.stats["gen"] / elapsed) if elapsed > 0 else 0
        
        table.add_row("Throughput:", f"{rate:,} seeds/sec")
        table.add_row("Total Generated:", f"{self.stats['gen']:,}")
        table.add_row("Vaporized (Noise):", f"[red]{self.stats['vaporized']:,}[/red]")
        table.add_row("Coherent Salvaged:", f"[bold green]{self.stats['coherent']}/{self.target_count}[/bold green]")
        
        return Panel(table, title="[bold blue]PROJECT enkidu[/bold blue]", border_style="cyan", padding=(0, 2))


# ==============================================================================
# REWRITE FUNCTION
# ==============================================================================

def rewrite_verses(dashboard, dir_candidates, dir_final_verses):
    try:
        requests.get("http://localhost:1234/v1/models", timeout=3)
        dashboard.log("LMStudio API connected for rewriting.", "green")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        dashboard.log("LMStudio not found. Skipping rewrite process.", "red")
        return

    dashboard.log("Starting final rewrite process...", "cyan")
    
    candidate_files = sorted(glob.glob(os.path.join(dir_candidates, "*.txt")))
    if len(candidate_files) < 2:
        dashboard.log("Not enough candidate files to pair for rewriting.", "red")
        return

    for i in range(0, len(candidate_files) - 1, 2):
        file1_path = candidate_files[i]
        file2_path = candidate_files[i+1]

        def extract_text(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                return content.split('---\n\n', 1)[-1] if '---\n\n' in content else content

        content1 = extract_text(file1_path)
        content2 = extract_text(file2_path)

        prompt = f"""You are a scribe tasked with creating a new religious text. You will be given two separate passages. Your task is to rewrite them as two connected verses from a sacred book. The verses should flow naturally from one to the other, maintaining a consistent, elevated, and archaic tone.

Passage 1:
{content1}

Passage 2:
{content2}

Rewrite these as two connected verses. Do not add outside commentary.
"""

        dashboard.log(f"Sending pair {i//2 + 1} for rewriting...", "yellow")
        
        url = "http://localhost:1234/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 1500}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            rewritten_text = response.json()['choices'][0]['message']['content']
            
            verse_filename = f"verse_{i//2}.txt"
            with open(os.path.join(dir_final_verses, verse_filename), "w", encoding='utf-8') as f:
                f.write(rewritten_text)
            dashboard.log(f"Saved {verse_filename}", "green")

        except Exception as e:
            dashboard.log(f"Error during rewriting: {e}", "red")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Project enkidu: Text Salvage")
    parser.add_argument("--input-dir", required=True, help="Directory containing input text files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output files.")
    parser.add_argument("-c", "--count", type=int, default=5, help="Target number of coherent documents")
    parser.add_argument("-b", "--batch", type=int, default=1000, help="Batch size per cycle")
    parser.add_argument("-s", "--seed", type=int, default=500, help="Initial seed length in words")
    parser.add_argument("-f", "--full", type=int, default=7000, help="Full document length after expansion")
    parser.add_argument("-n", "--n-gram", type=int, default=3, help="N-gram order for the Markov model.")
    parser.add_argument("--save-coherent", action=argparse.BooleanOptionalAction, default=True, help="Save passing texts.")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite candidate texts into final verses using LLM.")
    args = parser.parse_args()

    # --- Directory Setup ---
    dir_candidates = os.path.join(args.output_dir, "candidates")
    dir_final_verses = os.path.join(args.output_dir, "final_verses")
    for d in [dir_candidates, dir_final_verses]:
        os.makedirs(d, exist_ok=True)

    dashboard = enkiduDashboard(args.count)
    text_generator = MarkovGenerator(dashboard, input_dir=args.input_dir, n=args.n_gram)
    multi_filter = MultiStageFilter(text_generator, dashboard, lexical_density_threshold=0.3)

    # Wrap the simple rendered stats panel in the Live handler
    with Live(dashboard.render_stats(), console=dashboard.console, refresh_per_second=4) as live:
        try:
            while dashboard.stats["coherent"] < args.count:
                batch = text_generator.generate_batch(args.batch, args.seed)
                dashboard.stats["gen"] += len(batch)
                
                multi_filter.filter_and_archive(batch, args, dir_candidates)
                
                live.update(dashboard.render_stats())

        except KeyboardInterrupt:
            dashboard.log("Operation aborted by user.", "red")
            live.update(dashboard.render_stats())
            time.sleep(1)

    if args.rewrite:
        rewrite_verses(dashboard, dir_candidates, dir_final_verses)

if __name__ == "__main__":
    main()
