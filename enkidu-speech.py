import argparse
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import os
from tqdm import tqdm

# --- Setup: Download NLTK data if missing ---
def download_nltk_resources():
    resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4', 'punkt_tab', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'taggers/{resource}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{resource}')
                except LookupError:
                    nltk.download(resource, quiet=True)

download_nltk_resources()

# --- Helper Functions ---

def get_wordnet_pos(treebank_tag):
    """
    Maps NLTK POS tags to WordNet POS tags.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def translate_line_to_gloss(text):
    """
    Converts a single string of text into WordNet glosses.
    """
    if not text.strip():
        return text

    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    
    alien_output = []

    for word, tag in tagged_tokens:
        wn_tag = get_wordnet_pos(tag)
        
        # Check if it's a content word (Noun, Verb, Adj, Adv)
        if wn_tag:
            try:
                synsets = wordnet.synsets(word, pos=wn_tag)
                if synsets:
                    # Take the first (most frequent) definition
                    gloss = synsets[0].definition()
                    alien_output.append(f"[{gloss}]")
                else:
                    alien_output.append(word)
            except Exception:
                alien_output.append(word)
        else:
            # Keep punctuation and stopwords as is
            alien_output.append(word)

    return " ".join(alien_output)

def process_file(input_filename, output_filename):
    """
    Reads the input file line by line and writes to output.
    """
    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' was not found.")
        return

    print(f"Processing '{input_filename}'...")
    
    try:
        # Get total number of lines for progress bar
        with open(input_filename, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, total=total_lines, desc="Translating", unit="line"):
                # Process the line
                translated_line = translate_line_to_gloss(line)
                # Write to file with a newline
                outfile.write(translated_line + "\n")
                
        print(f"Success! Translation saved to '{output_filename}'.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    """
    Main function to parse arguments and process files.
    """
    parser = argparse.ArgumentParser(description="Translate text files to WordNet glosses.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input text files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the translated files.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]

    if not input_files:
        print(f"No files found in '{args.input_dir}'.")
        return

    for filename in input_files:
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        process_file(input_path, output_path)

# --- Main Execution ---

if __name__ == "__main__":
    main()