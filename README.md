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

This script builds a Markov model from all `.txt` files in the specified input directory. It generates text, filters it for coherence, and saves the results. We have measures of lexical density, entropy and topic cohesion to help pick, because the 'candidates' can fill up with a LOT. 

Sample output:

```
Entropy: 7.070000171661377
Lexical_density: 0.44
Topic_cohesion: 0.24
---

itself to be so insistent.
it seemed to me that one and all left
our native planets. this possibility had at first seemed
too fantastic ; but it gradually became obvious that

events were unfolding before us with fantastic speed.
each cloud visibly shrank, withdrawing into the distance,
and as i raced eastwards,
it seemed, partly with telepathic exploration of the great

hardships endured by gilgamesh. greater than other kings,
lofty in stature, and thicker in bone.
in sooth it must be very far afield.
on and on i travelled in the darkness,

not does it grow light.
five double-hours he marches ;
thick is the darkness, not does it grow light.
five double-hours he marches ;

thick is the darkness, not does it grow light.
at nine double-hours the wind begins to blow in
his face ; thick is the darkness,
not does it grow light.

at its heart was a vague brilliance,
which faded softly into the dim outer regions and
merged without perceptible boundary into the black sky was
itself another such ‘ universe ’

```

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

The `--rewrite` option has a local LLM doing a pass over the poetry to stitch it into more meaningful units. Untested. Requires a local LLM server compatible with the OpenAI Chat Completions API, running at `http://localhost:1234/v1`. This is the default address for LM Studio. If the server is not found, this step is skipped.

### `enkidu-speech.py`

This script processes each file in the input directory, replacing content words with their WordNet definitions.

Sample output:

```
I WILL [declare formally; declare someone to be something; of titles] to the [everything that exists anywhere] the [performance of moral or religious acts] of [a legendary Sumerian king who was the hero of an epic collection of mythic stories] . This [have the quality of being; (copula, used with an adjective or a predicate noun)] the [an adult person who is male (as opposed to a woman)] to whom all [any movable possession (especially articles of clothing)] [have the quality of being; (copula, used with an adjective or a predicate noun)] [be cognizant or aware of a fact or a specific piece of information; possess knowledge or information about] ; this
[have the quality of being; (copula, used with an adjective or a predicate noun)] the [a male sovereign; ruler of a kingdom] who [be cognizant or aware of a fact or a specific piece of information; possess knowledge or information about] the [a politically organized body of people under a single government] of the [everything that exists anywhere] . He [have the quality of being; (copula, used with an adjective or a predicate noun)] [a way of doing or being] , he [cut with a saw] [something that baffles understanding and cannot be explained] and [be cognizant or aware of a fact or a specific piece of information; possess knowledge or information about] [not open or public; kept private or not revealed] [any movable possession (especially articles of clothing)] , he
[take something or somebody with oneself somewhere] us a [a message that tells the particulars of an act or occurrence or course of events; presented in writing or drama or cinema or as a radio or television program] of the [the time during which someone's life continues] before the [the rising of a body of water and its overflowing onto normally dry land] . He [change location; move, travel, or proceed, also metaphorically] on a [primarily temporal sense; being or indicating a relatively great or greater than average duration or passage of time or a duration as specified] [the act of traveling from one place to another] , [have the quality of being; (copula, used with an adjective or a predicate noun)] [physically and mentally fatigued] , [used until no longer useful] with [a social class comprising those who do manual labor or work for wages] ,
[go or come back to place, condition, or activity where one has been before] he [not move; be in a resting position] , he [carve, cut, or etch into a material or surface] on a [a lump or mass of hard consolidated mineral matter] the [including all components without exception; being one unit or constituting the full amount or extent or duration; complete] [a message that tells the particulars of an act or occurrence or course of events; presented in writing or drama or cinema or as a radio or television program].
```

**Command:**

```bash
python3 enkidu-speech.py --input-dir enkidu-speech-input/ --output-dir enkidu-speech-output/
```

**Arguments:**

-   `--input-dir`: (Required) Path to the directory containing `.txt` files to be translated.
-   `--output-dir`: (Required) Path where the translated output files will be saved.
