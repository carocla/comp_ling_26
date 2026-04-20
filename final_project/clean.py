# --- Import packages ---
import glob
import re
import os
import nltk
from pathlib import Path
from multiprocessing import Pool
from stopwords import get_stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# --- Setup & prep ---
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
lemmatizer = WordNetLemmatizer()

# Pre-compile regex patterns for speed
RE_VIOLET = re.compile(r"\bviolet fate sect\b")
RE_NT = re.compile(r"\b\w+n't\b")
RE_SUFFIXES = re.compile(r"'(ve|d|ll|re|m|t|s)\b")
RE_DIGITS = re.compile(r"[\d%]")
RE_WORDS = re.compile(r"\b[a-z]+\b")
RE_SENTENCE = re.compile(r"[.!?]")
RE_AHH = re.compile(r"ah+")

def get_wordnet_pos(tag):
    if tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

# --- Dictionaries & Sets ---
STOPWORDS = set(get_stopwords("en")) | {
    # Numbers
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
    'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million',
    'meter', 'towards', 'upon', 'shall', 'however', 'although', 'also',
    # Character names / chinese fragments
    'gu', 'meng', 'fang', 'hao', 'klein', 'yuan', 'chen', 'changsheng',
    'bai', 'chang', 'zhou', 'sheng', 'tang', 'masego', 'luo', 'ning',
    'wang', 'shang', 'akua', 'yue', 'bing', 'tie', 'hei', 'vasquer',
    'han', 'yan', 'effie', 'yourong', 'shan', 'yun', 'qing', 'zheng',
    'yang', 'audrey', 'argrave', 'sunny', 'erin', 'anneliese', 'ryoka',
    'ceria', 'galamon','nephis', 'durran', 'orion', 'elenore', 'klbkch',
    'toren', 'cassie','hakram', 'flos', 'callow', 'pisces', 'ridia',
    'simon', 'bakri', 'eliasor', 'bruno', 'mingrui', 'melissa',
    # Noise / casual stuff
    'yeah', 'heh', 'tsk', 'huh', 'hmm', 'boom', 'bang',
    'tephramancy', 'tephramancer', 'tephramancers'
}
NOPE = ("chapter","(","[","-","*",".","translat","edit",
        "table of contents","note","epub","author","book",
        "i shall seal the heavens", "oceanofpdf.com",
        "skyfarrow", "prologue", "1","2","3","4","5","6","7","8",
        "9","0", "epilogue", "afterword", "contents", "tn:", "tl:",
        "political map", "chatper"
)
contraction_fixes = {
    "don't": "", "dont": "", "won't": "", "wont": "", "isn't": "", "isnt": "",
    "aren't": "", "arent": "", "weren't": "", "werent": "", "wouldn't": "", "wouldnt": "",
    "couldn't": "", "couldnt": "", "shouldn't": "", "shouldnt": "", "hasn't": "", "hasnt": "",
    "haven't": "", "havent": "", "hadn't": "", "hadnt": "", "didn't": "", "didnt": "",
    "doesn't": "", "doesnt": "", "can't": "", "cant": "", "it's": "", "its": "",
    "let's": "", "lets": "", "that's": "", "thats": "", "what's": "", "whats": "",
    "who's": "", "whos": "", "there's": "", "theres": "", "here's": "", "heres": "", "ain't":""
}
manual_replacements = {
    'heavens': 'heaven',
    'hellgods': 'god',
    'immortality': 'immortal',
    'mortals': 'mortal',
    'immortals': 'immortal',
    'heavenly':'heaven',
    'prayer':'pray',
    'cultivation':'cultivate',
    'cultivating':'cultivate',
    'die':'death',
    'dead':'death',
    'refinement':'refine',
    'bro':'brother',
    'mom':'mother',
    'medicinal':'medicine',
}

# --- Main Function ---
def process_file(args):
    filepath, output_folder = args
    out_dir = Path(output_folder)
    file_path = Path(filepath)
    output_path = out_dir / file_path.name.replace(".txt", "_clean.txt")
    skipping_thoughts = False

    cleaned_lines = []

    with open(filepath, "r") as infile:
        for line in infile:
            line = line.strip().lower()
            if not line: continue

            if "translator's thoughts" in line or "note from deathblade" in line:
                skipping_thoughts = True
                continue
            if skipping_thoughts:
                if line.startswith("chapter"):
                    skipping_thoughts = False
                    continue
                else:
                    continue
            if line.startswith(NOPE) or "http" in line:
                continue

            line = line.replace("’", "'")
            for c, r in contraction_fixes.items(): line = line.replace(c, r)

            line = RE_VIOLET.sub("", line)
            line = RE_NT.sub("", line)
            line = RE_SUFFIXES.sub("", line)
            line = RE_DIGITS.sub("", line)
            line = RE_AHH.sub("", line)
            line = line.replace("-", " ")

            sentences = RE_SENTENCE.split(line)
            for sentence in sentences:
                words = RE_WORDS.findall(sentence)
                if not words: continue

                pos_tags = nltk.pos_tag(words)
                lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in pos_tags]
                lemmatized = [manual_replacements.get(w, w) for w in lemmatized]
                final_words = [w for w in lemmatized if len(w) >= 3 and w not in STOPWORDS]
                if final_words:
                    cleaned_lines.append(' '.join(final_words))

    with open(output_path, "w") as outfile:
        outfile.write('\n'.join(cleaned_lines))

    return f"Finished {file_path.name}"

# --- Multiprocessing ---
if __name__ == "__main__":
    input_dirs = ["Corpus_A_raw", "Corpus_B_raw"]
    output_dirs = ["Corpus_A_clean", "Corpus_B_clean"]

    for in_dir, out_dir in zip(input_dirs, output_dirs):
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        files = glob.glob(os.path.join(in_dir, "*.txt"))

        # This will use all available CPU cores
        with Pool() as pool:
            results = pool.map(process_file, [(f, out_dir) for f in files])
            for res in results: print(res)