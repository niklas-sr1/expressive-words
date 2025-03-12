import json
import os
import time
import numpy as np
from ollama import Client
from tqdm import tqdm  # Progress bar for batching
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import pos_tag
import sys
import zstandard as zstd

if len(sys.argv) < 2:
    print("Usage: python expressive_word_gen.py <wordlist1.txt> [wordlist2.txt ...]")
    sys.exit(1)

# Download the required NLTK model (only once)
nltk.download("averaged_perceptron_tagger_eng")

# Configurations
BATCH_SIZE = 128  # Adjust batch size based on system performance
OUTPUT_FILTERED_WORDLIST_FILE = "expressive_words.txt"
CACHE_FILE = "cached_embeddings.json"
MODEL = "mxbai-embed-large"

# Define multiple elaborate prompts
POSITIVE_PROMPTS = [
    ("Represent a single noun that is highly pictorial, sensory-rich, and visually concrete. "
     "The word should refer to a tangible object, natural phenomenon, celestial body, striking color, "
     "or dramatic weather event. It must be something that can be directly seen, touched, or physically experienced. "
     "Avoid words related to language, description, abstract ideas, or emotions. "
     "Examples: volcano, aurora, sapphire, thunderstorm, eclipse, ember, hurricane, glacier, tornado, coral, "
     "wildfire, dusk, canyon, twilight, lightning, tide, storm, flame, ocean, peak, prism, sunrise, obsidian."),
    ("Represent a noun that creates a strong visual and sensory image. It should denote something specific and concrete "
     "that can be perceived directly in nature or art, such as a vivid gemstone or a dramatic natural event."),
]

NEGATIVE_PROMPTS = [
    ("Represent a general noun that describes a concept, category, or abstract idea rather than something concrete "
     "and visually striking. Examples: phenomenon, nature, thing, concept, expression, meaning, terminology, object, "
     "entity, appearance."),
    ("A word that represents an abstract idea or generalization rather than a specific, tangible item."),
    ("A word that denotes a category, concept, or general idea rather than a vivid, sensory-rich, and visually concrete object."),
]

# Highly pictorial, sensory-rich, and visually concrete nouns
POSITIVE_EXAMPLE_WORDS = [
    # Natural Phenomena
    "volcano", "hurricane", "tornado", "tsunami", "earthquake", "wildfire",
    "landslide", "thunderstorm", "lightning", "blizzard", "geyser", "monsoon", "eclipse", "aurora",
    
    # Celestial Objects
    "supernova", "nebula", "comet", "asteroid", "meteor", "moon", "sunspot", "starburst",
    
    # Geological Formations & Landscapes
    "canyon", "glacier", "waterfall", "cliff", "dune", "cavern", "archipelago", "fjord", "abyss",
    
    # Water & Sky Elements
    "tide", "maelstrom", "whirlpool", "tempest", "mist", "cloudburst", "deluge", "twilight", "dawn",
    
    # Gemstones & Vivid Objects
    "sapphire", "ruby", "emerald", "topaz", "onyx", "obsidian", "amethyst", "prism", "opal", "crystal",
    
    # Fire & Light
    "ember", "flare", "blaze", "inferno", "pyre", "glow", "flicker", "torch"
]

# General, abstract, or conceptual nouns that should be avoided
NEGATIVE_EXAMPLE_WORDS = [
    # Broad Concepts & Categories
    "phenomenon", "entity", "existence", "occurrence", "event", "instance", "object", "nature",
    "substance", "material", "matter", "element", "being", "system",
    
    # Vague or Non-Visual Words
    "concept", "idea", "abstraction", "process", "mechanism", "structure", "framework", "notion", "definition",
    
    # Meta & Language-Related Terms
    "terminology", "description", "phrase", "reference", "explanation", "meaning", "representation",
    
    # Emotion & Subjective Words
    "beauty", "sensation", "experience", "feeling", "impression", "thought", "perception", "wonder",
    
    # Broad Natural Terms
    "environment", "atmosphere", "weather", "climate", "space", "universe", "energy",
    
    # Scientific & Technical Words
    "measurement", "classification", "phenomenon", "analysis", "category", "spectrum", "wavelength"
]

# Initialize Ollama client
ollama = Client(host="http://192.168.178.142:11434")

# Step 1: Load the word list
words = []
for filename in sys.argv[1:]:
    with open(filename, "r", encoding="utf-8") as f:
        words.extend(line.strip() for line in f if line.strip())

print(f"Loaded {len(words)} words from {len(sys.argv[1:])} files.")

# Step 1.1: Filter out undesirable words
ALLOWED_POS = ["NN", "NNP"]  # Singular nouns

# Filter out words with apostrophes
words = [word for word in words if "'" not in word]
# Filter out words longer than 10 characters
words = [word for word in words if len(word) <= 10]
# Filter out words that are not nouns
words = [word for word, pos in pos_tag(words) if pos in ALLOWED_POS]

print(f"Filtered to {len(words)} words.")

# Step 2: Compute embeddings for the positive and negative prompts
def get_prompt_embeddings(prompts):
    embeddings = []
    for prompt in prompts:
        response = ollama.embed(model=MODEL, input=[prompt])
        embeddings.append(np.array(response["embeddings"][0]))
    return np.array(embeddings)

print("Generating embeddings for positive prompts...")
positive_prompt_embeddings = get_prompt_embeddings(POSITIVE_PROMPTS + POSITIVE_EXAMPLE_WORDS)
print("Generating embeddings for negative prompts...")
negative_prompt_embeddings = get_prompt_embeddings(NEGATIVE_PROMPTS + NEGATIVE_EXAMPLE_WORDS)

# Compute composite prompt vectors (mean of prompt embeddings)
# composite_positive = np.mean(positive_prompt_embeddings, axis=0)
# composite_negative = np.mean(negative_prompt_embeddings, axis=0)

# Compute PCA weights for each dimension
# Stack positive & negative prompt embeddings
prompt_matrix = np.vstack([positive_prompt_embeddings, negative_prompt_embeddings])

# Reduce to principal components
print("Performing PCA on prompt embeddings...")
pca = PCA(n_components=20)  # Retain top 20 distinguishing dimensions
prompt_pca = pca.fit_transform(prompt_matrix)
positive_prompt_pca = prompt_pca[:len(positive_prompt_embeddings)]
negative_prompt_pca = prompt_pca[len(positive_prompt_embeddings):]

print("Loading cached embeddings...")
compressed_cache_file = CACHE_FILE + ".zst"
cached_embeddings = {}

if os.path.exists(compressed_cache_file):
    with open(compressed_cache_file, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            decompressed_data = reader.read()
            cached_embeddings = json.loads(decompressed_data.decode("utf-8"))
elif os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cached_embeddings = json.load(f)
    with open(compressed_cache_file, "wb") as f:
        cctx = zstd.ZstdCompressor()
        with cctx.stream_writer(f) as compressor:
            compressor.write(json.dumps(cached_embeddings).encode("utf-8"))

def get_embeddings_batched(word_list, batch_size):
    embeddings = {}
    words_to_process = [word for word in word_list if word not in cached_embeddings]
    if words_to_process:
        print(f"Fetching {len(words_to_process)} new embeddings...")
        for i in tqdm(range(0, len(words_to_process), batch_size), desc="Processing batches"):
            batch = words_to_process[i : i + batch_size]
            response = ollama.embed(model=MODEL, input=batch)
            for word, embedding in zip(batch, response["embeddings"]):
                cached_embeddings[word] = embedding
            time.sleep(0.1)
        with open(compressed_cache_file, "wb") as f:
            cctx = zstd.ZstdCompressor()
            with cctx.stream_writer(f) as compressor:
                compressor.write(json.dumps(cached_embeddings).encode("utf-8"))
    for word in word_list:
        embeddings[word] = cached_embeddings[word]
    return np.array([embeddings[word] for word in word_list])

# Step 3: Get embeddings for the candidate words (with caching)
print("Generating embeddings for word list...")
word_embeddings = get_embeddings_batched(words, BATCH_SIZE)
# Project word embeddings into this PCA space before comparison
print("Projecting word embeddings into PCA space...")
word_embeddings_pca = pca.transform(word_embeddings)

# Step 4: Compute final similarity score for each candidate word
def final_score(word_embedding_pca):
    pos_sim = cosine_similarity([word_embedding_pca], positive_prompt_pca[:1]).mean()
    pos_word_sim = cosine_similarity([word_embedding_pca], positive_prompt_pca[len(POSITIVE_PROMPTS):]).mean()
    neg_sim = cosine_similarity([word_embedding_pca], negative_prompt_pca[:1]).mean()
    neg_word_sim = cosine_similarity([word_embedding_pca], negative_prompt_pca[len(NEGATIVE_PROMPTS):]).mean()
    # Weighted difference between positive and negative similarities
    POS_WEIGHT = 0.75
    NEG_WEIGHT = 0.25
    POS_WORD_WEIGHT = 0.5
    NEG_WORD_WEIGHT = 0.5
    return (POS_WEIGHT * pos_sim - NEG_WEIGHT * neg_sim +
            POS_WORD_WEIGHT * pos_word_sim - NEG_WORD_WEIGHT * neg_word_sim)

def normalize_array(arr):
    """Normalize an array to the range [0, 1]."""
    arr = np.array(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def mmr_selection(word_embeddings_pca, final_scores, num_selected, lambda_param=0.7):
    """
    Perform MMR selection with normalization and vectorized computation to ensure diversity.
    
    :param word_embeddings_pca: PCA-transformed embeddings of all candidate words (shape: [N, d]).
    :param final_scores: Precomputed final relevance scores for each candidate (length N).
    :param num_selected: The number of words to select.
    :param lambda_param: Balance between relevance (normalized final score) and diversity (cosine similarity).
    :return: List of selected candidate indices.
    """
    # Normalize the final scores to [0, 1]
    norm_final_scores = normalize_array(final_scores)
    
    # All candidate indices
    all_indices = np.arange(len(final_scores))
    
    # Initialize: select the candidate with the highest normalized relevance score.
    selected = [int(np.argmax(norm_final_scores))]
    remaining = np.setdiff1d(all_indices, selected)
    
    # Precompute norms for all embeddings (to avoid recomputing in every iteration)
    emb_norms = np.linalg.norm(word_embeddings_pca, axis=1)
    
    pbar = tqdm(total=num_selected - len(selected), desc="Selecting MMR words")
    while len(selected) < num_selected and remaining.size > 0:
        # Get embeddings of the selected candidates (shape: [k, d])
        sel_embeddings = word_embeddings_pca[selected]
        sel_norms = emb_norms[selected]  # shape: [k]
        
        # Get embeddings and norms for all remaining candidates (shape: [r, d])
        candidates = word_embeddings_pca[remaining]
        candidates_norms = emb_norms[remaining]  # shape: [r]
        
        # Compute the dot product matrix between candidates and selected words (shape: [r, k])
        dot_products = np.dot(candidates, sel_embeddings.T)
        
        # Compute cosine similarity: (a · b) / (||a|| * ||b||) using broadcasting
        sim_matrix = dot_products / (candidates_norms[:, None] * sel_norms[None, :])
        
        # For each candidate, take the maximum similarity with any selected word
        max_similarities = sim_matrix.max(axis=1)  # shape: [r]
        
        # Get normalized relevance scores for the remaining candidates
        candidates_relevance = norm_final_scores[remaining]  # shape: [r]
        
        # Compute MMR values for each candidate
        mmr_values = lambda_param * candidates_relevance - (1 - lambda_param) * max_similarities
        
        # Select the candidate with the highest MMR score
        best_idx_in_remaining = int(np.argmax(mmr_values))
        best_candidate = int(remaining[best_idx_in_remaining])
        
        # Add this candidate to the selected list
        selected.append(best_candidate)
        # Remove the selected candidate from remaining
        remaining = np.delete(remaining, best_idx_in_remaining)
        pbar.update(1)
    
    pbar.close()
    return selected

print("Computing final scores for each word...")
scores = np.array([final_score(embedding) for embedding in word_embeddings_pca])

# Rank words by final score (higher is better)
ranked_indices = np.argsort(scores)[::-1]

# Step 5: Do MMR selection to ensure diversity
TOP_PERCENTAGE = 0.1  # Adjust as needed
top_n = int(len(words) * TOP_PERCENTAGE)
print(f"Performing MMR selection on a maximum of {top_n} words...")
selected_indices = mmr_selection(word_embeddings_pca, ranked_indices, top_n, 0.75)

# Get the top-ranked words
ranked_words = [words[i] for i in selected_indices]
with open(OUTPUT_FILTERED_WORDLIST_FILE, "w", encoding="utf-8") as f:
    for word in ranked_words[:top_n]:
        f.write(word + "\n")

print(f"Saved {len(selected_indices)} expressive words to {OUTPUT_FILTERED_WORDLIST_FILE}.")
print("Example words:", ranked_words[:10])
