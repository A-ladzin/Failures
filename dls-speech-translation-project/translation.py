# translation.py

OPENROUTER_KEY = ""

from transformers import MarianMTModel, MarianTokenizer
import re
from openai import OpenAI
from typing import Dict, List
from tqdm import tqdm
import numpy as np

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_KEY
)

# class TranslationWrapper:
#     """
#     Wrapper for Machine Translation (English -> Russian).
#     Uses Helsinki-NLP/opus-mt-en-ru Marian model.
#     """
#     def __init__(self, model_name="Helsinki-NLP/opus-mt-en-ru"):
#         self.tokenizer = MarianTokenizer.from_pretrained(model_name)
#         self.model = MarianMTModel.from_pretrained(model_name)

#     def translate(self, text):
#         """
#         Translate English `text` to Russian. Returns Russian string.
#         """
#         # Preprocess input
#         batch = self.tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
#         translated = self.model.generate(**batch)
#         tgt = self.tokenizer.decode(translated[0], skip_special_tokens=True)
#         return tgt




# 2.2 Prompt template that tells DeepSeek-R1 to preserve style
PROMPT_TEMPLATE = """
Below is a sentence from a narrator who speaks in his unique specifical style.
Here are a few exemplar sentences of that style:

1. {ex0}
2. {ex1}
3. {ex2}
4. {ex3}
5. {ex4}
6. {ex5}
7. {ex6}
8. {ex7}
9. {ex8}
10. {ex9}
11. {ex10}
12. {ex11}

Translate the following English text into Russian, phrase by phrase preserving that same specifical tone,
including any humor or rhetorical flourishes where possible to fully convey the emotional and stylistic component, 
use letter representation of any numbers maintaining the form appropriate to the context,
phrases are separated by pattern of symbols "<|>" you must keep the order and structure of the phrases unchanged so that 
Hello World<|>Bye World -> Здравствуй Мир<|>Пока мир:

--- Begin English Text ---
{chunk}
--- End English Text ---

Provide only the Russian translation (no extra commentary).
""".strip()

# 2.3 Chunking helper: split a long string into ≤ max_chars chunks at sentence boundaries
def chunk_text_for_llm(text: str, max_chars: int = 3000) -> list[str]:
    """
    Splits `text` into chunks no longer than `max_chars` characters,
    breaking at sentence-ending punctuation (.!?).
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentence_end_re = re.compile(r'(?<=[\.\?\!])\s+')
    sentences = sentence_end_re.split(text)
    chunks = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # If adding this sentence would exceed max_chars, flush the current chunk
        if len(current) + len(sent) + 1 > max_chars:
            if current:
                chunks.append(current)
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current:
        chunks.append(current)
    return chunks



import os
import time
from huggingface_hub import InferenceClient

# 3.1 Initialize InferenceClient once (reads API key from environment)
# client = InferenceClient(api_key=API_KEY)

def translate_with_deepseek_r1(
    full_transcript: str,
    style_exemplars: list[str],
    max_chars_per_chunk: int = 2500,
    temperature: float = 0.7,
    max_retries: int = 2
) -> str:
    """
    Splits `full_transcript` into smaller chunks, prompts DeepSeek-R1 to translate each chunk into Russian
    preserving the narrator’s style, and returns the concatenated Russian text.
    """
    chunks = chunk_text_for_llm(full_transcript, max_chars=max_chars_per_chunk)
    russian_pieces = []

    for idx, chunk in enumerate(chunks):
        # 3.2 Build the prompt
        prompt = PROMPT_TEMPLATE.format(
            ex0=style_exemplars[0],
            ex1=style_exemplars[1],
            ex2=style_exemplars[2],
            ex3=style_exemplars[3],
            ex4=style_exemplars[4],
            ex5=style_exemplars[5],
            ex6=style_exemplars[6],
            ex7=style_exemplars[7],
            ex8=style_exemplars[8],
            ex9=style_exemplars[9],
            ex10=style_exemplars[10],
            ex11=style_exemplars[11],

            chunk=chunk
        )

        # 3.3 Attempt the API call up to max_retries times
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="deepseek/deepseek-r1-0528:free",
                    messages=[
                        {"role": "system", "content": "You are a translation assistant that preserves style."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=16000
                )
                rus_chunk = response.choices[0].message.content.strip()
                russian_pieces.append(rus_chunk)
                break  # success → exit retry loop

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1 + attempt)  # simple backoff
                    continue
                else:
                    raise RuntimeError(f"DeepSeek-R1 API failed on chunk {idx}: {e}")

    # 3.4 Join all Russian pieces with double newlines to preserve paragraph breaks
    full_translation_with_reasoning = "\n\n".join(russian_pieces)
    import re

    def remove_thinking(text: str) -> str:
        """
        Remove all substrings wrapped in <think>...</think> (including any newlines)
        and return the “raw” text.
        """
        # The DOTALL flag (re.S) makes `.` match newlines as well.
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    return remove_thinking(full_translation_with_reasoning)


def batch_translate_segments(segments: List[Dict],exemplars) -> List[Dict]:
    """
    Given Whisper segments [{"start","end","text"}...],
    translate each 'text' to Russian and return [{"start","end","ru_text"}...].
    """
    translated = []
    segments_joined = "<|>".join([seg["text"] for seg in segments])
    
    while True:
        ru = translate_with_deepseek_r1(segments_joined,exemplars)
        ru = ru.split("<|>")
        if len(ru) <= len(segments):
            break
        else:
            print("Inconsistency occured during translation, Retry... ")
    
    for seg_idx in tqdm(range(len(ru)),total = len(ru)):
        # ru = translate_with_deepseek_r1(seg["text"],exemplars)
        translated.append({
            "start": segments[seg_idx]["start"],
            "end":   segments[seg_idx]["end"],
            "ru_text": ru[seg_idx],
            "silence": segments[seg_idx]["silence"]
        })
        # Avoid rate‐limit spikes
        time.sleep(0.1)
    return translated

import re
from typing import List, Tuple

# 1) Define a small set of colloquial markers / style keywords
COLLOQUIAL_MARKERS = [
    r"\byou know\b", r"\blet's\b", r"\bI mean\b", r"\bright\b",
    r"\bI'm\b", r"\bwe're\b", r"\bain't\b", r"\bkind of\b", r"\bsort of\b"
]
FIRST_PERSON = [r"\bI\b", r"\bme\b", r"\bmy\b", r"\bwe\b", r"\bour\b"]
SECOND_PERSON = [r"\byou\b", r"\byour\b"]


def split_into_sentences(text: str) -> List[str]:
    """
    A simple regex‐based sentence splitter. Splits on period, exclamation, or question mark
    followed by whitespace and a capital letter. Falls back on newlines if needed.
    """
    # First, normalize whitespace
    text = text.strip().replace("\n", " ")
    # Use lookahead to split on . ! ? when followed by a space + capital letter
    sentence_end_re = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z])")
    sentences = sentence_end_re.split(text)

    # If we got nothing (e.g. no punctuation), fall back to splitting on .!? directly
    if len(sentences) == 1:
        sentences = re.split(r"(?<=[\.\?\!])\s+", text)

    # Trim whitespace on each
    return [s.strip() for s in sentences if s.strip()]


def score_sentence(sent: str) -> int:
    """
    Assign a simple score to `sent` based on presence of rhetorical/stylistic markers:
      +2 if it contains a "?" (rhetorical / energetic)
      +1 if it contains a "!" (exclamation)
      +1 per colloquial marker (e.g. "you know", "let's", "I mean")
      +1 per first‐person usage (I, we, my, etc.)
      +1 per second‐person usage (you, your)
    """
    score = 0

    # Count question marks
    if "?" in sent:
        score += 1

    # Count exclamation points
    if "!" in sent:
        score += 1

    # Check colloquial markers
    for pattern in COLLOQUIAL_MARKERS:
        if re.search(pattern, sent, flags=re.IGNORECASE):
            score += 1

    # Check first‐person pronouns
    for pattern in FIRST_PERSON:
        if re.search(pattern, sent):
            score += 1

    # Check second‐person pronouns
    for pattern in SECOND_PERSON:
        if re.search(pattern, sent):
            score += 1

    return score


def extract_style_exemplars(
    transcript: str = None,
    num_exemplars: int = 12,
    sentences: List = None
) -> List[str]:
    """
    From `transcript`, pick the top `num_exemplars` sentences most likely to reflect the narrator’s style.
    Returns a list of sentences (strings), in descending order of “style‐score.”
    """
    # 1) Split into sentences
    
    if sentences is None:
        sentences = split_into_sentences(transcript)


    # 2) Score each sentence
    scored: List[Tuple[int, str]] = []
    for sent in sentences:
        s = score_sentence(sent)
        scored.append((s, sent))

    scored_perm = np.random.permutation(np.arange(len(scored)))
    scored = [scored[i] for i in scored_perm]
    if len(scored) < num_exemplars:
        scored*= (num_exemplars//len(scored))+1
    scored = scored[:num_exemplars]
    # 3) Sort by descending score, then by length (longer = more content)
    scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

    # 4) Take the top `num_exemplars`, but skip any that are too short (< 20 chars)
    exemplars: List[str] = []
    for score, sent in scored:
        if len(sent) < 20:
            continue
        exemplars.append(sent)
        if len(exemplars) >= num_exemplars:
            break

    # If we didn’t find enough, just pad with the first few sentences
    if len(exemplars) < num_exemplars:
        for sent in sentences:
            if sent not in exemplars and len(sent) >= 20:
                exemplars.append(sent)
            if len(exemplars) >= num_exemplars:
                break

    return exemplars[:num_exemplars]



