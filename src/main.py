from transformers import AutoTokenizer, AutoModel
from core.analysis.profiling.profiler import Profiler
import json
from core.analysis.profiling.models.llm import (
    AnalyzeConnections,
    EstimateMemory,
    MeasureInferenceTime,
)
from core.analysis.profiling.models.profiler.llm_profile_options import (
    LLMProfilerOptions,
)
import random
import string
from functools import partial
from loguru import logger


model_name = "bert-base-uncased"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModel.from_pretrained(model_name)

profiler = Profiler(
    model=model, model_name=model_name, tokenizer=tokenizer, verbose=True
)


def generate_bert_prompt(prompt_type="classification", length="medium", seed=None):
    if seed is not None:
        random.seed(seed)

    subjects = [
        "The researcher",
        "A student",
        "The company",
        "Scientists",
        "The team",
        "Engineers",
        "The algorithm",
        "Data analysts",
        "The model",
        "Developers",
    ]

    verbs = [
        "developed",
        "analyzed",
        "created",
        "implemented",
        "discovered",
        "tested",
        "optimized",
        "evaluated",
        "designed",
        "processed",
    ]

    objects = [
        "machine learning models",
        "neural networks",
        "data structures",
        "algorithms",
        "software systems",
        "research methods",
        "experiments",
        "computational frameworks",
        "optimization techniques",
        "AI systems",
    ]

    adjectives = [
        "efficient",
        "robust",
        "innovative",
        "scalable",
        "accurate",
        "sophisticated",
        "advanced",
        "reliable",
        "powerful",
        "intelligent",
    ]

    contexts = [
        "in artificial intelligence",
        "for data processing",
        "in computer science",
        "for optimization",
        "in machine learning",
        "for research purposes",
        "in software engineering",
        "for computational analysis",
    ]

    length_params = {"short": (3, 8), "medium": (8, 15), "long": (15, 25)}

    min_sentences, max_sentences = length_params.get(length, (8, 15))
    num_sentences = random.randint(min_sentences, max_sentences)

    def generate_sentence():
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        adj = random.choice(adjectives)
        context = random.choice(contexts)

        patterns = [
            f"{subject} {verb} {adj} {obj} {context}.",
            f"{subject} successfully {verb} {obj} using {adj} methods.",
            f"The {adj} {obj} were {verb} by {subject.lower()} {context}.",
            f"{subject} {verb} and evaluated {adj} {obj} {context}.",
        ]

        return random.choice(patterns)

    def generate_classification_prompt():
        sentences = [generate_sentence() for _ in range(num_sentences)]
        return " ".join(sentences)

    def generate_pair_prompt():
        sentences1 = [generate_sentence() for _ in range(num_sentences // 2)]
        sentences2 = [
            generate_sentence() for _ in range(num_sentences - len(sentences1))
        ]

        text1 = " ".join(sentences1)
        text2 = " ".join(sentences2)

        return f"{text1} [SEP] {text2}"

    def generate_masked_prompt():
        sentences = []
        for _ in range(num_sentences):
            sentence = generate_sentence()
            words = sentence.split()

            mask_count = random.randint(1, min(2, len(words) - 1))
            mask_positions = random.sample(range(1, len(words) - 1), mask_count)

            for pos in mask_positions:
                words[pos] = "[MASK]"

            sentences.append(" ".join(words))

        return " ".join(sentences)

    if prompt_type == "classification":
        return generate_classification_prompt()
    elif prompt_type == "pair":
        return generate_pair_prompt()
    elif prompt_type == "masked":
        return generate_masked_prompt()
    elif prompt_type == "mixed":
        chosen_type = random.choice(["classification", "pair", "masked"])
        return generate_bert_prompt(chosen_type, length, seed)
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


def generate_random_string_prompt(min_length=50, max_length=200, seed=None):
    if seed is not None:
        random.seed(seed)

    length = random.randint(min_length, max_length)

    text = ""
    word_length = 0

    for _ in range(length):
        if word_length > random.randint(3, 12):
            text += " "
            word_length = 0
        else:
            text += random.choice(string.ascii_letters)
            word_length += 1

    return text.strip()


def get_inference_prompt(prompt_type="classification", length="medium", seed=None):
    return generate_bert_prompt(prompt_type=prompt_type, length=length, seed=seed)


fn = partial(get_inference_prompt, prompt_type="mixed", length="long", seed=None)

llm_profiler_opts = LLMProfilerOptions(
    analyze_connections=AnalyzeConnections(
        input_shape=(1, 512),
    ),
    estimate_memory=EstimateMemory(batch_size=1, sequence_length=512),
    measure_inference_time=MeasureInferenceTime(
        input_sample=fn, num_runs=20, warmup_runs=2, tokenizer_max_length=512
    ),
)

llm_info = profiler.profile_complete(model_profiler_options=llm_profiler_opts)

logger.debug(llm_info)

with open("test2.json", "w") as file:
    json.dump(llm_info.model_dump(), file, indent=4, default=str)
