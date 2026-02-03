from datasets import load_dataset


def prep_gsm8k():
    gsm8k = load_dataset("gsm8k", "main")

    def process(example):
        return {
            "prompt": example["question"],
            "answer": example["answer"].split("####")[-1].strip(),
            "full_solution": example["answer"],
        }

    return gsm8k.map(process)
