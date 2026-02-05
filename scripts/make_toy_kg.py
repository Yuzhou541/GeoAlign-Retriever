import argparse
from pathlib import Path

TOY_TRIPLES = [
    ("cat", "is_a", "animal"),
    ("dog", "is_a", "animal"),
    ("kitten", "is_a", "cat"),
    ("puppy", "is_a", "dog"),
    ("animal", "part_of", "nature"),
    ("cat", "likes", "milk"),
    ("dog", "likes", "bone"),
    ("kitten", "likes", "milk"),
    ("puppy", "likes", "bone"),
    ("milk", "is_a", "drink"),
    ("bone", "is_a", "food"),
    ("drink", "is_a", "thing"),
    ("food", "is_a", "thing"),
    ("nature", "is_a", "thing"),
]

ENTITY_TEXTS = {
    "cat": "a small domesticated carnivorous mammal; a cat",
    "dog": "a domesticated carnivorous mammal; a dog",
    "kitten": "a young cat; kitten",
    "puppy": "a young dog; puppy",
    "animal": "a living organism that feeds on organic matter; animal",
    "nature": "the physical world collectively, including plants and animals; nature",
    "milk": "a white nutritious liquid produced by mammals; milk",
    "bone": "a rigid organ that forms part of the skeleton; bone",
    "drink": "a liquid that can be consumed; drink",
    "food": "a substance consumed to provide nutritional support; food",
    "thing": "an object or entity; thing",
}

def write_triples(path: Path, triples):
    with path.open("w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/toykg")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = TOY_TRIPLES[:10]
    valid = TOY_TRIPLES[10:12]
    test  = TOY_TRIPLES[12:]

    write_triples(out / "train.txt", train)
    write_triples(out / "valid.txt", valid)
    write_triples(out / "test.txt", test)

    with (out / "entity_texts.tsv").open("w", encoding="utf-8") as f:
        for e, txt in ENTITY_TEXTS.items():
            f.write(f"{e}\t{txt}\n")

    queries = [
        ("a young cat", "kitten"),
        ("a young dog", "puppy"),
        ("white nutritious liquid", "milk"),
        ("domesticated carnivorous mammal dog", "dog"),
        ("physical world including plants and animals", "nature"),
    ]
    with (out / "queries.tsv").open("w", encoding="utf-8") as f:
        for q, ans in queries:
            f.write(f"{q}\t{ans}\n")

    print(f"[OK] wrote toy KG to {out}")

if __name__ == "__main__":
    main()
