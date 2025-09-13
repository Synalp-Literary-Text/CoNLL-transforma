import argparse
from bertalign import Bertalign
import os


def read_conll(filepath):
    with open(filepath, encoding='utf-8') as f:
        lines = f.read().strip().split('\n\n')
        sents = []
        for block in lines:
            tokens = [line.split()[0] for line in block.strip().splitlines()]
            labels = [line.split()[1] for line in block.strip().splitlines()]
            sents.append((tokens, labels))
        return sents


def get_sent_strings(sents):
    return [' '.join(tokens) for tokens, _ in sents]


def convert_alignment_numpy_to_int(alignment):
    new_alignment = []
    for src_group, tgt_group in alignment:
        src_ids = [int(x) for x in src_group]
        tgt_ids = [int(x) for x in tgt_group]
        new_alignment.append((src_ids, tgt_ids))
    return new_alignment


def merge_sentences(sents, indices):
    tokens, labels = [], []
    for idx in indices:
        tokens += sents[idx][0]
        labels += sents[idx][1]
    return tokens, labels

# Writing new CoNLL file
def write_conll(sentences, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding='utf-8') as f:
        for sent in sentences:
            for token, label in sent:
                f.write(f"{token}\t{label}\n")
            f.write('\n')


def main(en_path, fr_path, output_prefix):
    # Load input
    en_sents = read_conll(en_path)
    fr_sents = read_conll(fr_path)

    en_raw = get_sent_strings(en_sents)
    fr_raw = get_sent_strings(fr_sents)

    # Aligning sentences with Bertalign
    aligner = Bertalign('\n'.join(en_raw), '\n'.join(fr_raw), is_split=True)
    aligner.align_sents()

    raw_alignment = aligner.result
    alignment = convert_alignment_numpy_to_int(raw_alignment) # treating np.zero(64)

    aligned_en, aligned_fr = [], []

    for src_ids, tgt_ids in alignment:
        en_tokens, en_labels = merge_sentences(en_sents, src_ids)
        fr_tokens, fr_labels = merge_sentences(fr_sents, tgt_ids)
        aligned_en.append(list(zip(en_tokens, en_labels)))
        aligned_fr.append(list(zip(fr_tokens, fr_labels)))

    # Save output
    write_conll(aligned_en, f"{output_prefix}_aligned_en.conll")
    write_conll(aligned_fr, f"{output_prefix}_aligned_fr.conll")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align two CoNLL files using Bertalign.")
    parser.add_argument("--sl", required=True, help="Path to the Source Language CoNLL file")
    parser.add_argument("--tl", required=True, help="Path to the Target Language CoNLL file")
    parser.add_argument("--out", required=True, help="Prefix for output files")

    args = parser.parse_args()
    main(args.sl, args.tl, args.out)
