import sys
import argparse
import scipy
from sentence_transformers import SentenceTransformer


def main(args):
    embedder = SentenceTransformer('distiluse-base-multilingual-cased')

    with open(args.hyp_path) as fr1, open(args.ref_path) as fr2:
        total_sim = 0
        for i, (line1, line2) in enumerate(zip(fr1,fr2)):
            _, hyp = line1.strip().split('\t')
            word, _, ref = line2.strip().split('\t')
            hyp_embedding = embedder.encode([hyp])
            ref_embedding = embedder.encode([ref])
            sim = 1 - scipy.spatial.distance.cdist(hyp_embedding, ref_embedding, 'cosine')[0][0]
            total_sim += sim
        print(total_sim/(i+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp_path', type=str, help='path of hypothese file')
    parser.add_argument('--ref_path', type=str, help='path of reference file')
    args = parser.parse_args()
    sys.exit(main(args))