import os
import collections
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from __init__ import DATA_PATH, PAT

# 辅助函数
def word_to_token_tuple(
    word: str
) -> tuple[bytes, ...]:
    """ 将一个 pre-token 转换成初始byte token sequence """
    byte_sequence = word.encode("utf-8")
    return tuple(bytes([b]) for b in byte_sequence)

def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)] 
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        # bi: [0, 1, ..., desired_num_chunks] * chunk_size
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)

            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

"""
    Stage 1: 并行 pre-tokenize
"""
def _pretokenize_chunk(
    args: tuple[str, int, int, list[str]]
) -> dict[tuple[bytes, ...], int]:
    """
    读取文件的 [start, end) 字节范围，对普通文本段做 pre-tokenization，返回局部 corpus dict
    """

    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start).decode("utf-8", errors="ignore")
    
    local_corpus: dict[tuple[bytes, ...], int] = collections.defaultdict(int)

    if special_tokens:
        escaped = [re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)]
        pattern = "(" + "|".join(escaped) + ")"
        parts = re.split(pattern, chunk_bytes)
    else:
        parts = [chunk_bytes]
    
    for i, part in enumerate(parts):
        if i % 2 == 0: # 普通文本
            for match in re.finditer(PAT, part):
                word_tuple = word_to_token_tuple(match.group())
                local_corpus[word_tuple] += 1
    
    return dict(local_corpus)

def _merge_corpus_dicts(
    dicts: list[dict]
) -> dict[tuple[bytes, ...], int]:
    merged: dict[tuple[bytes, ...], int] = collections.defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return merged

def parallel_pretokenize(
    input_path: str,
    special_tokens: list[str],
    num_workers: int
) -> dict[tuple[bytes, ...], int]:
    """ 多进程 并行读取文件 + 统计 pre-token 频次 """
    with open(input_path, "rb") as f:
        # 用第一个 special token 作为 chunk 分割点
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n\n"
        boundaries = _find_chunk_boundaries(f, num_workers, split_token)

    chunks_boundaries = list(zip(boundaries[: -1], boundaries[1: ]))
    args = [(input_path, start, end, special_tokens) for start, end in chunks_boundaries]

    with Pool(processes=num_workers) as pool:
        partial_corpora = pool.map(_pretokenize_chunk, args)
    
    return _merge_corpus_dicts(partial_corpora)


"""
    Stage 2: 构建 pair_counts + merge 维护 corpus
"""
def _build_pair_counts(
    corpus: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:
    """ 得到 pre-tokens corpus, 从头统计所有 pair counts """
    pair_counts: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    for token_seq, freq in corpus.items():
        for a, b in zip(token_seq, token_seq[1:]):
            pair_counts[(a, b)] += freq
    return pair_counts

def _apply_merge(
    corpus: dict[tuple[bytes, ...], int],
    pair_counts: dict[tuple[bytes, bytes], int],
    best_pair: tuple[bytes, bytes]
) -> dict[tuple[bytes, ...], int]:
    """
    将 corpus 中所有包含 best_pair 的词条进行 merge
    增量更新 pair_count
    """
    first, second = best_pair
    merged_token = first + second

    new_corpus: dict[tuple[bytes, ...], int] = {}

    for token_seq, freq in corpus.items():
        # token_seq 不含 best_pair，直接保留
        if first not in token_seq or (
            not any(
                token_seq[i] == first and token_seq[i + 1] == second
                for i in range(len(token_seq) - 1)
            )
        ):
            new_corpus[token_seq] = freq
            continue
        # 包含 best_pair
        # 构建新 token_seq 并 增量更新 pair_counts
        new_seq: list[bytes] = []
        i = 0
        while i < len(token_seq):
            if (
                i < len(token_seq) - 1
                and token_seq[i] == first
                and token_seq[i + 1] == second
            ):
                # 消去旧的单个相邻 pair
                ## (token_seq[i - 1], first) 如果存在，消去
                if new_seq:
                    pair_counts[(new_seq[-1], first)] -= freq
                    pair_counts[(new_seq[-1], merged_token)] += freq
                # (second, token_seq[i + 2]) 如果存在
                if i + 2 < len(token_seq):
                    pair_counts[(second, token_seq[i + 2])] -= freq
                    pair_counts[(merged_token, token_seq[i + 2])] += freq
                
                # 消去 best_pair 本身
                pair_counts[best_pair] -= freq

                new_seq.append(merged_token)
                i += 2
            else:
                new_seq.append(token_seq[i])
                i += 1
        new_key = tuple(new_seq)
        # 两个 pre-token 合并后，可能产生相同的新词条
        # 累加
        new_corpus[new_key] = new_corpus.get(new_key, 0) + freq

    return new_corpus


"""
    train_bpe
"""
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """ 
    Args:
        input_path:     训练语料文件路径
        vocab_size:     目标词表大小
        special_tokens: special tokens 列表（如 ["<|endoftext|>"]）
        num_workers:    并行进程数，None 则自动取 CPU 核数
 
    Returns:
        vocab:   dict[int, bytes] 词表
        merges:  list[tuple[bytes, bytes]] merge 规则列表（有序）
    """
    if num_workers is None:
        num_workers = cpu_count()
 
    # ------------------------------------------------------------------
    # 1. 初始化 vocab
    # ------------------------------------------------------------------
    vocab: dict[int, bytes] = {}
    vocab_index = 0
 
    for token in special_tokens:
        vocab[vocab_index] = token.encode("utf-8")
        vocab_index += 1
 
    for i in range(256):
        vocab[vocab_index] = bytes([i])
        vocab_index += 1
 
    min_vocab_size = len(special_tokens) + 256
    if vocab_size < min_vocab_size:
        raise ValueError(f"vocab_size must be at least {min_vocab_size}")
 
    num_merges = vocab_size - min_vocab_size
    if num_merges == 0:
        return vocab, []
 
    # ------------------------------------------------------------------
    # 2. 并行 pre-tokenization → corpus
    # ------------------------------------------------------------------
    corpus = parallel_pretokenize(input_path, special_tokens, num_workers)
 
    # ------------------------------------------------------------------
    # 3. 初始化 pair_counts（只做一次完整扫描）
    # ------------------------------------------------------------------
    pair_counts = _build_pair_counts(corpus)
 
    # ------------------------------------------------------------------
    # 4. Merge 循环（增量维护 pair_counts）
    # ------------------------------------------------------------------
    merges: list[tuple[bytes, bytes]] = []
 
    for _ in tqdm(range(num_merges), desc="Training BPE", unit="merge"):
        if not pair_counts:
            break
 
        # 过滤掉计数 <= 0 的 pair（增量更新可能产生零/负值）
        # 选出最优 pair：先按频次，再按字典序 tie-break
        best_pair = max(
            ((p, c) for p, c in pair_counts.items() if c > 0),
            key=lambda x: (x[1], x[0]),
            default=None,
        )
        if best_pair is None:
            break
 
        best_pair_key = best_pair[0]
        merges.append(best_pair_key)
 
        # 更新 vocab
        vocab[vocab_index] = best_pair_key[0] + best_pair_key[1]
        vocab_index += 1
 
        # 增量更新 corpus 和 pair_counts
        corpus = _apply_merge(corpus, pair_counts, best_pair_key)
 
    return vocab, merges

if __name__ == "__main__":
    import time
    print(os.listdir())
    start = time.perf_counter()
    vocab, merges = train_bpe(DATA_PATH, 10000, ["<|endoftext|>"])
    elapsed = time.perf_counter() - start

    # 准备存储的数据
    import json
    import base64

    def to_visual(b: bytes) -> str:
        # 尝试解码为 utf-8，失败则显示十六进制
        try:
            return b.decode('utf-8')
        except UnicodeDecodeError:
            return str(b)

    def save_tokenizer(vocab, merges, filename="./data/tokenizer_result/vocab.json"):
        data = {
            "vocab": {},
            "merges": []
        }
        
        # 处理 Vocab
        for idx, b in vocab.items():
            data["vocab"][idx] = {
                "bytes": base64.b64encode(b).decode('utf-8'),
                "display": to_visual(b)
            }
        
        # 处理 Merges
        for p1, p2 in merges:
            res = p1 + p2
            data["merges"].append({
                "pair": [
                    base64.b64encode(p1).decode('utf-8'),
                    base64.b64encode(p2).decode('utf-8')
                ],
                "visual": f"{to_visual(p1)} + {to_visual(p2)} -> {to_visual(res)}"
            })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    save_tokenizer(vocab, merges)

    print(f"Done in {elapsed:.2f}s — vocab size: {len(vocab)}, merges: {len(merges)}")
    print("\nFirst 10 merges:")
    for a, b in merges[:10]:
        print(f"  {a!r} + {b!r} → {(a+b)!r}")
