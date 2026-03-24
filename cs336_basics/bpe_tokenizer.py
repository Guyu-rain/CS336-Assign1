import os
from typing import BinaryIO
import regex as re
import collections
import pprint

DATA_PATH = "./data/TinyStoriesV2-GPT4-valid.txt"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def word_to_token_tuple(word: str) -> tuple[bytes, ...]:
    """
    将一个 pre-token 转换成初始byte token sequence
    input: 
        word(str)
    output:
        byte_token_sequence(tuple[bytes, ...])
    """
    byte_sequence = word.encode("utf-8")
    return tuple(bytes([b]) for b in byte_sequence)

def find_chunk_boundaries(
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

# serial version
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    merges = []
    
    # 1. 初始化 vocab(dict[int, bytes])
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

    # 2. 构建 pre-token 语料统计 corpus = {(b'l', b'o', b'w'): 5, ...}
    corpus = collections.defaultdict(int)

    with open(input_path, "r", encoding="utf-8") as f:
            ## 先进行 serial implementation
            # num_processes = 4
            # boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            # for start, end in zip(boundaries[:-1], boundaries[1:]):
            #     f.seek(start)
            #     chunk_bytes = f.read(end - start).decode("utf-8", errors="ignore")
            #     # 将 special tokens 去掉，不让其参与 BPE 训练
                
            #     # Run pre-tokenization on your chunk and store the counts for each pre-token
            #     word_iter = re.finditer(PAT, chunk_bytes)
            #     for match in word_iter:
            #         word = match.group()
            #         word_tuple = word_to_token_tuple(word)
            #         corpus[word_tuple] += 1
            
            # Serial Implementation
            text = f.read()
            # 1. 先将文本按 special tokens 切开
            if special_tokens:
                escaped = [re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)]
                special_token_pattern = "(" + "|".join(escaped) + ")"
                parts = re.split(special_token_pattern, text)
            else:
                parts = [text]

            # 2. 只对普通文本做 pre-word 统计
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # 普通文本
                    word_iter = re.finditer(PAT, part)
                    for match in word_iter:
                        word = match.group()
                        word_token_tuple = word_to_token_tuple(word)
                        corpus[word_token_tuple] += 1

            # 3. special token 只加入 vocab，不参与训练
            


    # 3. 反复 merge，直到 vocab_size 达到目标
    while len(vocab) < vocab_size:
        pair_counts = collections.defaultdict(int)
        # 3.1 统计所有相邻 pair 的频次
        for word_tuple, count in corpus.items():
            pre_token = word_tuple[0]
            for token in word_tuple[1:]:
                pair_counts[(pre_token, token)] += count
                pre_token = token
        # 3.2 选出最优 pair,选用字典序大的 tie-break
        # 如果所有 token sequence 长度都 <2，pair_counts为空，best_pair 这句对 pair_counts 取 items() 会炸
        if not pair_counts:
            break
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        first_token, second_token = best_pair[0]
        merge = b''.join([first_token, second_token])
        # 更新 merges
        merges += [best_pair[0]]
        # 更新 vocab
        vocab[vocab_index] = merge
        vocab_index += 1 
        # 3.3 将整个 corpus 有这个 pair 的 pre-token merge起来，然后更新 tuple, merge
        new_corpus = collections.defaultdict(int)
        for bytes_tokens, count in corpus.items():
            if len(bytes_tokens) < 2:
                new_corpus[bytes_tokens] += count
                continue
            
            new_tokens = []
            i = 0
            while i < len(bytes_tokens):
                if (
                    i < len(bytes_tokens) - 1 
                    and bytes_tokens[i] == first_token
                    and bytes_tokens[i + 1] == second_token
                ):
                    new_tokens.append(merge)
                    i += 2
                else:
                    new_tokens.append(bytes_tokens[i])
                    i += 1
            # 注意：两个原来不同的 corpus vocab 合并之后可能相同，因此要累加
            new_corpus[tuple(new_tokens)] += count
        corpus = new_corpus
    
    return vocab, merges
            

if __name__ == "__main__":
    vocab, merges = train_bpe(DATA_PATH, 300, ["<|endoftext|>"])
    print(vocab, merges)