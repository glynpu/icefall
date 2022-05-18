from pathlib import Path
from icefall.lexicon import read_lexicon
import sentencepiece as spm
import kenlm

def extract_start_tokens(lang_dir: Path = Path("./data/lang_bpe_500/"):
    tokens = read_lexicon(lang_dir / "/tokens.txt")

    # Get the  leading underscore of '▁THE 4'.
    # Actually its not a underscore, its just looks similar to it.
    word_start_char = tokens[4][0][0]

    word_start_token = []
    non_start_token = []

    aux=['<sos/eos>', '<unk>']
    for t in tokens:
        leading_char = t[0][0]
        if leading_char == word_start_char or t[0] in aux:
            word_start_token.append(t)
        else:
            non_start_token.append(t)

    write_lexicon(lang_dir / "word_start_tokens.txt", word_start_token)
    write_lexicon(lang_dir / "non_start_tokens.txt", non_start_token)

def lexicon_to_dict(lexicon):
    token2idx = {}
    idx2token = {}
    for token, idx in lexicon:
        assert len(idx) == 1
        idx = idx[0]
        token2idx[token] = int(idx)
        idx2token[int(idx)] = token
    return token2idx, idx2token


class LMRescorer:
    def __init__(self, lang_dir, blank_id, lm, weight):
        self.lm=lm
        self.start_token2idx, self.start_idx2token = lexicon_to_dict(read_lexicon(lang_dir/"word_start_tokens.txt"))
        self.nonstart_token2idx, self.nonstart_idx2token = lexicon_to_dict(read_lexicon(lang_dir/"non_start_tokens.txt"))
        self.token2idx, self.idx2token = lexicon_to_dict(read_lexicon(lang_dir/"tokens.txt"))
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(lang_dir/"bpe.model"))
        self.blank_id = blank_id
        self.weight = weight

    def rescore(self, hyp):
        if self.weight > 0 and hyp.ys[-1] in self.start_idx2token:
            word = self.previous_word(hyp)
            output_state= kenlm.State()
            lm_score = self.lm.BaseScore(hyp.state, word, output_state)
            hyp.state = output_state
            hyp.log_prob += self.weight * lm_score
        return hyp

    def previous_word(self, hyp):
        last_start_idx = hyp.last_start_idx
        tokens_seq = hyp.ys[last_start_idx: -1]
        tokens_seq = [t for t in tokens_seq if t!=self.blank_id]
        word = self.sp.decode(tokens_seq)
        hyp.last_start_idx = len(hyp.ys) - 1
        return word



