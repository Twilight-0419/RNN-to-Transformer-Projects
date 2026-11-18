import unittest
from pathlib import Path

from src.text import (
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    decode_english,
    encode_chinese,
    encode_english,
    load_english_bpe,
    normalize_english,
    special_id,
)


ROOT = Path(__file__).resolve().parents[1]


class TextPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = load_english_bpe(
            ROOT / "models" / "bpe_vocab.json",
            ROOT / "models" / "bpe_merges.txt",
        )

    def test_special_ids_are_valid(self) -> None:
        ids = [
            special_id(self.tokenizer, "<pad>"),
            special_id(self.tokenizer, "<sos>"),
            special_id(self.tokenizer, "<eos>"),
        ]
        self.assertEqual(len(set(ids)), 3)
        self.assertTrue(all(token_id >= 10000 for token_id in ids))

    def test_english_normalization_and_bpe_round_trip(self) -> None:
        text = "I don't know what to do."
        self.assertEqual(normalize_english(text), "i don 't know what to do .")
        ids = encode_english(text, self.tokenizer)
        self.assertEqual(ids[-1], special_id(self.tokenizer, "<eos>"))
        self.assertEqual(decode_english(ids, self.tokenizer), "I don't know what to do.")

    def test_chinese_encoding_adds_eos(self) -> None:
        vocab = {"PAD": PAD_TOKEN, "SOS": SOS_TOKEN, "EOS": EOS_TOKEN, "UNK": UNK_TOKEN, "我": 4}
        ids = encode_chinese("我爱你。", vocab, max_length=4)
        self.assertEqual(ids[-1], EOS_TOKEN)
        self.assertLessEqual(len(ids), 4)


if __name__ == "__main__":
    unittest.main()
