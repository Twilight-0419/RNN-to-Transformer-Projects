import unittest

from src.tokenizer import BOS_ID, EOS_ID, UNK_ID, WordTokenizer


class WordTokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WordTokenizer(["春", "风", "明月", "，", "。"])

    def test_special_ids_are_stable(self) -> None:
        self.assertEqual(BOS_ID, 1)
        self.assertEqual(EOS_ID, 2)
        self.assertEqual(UNK_ID, 3)

    def test_encode_decode_round_trip(self) -> None:
        token_ids = self.tokenizer.encode_tokens(["春风", "明月", "。"])
        self.assertEqual(token_ids, [UNK_ID, 6, 8])
        self.assertEqual(self.tokenizer.decode(token_ids), "明月。")

    def test_prompt_falls_back_to_characters(self) -> None:
        token_ids = self.tokenizer.encode_prompt("春风")
        self.assertEqual(token_ids, [4, 5])


if __name__ == "__main__":
    unittest.main()
