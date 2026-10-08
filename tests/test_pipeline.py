import unittest

from descriptor import DESCRIPTOR_KEYS, compute_descriptors
from retrieval import RetrievedExample, prompt_examples


class PipelineTest(unittest.TestCase):
    def test_descriptor_api_and_class_covered_retrieval(self):
        self.assertEqual(set(compute_descriptors("CCO")), set(DESCRIPTOR_KEYS))

        item = lambda index, label: RetrievedExample(index, index, label, f"S{index}", 1 / index)
        primary = [item(1, "A"), item(2, "B"), item(3, "A"), item(4, "C")]
        self.assertEqual(len(prompt_examples(primary, item(5, "B"), "B", 4)[0]), 4)
        self.assertEqual(len(prompt_examples(primary, item(5, "D"), "D", 4)[0]), 5)


if __name__ == "__main__":
    unittest.main()
