import unittest
import markovify
import os


class MarkovifyTest(unittest.TestCase):
    def test_simple(self) -> None:
        with open(os.path.join(os.path.dirname(__file__), "texts/sherlock.txt")) as f:
            sherlock_model = markovify.Text(f)
        sent = sherlock_model.make_sentence()
        assert sent is not None
        assert len(sent) != 0

    def test_without_retaining(self) -> None:
        with open(
            os.path.join(os.path.dirname(__file__), "texts/senate-bills.txt"),
            encoding="utf-8",
        ) as f:
            senate_model = markovify.Text(f, retain_original=False)
        sent = senate_model.make_sentence()
        assert sent is not None
        assert len(sent) != 0

    def test_from_json_without_retaining(self) -> None:
        with open(
            os.path.join(os.path.dirname(__file__), "texts/senate-bills.txt"),
            encoding="utf-8",
        ) as f:
            original_model = markovify.Text(f, retain_original=False)
        d = original_model.to_json()
        new_model = markovify.Text.from_json(d)
        sent = new_model.make_sentence()
        assert sent is not None
        assert len(sent) != 0

    def test_from_mult_files_without_retaining(self) -> None:
        models = []
        for dirpath, _, filenames in os.walk(
            os.path.join(os.path.dirname(__file__), "texts")
        ):
            for filename in filenames:
                with open(
                    os.path.join(dirpath, filename),
                    encoding="utf-8",
                ) as f:
                    models.append(markovify.Text(f, retain_original=False))
        combined_model = markovify.combine(models)
        sent = combined_model.make_sentence()
        assert sent is not None
        assert len(sent) != 0


if __name__ == "__main__":
    unittest.main()
