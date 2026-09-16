import tempfile
import unittest
from pathlib import Path

from tools.run_ttt_pilot import is_evaluation_complete, load_seqmap


class OfficialTTTEvaluationTests(unittest.TestCase):
    def test_seqmap_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expressions = root / 'expression'
            for video, name in [('0005', 'first'), ('0011', 'second')]:
                path = expressions / video / f'{name}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            seqmap = root / 'seqmap.txt'
            seqmap.write_text('0005+first\n0011+second\n')
            selected = load_seqmap(seqmap, expressions)
            self.assertEqual([(p.parent.name, p.stem) for p in selected],
                             [('0005', 'first'), ('0011', 'second')])

    def test_seqmap_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expression = root / 'expression' / '0005' / 'first.json'
            expression.parent.mkdir(parents=True)
            expression.write_text('{}')
            seqmap = root / 'seqmap.txt'
            seqmap.write_text('0005+first\n0005+first\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_seqmap(seqmap, root / 'expression')

    def test_resume_requires_prediction_and_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expression = root / 'expression' / '0005' / 'first.json'
            expression.parent.mkdir(parents=True)
            expression.write_text('{}')
            result = root / 'results_structured' / '0005' / 'first'
            result.mkdir(parents=True)
            (result / 'predict.txt').write_text('')
            self.assertFalse(is_evaluation_complete(root, 'structured', expression))
            (result / 'gt.txt').write_text('ground truth')
            self.assertTrue(is_evaluation_complete(root, 'structured', expression))


if __name__ == '__main__':
    unittest.main()
