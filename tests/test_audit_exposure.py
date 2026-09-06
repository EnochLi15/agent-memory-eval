import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from audit_exposure import audit


class ExposureTests(unittest.TestCase):
    def test_all_variants_excluded_without_reading_candidate_contents(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / 'source'
            source.mkdir()
            for name in ('A01_update.json', 'A01_reflect.json', 'A02_forget.json'):
                (source/name).write_bytes(b'not even valid JSON; must not be parsed')
            log = root/'requests.jsonl'
            log.write_text(json.dumps({'user_id': 'run:memops:A01_update'}))
            report = audit(source, [log])
            self.assertEqual(report['recorded_exposed_groups'], ['A01'])
            self.assertEqual(report['excluded_files'], ['A01_reflect.json', 'A01_update.json'])
            self.assertEqual(report['no_recorded_exposure_groups'], ['A02'])
            self.assertFalse(report['candidate_contents_read'])
            self.assertEqual(audit(source, [log], ['A02'])['no_recorded_exposure_groups'], [])
