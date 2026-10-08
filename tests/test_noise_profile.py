import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.noise_profile import reduce_with_profile, seconds


class NoiseProfileTest(unittest.TestCase):
    def test_time_formats_and_invalid_ranges(self):
        self.assertEqual(seconds("00：10"), 10)
        self.assertEqual(seconds("1:02:03.5"), 3723.5)
        for invalid in ("nan", "inf", "-1", "1:60", "1:2:3:4"):
            with self.assertRaises(ValueError):
                seconds(invalid)

    def test_profile_reduces_known_stationary_noise_without_shifting_audio(self):
        # これは定常ノイズの機能テスト。実際のざわめき・発話の改善を証明するものではない。
        rate = 16000
        t = np.arange(rate * 3) / rate
        clean = np.where(t < 1, 0, .25 * np.sin(2 * np.pi * 600 * t))
        mixed = clean + np.random.default_rng(123).normal(0, .03, len(t))
        with tempfile.TemporaryDirectory() as tmp:
            src, out, removed = [Path(tmp) / n for n in ("in.wav", "out.wav", "removed.wav")]
            sf.write(src, mixed, rate, subtype="PCM_16")
            original, _ = sf.read(src)
            reduce_with_profile(src, out, removed, 0, 1, .35)
            reduced, r = sf.read(out)
            residue, _ = sf.read(removed)
            self.assertEqual((len(reduced), r), (len(original), rate))
            np.testing.assert_allclose(reduced + residue, original, atol=2 / 32768)
            self.assertLess(np.mean(reduced[:rate] ** 2), np.mean(original[:rate] ** 2))
            np.testing.assert_array_equal(sf.read(src)[0], original)
            with self.assertRaisesRegex(ValueError, "上書き"):
                reduce_with_profile(src, src, removed, 0, 1)

    def test_silent_or_out_of_range_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.wav"
            sf.write(src, np.zeros(16000), 16000)
            for a, b in [(0, .2), (0, 2), (0, .5)]:
                with self.assertRaises(ValueError):
                    reduce_with_profile(src, Path(tmp) / "out.wav", Path(tmp) / "removed.wav", a, b)


if __name__ == "__main__":
    unittest.main()
