"""指定したノイズ区間を見本として、音声全編を弱く除去。元音声は保持する。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.noise_profile import create_trial

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--start", required=True, help="見本の開始。秒数・分:秒・時:分:秒")
    p.add_argument("--end", required=True)
    p.add_argument("--strength", type=float, default=0.35)
    p.add_argument("--output", type=Path, required=True, help="新しい保存フォルダ")
    a = p.parse_args()
    result = create_trial(a.audio, a.start, a.end, a.output, a.strength,
                          progress=lambda f: print(f"{f:.0%}", flush=True))
    print(json.dumps(result, ensure_ascii=False, indent=2))
