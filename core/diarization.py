"""話者分離と声紋識別

speechbrain (ECAPA-TDNN) + torch が導入されている場合のみ動作する。
未導入の場合 available() が False を返し、アプリは話者分離なしで動く。

方式:
  1. Whisperの単語タイムスタンプから発話区間を作る
  2. 発話区間を 1.5秒窓 / 0.75秒ホップで切り、各窓の声紋ベクトルを抽出（GPU可）
  3. 窓を約3秒ずつ平均 → 固有値ギャップで話者数を自動推定 → スペクトルクラスタリング
  4. 声紋DBと照合して既知の人物名を付与（1人物は1クラスタにのみ対応）
  5. 各「単語」の中心時刻に最も近い窓の話者を割り当てる → 発話途中の話者交代も拾える
"""
import importlib.util
import logging
import re
from pathlib import Path

from .config import VOICEPRINTS_DIR

_classifier = None
_classifier_device = None

SAMPLE_RATE = 16000
WIN_SEC = 1.5
HOP_SEC = 0.75
MIN_WIN_SEC = 0.5        # これより短い発話区間は声紋を取らない（前後の窓で補完）
REGION_GAP_SEC = 0.6     # 単語間の隙間がこれ以下なら同じ発話区間とみなす
MIN_CLUSTER_SEC = 3.0    # 合計発話がこれ未満のクラスタは近いクラスタへ吸収（ノイズ対策）
MAX_PROFILE_SAMPLES = 60  # 1人あたり保存する声紋ベクトル数の上限


def available() -> bool:
    return (importlib.util.find_spec("speechbrain") is not None
            and importlib.util.find_spec("torch") is not None)


def device_label() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_classifier():
    global _classifier, _classifier_device
    if _classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _classifier_device = device_label()
        logging.info(f"話者埋め込みモデル (ECAPA-TDNN) をロード中... (device={_classifier_device})")
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home() / ".cache" / "speechbrain_ecapa"),
            run_opts={"device": _classifier_device},
        )
    return _classifier


def unload() -> None:
    global _classifier
    _classifier = None
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_wav(wav_path: str):
    import soundfile as sf
    data, rate = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise ValueError(f"想定外のサンプルレートです: {rate}")
    return data


def _embed_chunks(chunks: list, batch_size: int = 64):
    """音声チャンク（np.ndarray の list）→ 正規化済み声紋ベクトル (N, D)"""
    import numpy as np
    import torch

    classifier = _get_classifier()
    out = []
    for b in range(0, len(chunks), batch_size):
        batch = chunks[b:b + batch_size]
        max_len = max(len(c) for c in batch)
        padded = np.zeros((len(batch), max_len), dtype="float32")
        lens = np.zeros(len(batch), dtype="float32")
        for i, c in enumerate(batch):
            padded[i, :len(c)] = c
            lens[i] = len(c) / max_len
        with torch.no_grad():
            emb = classifier.encode_batch(
                torch.from_numpy(padded).to(_classifier_device),
                torch.from_numpy(lens).to(_classifier_device),
            )
        out.append(emb.squeeze(1).cpu().numpy())
    vecs = np.concatenate(out, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


# =====================================================================
# 話者分離
# =====================================================================

def _speech_regions(segments: list[dict]) -> list[tuple[float, float]]:
    """単語タイムスタンプから発話区間 [(start, end)] を作る"""
    spans = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            spans.extend((w["start"], w["end"]) for w in words if w["end"] > w["start"])
        elif seg["end"] > seg["start"]:
            spans.append((seg["start"], seg["end"]))
    spans.sort()
    regions = []
    for s, e in spans:
        if regions and s - regions[-1][1] <= REGION_GAP_SEC:
            regions[-1][1] = max(regions[-1][1], e)
        else:
            regions.append([s, e])
    return [(s, e) for s, e in regions]


def _make_windows(regions) -> list[tuple[float, float]]:
    windows = []
    for s, e in regions:
        length = e - s
        if length < MIN_WIN_SEC:
            continue
        if length <= WIN_SEC:
            windows.append((s, e))
            continue
        t = s
        while t + WIN_SEC <= e:
            windows.append((t, t + WIN_SEC))
            t += HOP_SEC
        if e - windows[-1][1] > HOP_SEC * 0.5:
            windows.append((e - WIN_SEC, e))
    return windows


def _chunk_windows(vecs, windows, per: int = 3):
    """連続する窓を per 個ずつ平均して約3秒の声紋にする（1.5秒窓単体より安定）。
    Returns: (チャンク声紋 (N, D), 各チャンクに属する窓indexのリスト)"""
    import numpy as np

    groups, cur = [], []
    for i in range(len(windows)):
        if cur and (windows[i][0] - windows[cur[-1]][1] > 0.5 or len(cur) >= per):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    cv = np.stack([vecs[g].mean(axis=0) for g in groups])
    cv /= np.maximum(np.linalg.norm(cv, axis=1, keepdims=True), 1e-8)
    return cv, groups


def _pruned_affinity(sim, p: int):
    """各点の上位 p 近傍だけを残した対称な類似度グラフ"""
    import numpy as np

    n = len(sim)
    aff = np.zeros_like(sim)
    idx = np.argsort(-sim, axis=1)[:, :p]
    aff[np.repeat(np.arange(n), p), idx.ravel()] = 1.0
    return (aff + aff.T) / 2.0


def _laplacian_eig(aff):
    import numpy as np

    lap = np.diag(aff.sum(axis=1)) - aff
    return np.linalg.eigh(lap)


def _estimate_speakers(emb, max_speakers: int):
    """話者数の自動推定（NME-SC: 近傍数 p を振り、固有値ギャップが最も明確な p を採用）。
    Returns: (話者数, そのときの固有ベクトル)"""
    import numpy as np

    n = len(emb)
    sim = emb @ emb.T
    np.fill_diagonal(sim, 1.0)
    k_max = min(max_speakers, n - 1)
    p_min = max(4, int(np.sqrt(n)))
    p_list = sorted({min(n - 1, max(p_min, int(n * r)))
                     for r in (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2)})
    best = None
    for p in p_list:
        vals, vecs = _laplacian_eig(_pruned_affinity(sim, p))
        gaps = np.diff(vals[:k_max + 1])
        k = int(np.argmax(gaps)) + 1
        ratio = p / (gaps.max() / (vals[-1] + 1e-10) + 1e-10)
        if best is None or ratio < best[0]:
            best = (ratio, k, vecs)
    return best[1], best[2]


def _spectral_cluster(cv, num_speakers: int, max_speakers: int):
    """チャンク声紋 → ラベル。人数未指定なら複数サンプルで推定して中央値を採る。"""
    import numpy as np
    from sklearn.cluster import KMeans

    n = len(cv)
    if n < 6:
        return np.zeros(n, dtype=int)
    max_points = 2500  # 固有値分解のコスト上限（2500点で1回約9秒）
    rng = np.random.default_rng(0)

    def sample():
        return np.sort(rng.choice(n, max_points, replace=False)) if n > max_points else np.arange(n)

    if num_speakers and num_speakers > 0:
        k = min(num_speakers, n)
    else:
        runs = 5 if n > max_points else 1
        k = int(np.median([_estimate_speakers(cv[sample()], max_speakers)[0] for _ in range(runs)]))
    if k <= 1:
        return np.zeros(n, dtype=int)

    sub = sample()
    emb = cv[sub]
    sim = emb @ emb.T
    np.fill_diagonal(sim, 1.0)
    _, spec = _laplacian_eig(_pruned_affinity(sim, max(4, int(np.sqrt(len(emb))), int(len(emb) * 0.05))))
    feats = spec[:, :k]
    feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
    sub_labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(feats)

    # サンプルで得た重心に全チャンクを割り当て、重心の更新を数回（k-means）
    cents = np.stack([emb[sub_labels == c].mean(axis=0) for c in range(k)])
    for _ in range(10):
        cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-8)
        labels = (cv @ cents.T).argmax(axis=1)
        new = np.stack([cv[labels == c].mean(axis=0) if (labels == c).any() else cents[c]
                        for c in range(k)])
        if np.allclose(new, cents):
            break
        cents = new
    return labels


def _merge_tiny(labels, vecs, windows, fixed_count: bool):
    """発話がごく短いクラスタ（咳・雑音・混信など）を最も近い話者へ吸収する"""
    import numpy as np

    if fixed_count:
        return labels
    durations = np.array([e - s for s, e in windows]) * (HOP_SEC / WIN_SEC)
    total = durations.sum()
    limit = max(MIN_CLUSTER_SEC, total * 0.005)
    for _ in range(3):
        cents = {int(c): vecs[labels == c].mean(axis=0) for c in np.unique(labels)}
        small = [c for c in cents if durations[labels == c].sum() < limit]
        big = [c for c in cents if c not in small]
        if not small or not big:
            break
        mat = np.stack([cents[b] / (np.linalg.norm(cents[b]) or 1.0) for b in big], axis=1)
        for c in small:
            idx = np.where(labels == c)[0]
            labels[idx] = np.array(big)[(vecs[idx] @ mat).argmax(axis=1)]
    return labels


def _smooth(labels, min_run: int = 2):
    """1窓だけ話者が飛ぶようなチラつきを前後に合わせて除去"""
    labels = labels.copy()
    n = len(labels)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        run = j - i + 1
        if run < min_run and 0 < i and j < n - 1 and labels[i - 1] == labels[j + 1]:
            labels[i:j + 1] = labels[i - 1]
        i = j + 1
    return labels


def _recheck_long_segments(data, segments, word_speakers, centroids, cluster_names,
                           min_sec: float = 3.0, margin: float = 0.08) -> int:
    """3秒以上の一続きの発言は、発言全体の声紋でもう一度話者を確かめる。
    1.5秒窓より長い音声の声紋のほうが信頼できるため、明確に別の話者に近ければ付け替える。"""
    import numpy as np
    from collections import Counter

    targets, chunks = [], []
    for si, seg in enumerate(segments):
        labels = word_speakers.get(si) or []
        if not labels or seg["end"] - seg["start"] < min_sec:
            continue
        name, count = Counter(labels).most_common(1)[0]
        if count / len(labels) < 0.8:  # 発言の途中で本当に話者が替わっている可能性 → 触らない
            continue
        s, e = seg["start"], min(seg["end"], seg["start"] + 30.0)
        chunk = data[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
        if len(chunk) >= int(min_sec * SAMPLE_RATE * 0.8):
            targets.append((si, name))
            chunks.append(chunk)
    if not chunks:
        return 0
    vecs = _embed_chunks(chunks, batch_size=16)
    cids = list(centroids.keys())
    mat = np.stack([centroids[c] for c in cids], axis=1)
    name_to_cid = {cluster_names[c]: c for c in cids}
    changed = 0
    for (si, name), vec in zip(targets, vecs):
        sims = vec @ mat
        best = int(np.argmax(sims))
        cur = cids.index(name_to_cid[name]) if name in name_to_cid else None
        if cur is not None and best != cur and sims[best] - sims[cur] > margin:
            word_speakers[si] = [cluster_names[cids[best]]] * len(word_speakers[si])
            changed += 1
    logging.info(f"長い発言の話者を再確認: {len(targets)}件中 {changed}件を修正")
    return changed


def diarize(wav_path: str, segments: list[dict], num_speakers: int = 0,
            max_speakers: int = 12, match_similarity: float = 0.55,
            progress_cb=None):
    """話者分離＋声紋DB照合を実行する。

    Returns:
        word_speakers: {seg_idx: [話者名 per word]}（単語が無いセグメントは長さ1のリスト）
        cluster_embeddings: {話者名: 重心ベクトル} … UIでの割り当て→声紋保存用
        matched: {話者名: 類似度} … 声紋DBと照合できた人物
    失敗・ライブラリ未導入時は ({}, {}, {}) を返す。
    """
    if not available():
        return {}, {}, {}
    import numpy as np

    try:
        data = _load_wav(wav_path)
        audio_end = len(data) / SAMPLE_RATE
        regions = [(s, min(e, audio_end)) for s, e in _speech_regions(segments) if s < audio_end]
        windows = _make_windows(regions)
        if not windows:
            return {}, {}, {}

        chunks = [data[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)] for s, e in windows]
        keep = [i for i, c in enumerate(chunks) if len(c) >= int(0.3 * SAMPLE_RATE)]
        windows = [windows[i] for i in keep]
        chunks = [chunks[i] for i in keep]
        if not chunks:
            return {}, {}, {}
        if progress_cb:
            progress_cb(0.1)
        vecs = _embed_chunks(chunks)
        if progress_cb:
            progress_cb(0.7)

        cv, groups = _chunk_windows(vecs, windows)
        chunk_labels = _spectral_cluster(cv, num_speakers, max_speakers)
        labels = np.zeros(len(windows), dtype=int)
        for g, lab in zip(groups, chunk_labels):
            labels[g] = lab
        labels = _merge_tiny(labels, vecs, windows, fixed_count=bool(num_speakers))
        labels = _smooth(labels)
        # 平滑化で消えたクラスタが「幻の話者」として残らないよう、重心は平滑化後に計算する
        centroids = {}
        for c in np.unique(labels):
            m = vecs[labels == c].mean(axis=0)
            centroids[int(c)] = m / (np.linalg.norm(m) or 1.0)

        # 声紋DB照合（類似度の高い組から貪欲に確定 → 同じ人物が2クラスタに付かない）
        known = load_voiceprints()
        pairs = []
        for cid, cent in centroids.items():
            for name, vp in known.items():
                pairs.append((float(cent @ vp), cid, name))
        pairs.sort(reverse=True)
        cluster_names, used_names, matched = {}, set(), {}
        for sim, cid, name in pairs:
            if sim < match_similarity:
                break
            if cid in cluster_names or name in used_names:
                continue
            cluster_names[cid] = name
            used_names.add(name)
            matched[name] = round(sim, 3)

        # 未知話者は登場順に 話者A, 話者B...
        first_seen = {}
        for i, c in enumerate(labels):
            first_seen.setdefault(int(c), i)
        letter = 0
        for cid in sorted(centroids, key=lambda c: first_seen.get(c, 1 << 30)):
            if cid not in cluster_names:
                cluster_names[cid] = (f"話者{chr(ord('A') + letter)}" if letter < 26
                                      else f"話者{letter + 1}")
                letter += 1

        # 単語 → 最も近い窓の話者
        centers = np.array([(s + e) / 2 for s, e in windows])
        order = np.argsort(centers)
        centers_sorted = centers[order]

        def label_at(t: float) -> str:
            k = int(np.searchsorted(centers_sorted, t))
            cand = [x for x in (k - 1, k) if 0 <= x < len(centers_sorted)]
            best = min(cand, key=lambda x: abs(centers_sorted[x] - t))
            return cluster_names[int(labels[order[best]])]

        word_speakers = {}
        for si, seg in enumerate(segments):
            words = seg.get("words") or []
            if words:
                word_speakers[si] = [label_at((w["start"] + w["end"]) / 2) for w in words]
            else:
                word_speakers[si] = [label_at((seg["start"] + seg["end"]) / 2)]

        _recheck_long_segments(data, segments, word_speakers, centroids, cluster_names)

        cluster_embeddings = {cluster_names[cid]: centroids[cid] for cid in centroids}
        logging.info(f"話者分離完了: {len(centroids)}名 / 窓{len(windows)}個 / 照合={matched}")
        return word_speakers, cluster_embeddings, matched
    except Exception as e:
        logging.exception(f"話者分離に失敗したためスキップします: {e}")
        return {}, {}, {}


# =====================================================================
# 声紋DB
# =====================================================================

INVALID_NAME_CHARS = r'\/:*?"<>|'


def validate_name(name: str) -> str | None:
    """声紋名として使えない場合はエラーメッセージを返す（ファイル名と一致させるため）"""
    if not name.strip():
        return "名前が空です"
    bad = [c for c in INVALID_NAME_CHARS if c in name]
    if bad:
        return f"名前に使えない文字が含まれています: {' '.join(bad)}"
    return None


def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "unknown"


def _profile_path(name: str) -> Path:
    return VOICEPRINTS_DIR / f"{_safe_filename(name)}.npy"


def _load_profile_matrix(path: Path):
    import numpy as np
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype("float32")


def load_voiceprints() -> dict:
    """profiles/voiceprints/*.npy → {名前: 正規化済み代表ベクトル}"""
    import numpy as np
    prints = {}
    if VOICEPRINTS_DIR.exists():
        for f in VOICEPRINTS_DIR.glob("*.npy"):
            try:
                mat = _load_profile_matrix(f)
                mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)
                vec = mat.mean(axis=0)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    prints[f.stem] = vec / norm
            except Exception:
                continue
    return prints


def list_voiceprints() -> list[tuple[str, int]]:
    """[(名前, 保存サンプル数)]"""
    rows = []
    if VOICEPRINTS_DIR.exists():
        for f in sorted(VOICEPRINTS_DIR.glob("*.npy")):
            try:
                rows.append((f.stem, int(_load_profile_matrix(f).shape[0])))
            except Exception:
                continue
    return rows


def save_voiceprint(name: str, embeddings) -> int:
    """声紋を追加保存（既存サンプルに追記）。保存後のサンプル数を返す。"""
    import numpy as np
    VOICEPRINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(name)
    new = np.asarray(embeddings, dtype="float32")
    if new.ndim == 1:
        new = new[None, :]
    if path.exists():
        try:
            new = np.concatenate([_load_profile_matrix(path), new], axis=0)
        except Exception:
            pass
    new = new[-MAX_PROFILE_SAMPLES:]
    np.save(path, new)
    return int(new.shape[0])


def delete_voiceprint(name: str) -> bool:
    path = _profile_path(name)
    if path.exists():
        path.unlink()
        return True
    return False


def enroll_from_audio(wav_path: str, name: str) -> tuple[int, float]:
    """登録用音声（本人だけが話しているもの）から声紋を登録する。

    Returns: (追加したサンプル数, 有効音声秒数)
    """
    import numpy as np

    data = _load_wav(wav_path)
    win = int(3.0 * SAMPLE_RATE)
    hop = int(1.5 * SAMPLE_RATE)
    chunks = []
    if len(data) < int(3.0 * SAMPLE_RATE):
        raise ValueError("音声が短すぎます（3秒以上、できれば10〜30秒話している音声を使ってください）")

    # 無音窓を除外（全体RMSの30%未満）
    frames = [data[i:i + win] for i in range(0, max(len(data) - win, 0) + 1, hop)] or [data]
    rms = np.array([np.sqrt(np.mean(c ** 2)) if len(c) else 0.0 for c in frames])
    ref = np.percentile(rms, 90) if len(rms) else 0.0
    for c, r in zip(frames, rms):
        if ref > 0 and r >= ref * 0.3:
            chunks.append(c)
    if not chunks:
        raise ValueError("声が検出できませんでした。録音レベルを確認してください。")

    vecs = _embed_chunks(chunks)
    # 外れ値（咳・他人の声など）を除外：代表ベクトルとの類似度下位を捨てる
    if len(vecs) >= 4:
        mean = vecs.mean(axis=0)
        mean /= np.linalg.norm(mean) or 1.0
        sims = vecs @ mean
        vecs = vecs[sims >= np.percentile(sims, 20)]
    save_voiceprint(name, vecs)
    voiced_sec = min(len(data) / SAMPLE_RATE, len(chunks) * 1.5 + 1.5)
    return len(vecs), voiced_sec


def identify(wav_path: str) -> list[tuple[str, float]]:
    """音声1本が誰に近いかを類似度順に返す（登録確認・テスト用）"""
    import numpy as np
    data = _load_wav(wav_path)
    win = int(3.0 * SAMPLE_RATE)
    chunks = [data[i:i + win] for i in range(0, max(len(data) - win, 0) + 1, win // 2)] or [data]
    vecs = _embed_chunks(chunks)
    mean = vecs.mean(axis=0)
    mean /= np.linalg.norm(mean) or 1.0
    scores = [(name, float(mean @ vp)) for name, vp in load_voiceprints().items()]
    return sorted(scores, key=lambda x: -x[1])
