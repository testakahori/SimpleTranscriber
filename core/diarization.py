"""話者分離と声紋識別（オプショナル機能）

speechbrain (ECAPA-TDNN) + torch が導入されている場合のみ動作する。
未導入の場合 available() が False を返し、アプリは話者分離なしで動く。

導入方法（任意）:
    pip install torch torchaudio speechbrain
初回実行時に話者埋め込みモデル（約80MB）を自動ダウンロードする。
"""
import importlib.util
import logging
from pathlib import Path

from .config import VOICEPRINTS_DIR

_classifier = None

MIN_SEGMENT_SEC = 0.6  # これより短いセグメントは埋め込み対象外


def available() -> bool:
    return (importlib.util.find_spec("speechbrain") is not None
            and importlib.util.find_spec("torch") is not None)


def _get_classifier():
    global _classifier
    if _classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        logging.info("話者埋め込みモデル (ECAPA-TDNN) をロード中...")
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home() / ".cache" / "speechbrain_ecapa"),
        )
    return _classifier


def _embed_segments(wav_path: str, segments: list[dict]):
    """各セグメントの話者埋め込みベクトルを返す。{idx: np.ndarray}"""
    import numpy as np
    import soundfile as sf
    import torch

    classifier = _get_classifier()
    data, rate = sf.read(wav_path, dtype="float32")
    embeddings = {}
    for i, seg in enumerate(segments):
        start = int(seg["start"] * rate)
        end = int(seg["end"] * rate)
        if (end - start) / rate < MIN_SEGMENT_SEC:
            continue
        chunk = data[start:end]
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        try:
            tensor = torch.from_numpy(chunk).unsqueeze(0)
            emb = classifier.encode_batch(tensor).squeeze().detach().cpu().numpy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                embeddings[i] = emb / norm
        except Exception as e:
            logging.warning(f"セグメント{i}の埋め込みに失敗: {e}")
    return embeddings


def _cluster(embeddings: dict, threshold: float):
    """単純な逐次クラスタリング（コサイン類似度）。{idx: cluster_id}"""
    import numpy as np

    centroids = []  # [(sum_vec, count)]
    assignment = {}
    for idx in sorted(embeddings.keys()):
        vec = embeddings[idx]
        best_cid, best_sim = -1, -1.0
        for cid, (total, count) in enumerate(centroids):
            centroid = total / count
            centroid = centroid / (np.linalg.norm(centroid) or 1.0)
            sim = float(np.dot(vec, centroid))
            if sim > best_sim:
                best_cid, best_sim = cid, sim
        if best_cid >= 0 and best_sim >= threshold:
            total, count = centroids[best_cid]
            centroids[best_cid] = (total + vec, count + 1)
            assignment[idx] = best_cid
        else:
            centroids.append((vec.copy(), 1))
            assignment[idx] = len(centroids) - 1
    cluster_centroids = {}
    for cid, (total, count) in enumerate(centroids):
        c = total / count
        cluster_centroids[cid] = c / (np.linalg.norm(c) or 1.0)
    return assignment, cluster_centroids


def load_voiceprints() -> dict:
    """profiles/voiceprints/*.npy → {名前: 正規化ベクトル}"""
    import numpy as np
    prints = {}
    if VOICEPRINTS_DIR.exists():
        for f in VOICEPRINTS_DIR.glob("*.npy"):
            try:
                vec = np.load(f)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    prints[f.stem] = vec / norm
            except Exception:
                continue
    return prints


def save_voiceprint(name: str, embedding) -> None:
    """声紋を保存。既存があれば平均を取って更新（使うほど精度向上）。"""
    import numpy as np
    VOICEPRINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = VOICEPRINTS_DIR / f"{name}.npy"
    vec = np.asarray(embedding, dtype="float32")
    if path.exists():
        try:
            old = np.load(path)
            vec = (old + vec) / 2.0
        except Exception:
            pass
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    np.save(path, vec)


def diarize(wav_path: str, segments: list[dict], match_threshold: float = 0.75,
            cluster_threshold: float = 0.68):
    """話者分離＋声紋DB照合を実行する。

    Returns:
        speakers: {segment_index: 話者名}  … 「坪内」or「話者A」
        cluster_embeddings: {話者ラベル: 埋め込みベクトル} … UI割り当て→声紋保存用
    失敗・ライブラリ未導入時は ({}, {}) を返す。
    """
    if not available():
        return {}, {}
    import numpy as np

    try:
        embeddings = _embed_segments(wav_path, segments)
        if not embeddings:
            return {}, {}
        assignment, centroids = _cluster(embeddings, cluster_threshold)

        known = load_voiceprints()
        cluster_names = {}
        used_letters = 0
        for cid, centroid in centroids.items():
            best_name, best_sim = None, -1.0
            for name, vp in known.items():
                sim = float(np.dot(centroid, vp))
                if sim > best_sim:
                    best_name, best_sim = name, sim
            if best_name and best_sim >= match_threshold:
                cluster_names[cid] = best_name
            else:
                cluster_names[cid] = f"話者{chr(ord('A') + used_letters)}"
                used_letters += 1

        speakers = {idx: cluster_names[cid] for idx, cid in assignment.items()}
        cluster_embeddings = {cluster_names[cid]: centroids[cid] for cid in centroids}
        return speakers, cluster_embeddings
    except Exception as e:
        logging.warning(f"話者分離に失敗したためスキップします: {e}")
        return {}, {}
