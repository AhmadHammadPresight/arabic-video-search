from __future__ import annotations

from PIL import Image

from avs.inference.crops import Candidate, TrackCandidates, crop_quality, padded_box, select_crops, touches_edge

IMG = Image.new("RGB", (4, 4))


def cand(t: float, quality: float) -> Candidate:
    return Candidate(t=t, box=(0, 0, 1, 1), quality=quality, image=IMG)


def test_quality_prefers_confident_large_uncut_boxes():
    big = crop_quality((100, 100, 200, 400), 0.9, 1920, 1080)
    small = crop_quality((100, 100, 130, 180), 0.9, 1920, 1080)
    cut = crop_quality((0, 100, 100, 400), 0.9, 1920, 1080)
    unsure = crop_quality((100, 100, 200, 400), 0.4, 1920, 1080)
    assert big > small and big > cut and big > unsure
    assert cut == big / 2


def test_touches_edge():
    assert touches_edge((0, 10, 50, 50), 100, 100)
    assert touches_edge((10, 10, 100, 50), 100, 100)
    assert not touches_edge((10, 10, 90, 90), 100, 100)


def test_padded_box_stays_inside_the_frame():
    assert padded_box((5, 5, 95, 95), 100, 100, pad=0.1) == (0, 0, 100, 100)
    assert padded_box((40, 40, 60, 60), 100, 100, pad=0.1) == (38, 38, 62, 62)


def test_pool_keeps_only_the_best_candidates():
    pool = TrackCandidates(capacity=3)
    for i, q in enumerate([0.1, 0.9, 0.5, 0.3, 0.8]):
        pool.offer(cand(float(i), q))
    assert sorted(c.quality for c in pool.candidates()) == [0.5, 0.8, 0.9]
    assert not pool.would_accept(0.4) and pool.would_accept(0.6)


def test_select_crops_spreads_picks_over_time():
    cands = [cand(0.0, 0.9), cand(0.1, 0.89), cand(0.2, 0.88), cand(5.0, 0.5), cand(9.0, 0.4)]
    chosen = select_crops(cands, k=3, min_gap=1.0)
    assert [c.t for c in chosen] == [0.0, 5.0, 9.0]


def test_select_crops_fills_up_when_spacing_leaves_too_few():
    cands = [cand(0.0, 0.9), cand(0.1, 0.8), cand(0.2, 0.7)]
    chosen = select_crops(cands, k=2, min_gap=5.0)
    assert [c.quality for c in sorted(chosen, key=lambda c: -c.quality)] == [0.9, 0.8]
