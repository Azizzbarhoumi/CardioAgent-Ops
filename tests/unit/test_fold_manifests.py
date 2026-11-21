"""Fold protocol checks on the REAL manifests (Data/processed/splits/v2_patient_grouped)."""

import itertools

import pytest

from audio_engine.evaluation.folds import (
    N_SPLITS, SPLIT_NAMES, assign_groups, load_folds, manifest_path, parse_stem, read_manifest, validate_fold,
)

pytestmark = pytest.mark.skipif(not manifest_path(1).is_file(), reason="fold manifests not built")


@pytest.fixture(scope="module")
def folds():
    return load_folds()


def test_three_folds_cover_every_recording_once_as_test(folds):
    assert len(folds) == N_SPLITS
    all_stems = set(folds[0].stems("train") + folds[0].stems("validation") + folds[0].stems("test"))
    tested = list(itertools.chain.from_iterable(f.stems("test") for f in folds))
    assert len(tested) == len(set(tested)) == len(all_stems)


@pytest.mark.parametrize("fold_index", range(N_SPLITS))
def test_no_group_or_pascal_patient_in_two_splits(folds, fold_index):
    fold = folds[fold_index]
    groups = {s: {r.group for r in fold.split(s)} for s in SPLIT_NAMES}
    patients = {
        s: {parse_stem(r.recording_stem)[1] for r in fold.split(s) if parse_stem(r.recording_stem)[0] == "pascal_patient"}
        for s in SPLIT_NAMES
    }
    for a, b in itertools.combinations(SPLIT_NAMES, 2):
        assert not groups[a] & groups[b], f"fold {fold.number}: groups shared by {a}/{b}"
        assert not patients[a] & patients[b], f"fold {fold.number}: PASCAL patients shared by {a}/{b}"


@pytest.mark.parametrize("fold_index", range(N_SPLITS))
def test_every_split_has_both_classes(folds, fold_index):
    for s in SPLIT_NAMES:
        assert set(folds[fold_index].labels(s).tolist()) == {0, 1}


def test_group_assignment_is_the_same_in_every_fold(folds):
    maps = [{r.recording_stem: r.group for r in f.rows} for f in folds]
    assert maps[0] == maps[1] == maps[2]
    assert maps[0] == assign_groups(list(maps[0]))


def test_same_pascal_patient_always_gets_one_key():
    groups = assign_groups([
        "pascal_normal_106_1306776721273_A", "pascal_murmur_106_1306776721273_C2", "pascal_extra_106_1306776721273_D 2",
    ])
    assert len(set(groups.values())) == 1


def test_validator_fails_loudly_on_leakage():
    rows = read_manifest(manifest_path(1))
    test_group = next(r.group for r in rows if r.split == "test")
    i = next(i for i, r in enumerate(rows) if r.split == "train")
    rows[i] = type(rows[i])(**{**rows[i].__dict__, "group": test_group})
    with pytest.raises(AssertionError, match="leakage"):
        validate_fold(rows)


def test_unknown_filename_is_rejected():
    with pytest.raises(ValueError):
        parse_stem("mystery_file")
