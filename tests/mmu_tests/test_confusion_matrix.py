import pytest
import sklearn as sk
import sklearn.metrics as skm
import numpy as np
import mmu

from mmu_tests import generate_test_labels


def test_confusion_matrix_default():
    """Test confusion_matrix int64"""
    _, yhat, y = generate_test_labels(1000)
    conf_mat = mmu.confusion_matrix(y, yhat)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat), conf_mat


def test_confusion_matrix_bool():
    """Test confusion_matrix bool"""
    _, yhat, y = generate_test_labels(1000, np.bool_)
    conf_mat = mmu.confusion_matrix(y, yhat)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat)


def test_confusion_matrix_int():
    """Test confusion_matrix int32"""
    _, yhat, y = generate_test_labels(1000, np.int32)
    conf_mat = mmu.confusion_matrix(y, yhat)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat)


def test_confusion_matrix_float():
    """Test confusion_matrix float"""
    _, yhat, y = generate_test_labels(1000, np.float32)
    conf_mat = mmu.confusion_matrix(y, yhat)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat)


def test_confusion_matrix_double():
    """Test confusion_matrix double"""
    _, yhat, y = generate_test_labels(1000, np.float64)
    conf_mat = mmu.confusion_matrix(y, yhat)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat)


def test_confusion_matrix_exceptions():
    """"""
    _, yhat, y = generate_test_labels(1000)
    with pytest.raises(RuntimeError):
        mmu.confusion_matrix(y, yhat[:100])
    with pytest.raises(RuntimeError):
        mmu.confusion_matrix(y[:100], yhat)
    with pytest.raises(RuntimeError):
        mmu.confusion_matrix(np.tile(y, 2), yhat)
    with pytest.raises(RuntimeError):
        mmu.confusion_matrix(y, np.tile(yhat, 2))


def test_confusion_matrix_proba_default():
    """Test confusion_matrix int64"""
    proba, yhat, y = generate_test_labels(1000)
    conf_mat = mmu.confusion_matrix_proba(y, proba, 0.5)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat), conf_mat


def test_confusion_matrix_proba_bool():
    """Test confusion_matrix bool"""
    proba, yhat, y = generate_test_labels(1000, np.bool_)
    conf_mat = mmu.confusion_matrix_proba(y, proba, 0.5)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat), conf_mat


def test_confusion_matrix_proba_int():
    """Test confusion_matrix int"""
    proba, yhat, y = generate_test_labels(1000, np.int32)
    conf_mat = mmu.confusion_matrix_proba(y, proba, 0.5)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat), conf_mat


def test_confusion_matrix_proba_float():
    """Test confusion_matrix float"""
    proba, yhat, y = generate_test_labels(1000, np.float32)
    conf_mat = mmu.confusion_matrix_proba(y, proba, 0.5)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat), conf_mat


def test_confusion_matrix_proba_double():
    """Test confusion_matrix double"""
    proba, yhat, y = generate_test_labels(1000, np.float64)
    conf_mat = mmu.confusion_matrix_proba(y, proba, 0.5)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat), conf_mat


def test_confusion_matrix_proba_int64_thresholds_check():
    """Test confusion_matrix double with different threshold."""
    proba, yhat, y = generate_test_labels(1000, np.int64)
    conf_mat = mmu.confusion_matrix_proba(y, proba, 0.3)
    sk_conf_mat = skm.confusion_matrix(y, (proba > 0.3)).astype(np.int64)
    assert np.array_equal(conf_mat, sk_conf_mat), conf_mat

    conf_mat = mmu.confusion_matrix_proba(y, proba, 0.9)
    sk_conf_mat = skm.confusion_matrix(y, (proba > 0.9)).astype(np.int64)
    assert np.array_equal(conf_mat, sk_conf_mat), conf_mat

def test_confusion_matrix_proba_thresholds_check():
    """Test confusion_matrix double with different threshold."""
    proba, yhat, y = generate_test_labels(1000, np.float64)
    conf_mat = mmu.confusion_matrix_proba(y, proba, 0.3)
    sk_conf_mat = skm.confusion_matrix(y, (proba > 0.3)).astype(np.int64)
    assert np.array_equal(conf_mat, sk_conf_mat), conf_mat

    conf_mat = mmu.confusion_matrix_proba(y, proba, 0.9)
    sk_conf_mat = skm.confusion_matrix(y, (proba > 0.9)).astype(np.int64)
    assert np.array_equal(conf_mat, sk_conf_mat), conf_mat
