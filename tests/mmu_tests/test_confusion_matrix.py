import pytest
import sklearn as sk
import sklearn.metrics as skm
import numpy as np
from mmu import _core

from mmu_tests import generate_test_labels


def test_confusion_matrix_default():
    """Test confusion_matrix int64"""
    _, yhat, y = generate_test_labels(1000)
    conf_mat = _core.confusion_matrix(y, yhat)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat), conf_mat


def test_confusion_matrix_bool():
    """Test confusion_matrix bool"""
    _, yhat, y = generate_test_labels(1000, np.bool_)
    conf_mat = _core.confusion_matrix(y, yhat)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat)


def test_confusion_matrix_int():
    """Test confusion_matrix int32"""
    _, yhat, y = generate_test_labels(1000, np.int32)
    conf_mat = _core.confusion_matrix(y, yhat)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat)


def test_confusion_matrix_float():
    """Test confusion_matrix float"""
    _, yhat, y = generate_test_labels(1000, np.float32)
    conf_mat = _core.confusion_matrix(y, yhat)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat)


def test_confusion_matrix_double():
    """Test confusion_matrix double"""
    _, yhat, y = generate_test_labels(1000, np.float64)
    conf_mat = _core.confusion_matrix(y, yhat)
    sk_conf_mat = skm.confusion_matrix(y, yhat)
    assert np.array_equal(conf_mat, sk_conf_mat)


def test_confusion_matrix_exceptions():
    """"""
    _, yhat, y = generate_test_labels(1000)
    with pytest.raises(RuntimeError):
        _core.confusion_matrix(y, yhat[:100])
    with pytest.raises(RuntimeError):
        _core.confusion_matrix(y[:100], yhat)
    with pytest.raises(RuntimeError):
        _core.confusion_matrix(np.tile(y, 2), yhat)
    with pytest.raises(RuntimeError):
        _core.confusion_matrix(y, np.tile(yhat, 2))
