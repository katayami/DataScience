import unittest

import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixVectorBatchInput,
    OneHotInput, RandomMatrixInput, RectangleInput, TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series, binarize, chess, draw_ellipse, draw_rectangle,
    matrix_statistics, one_hot, sum_prod, unique_columns, unique_rows,
)


class SumProdTest(unittest.TestCase):
    def test_against_loop(self):
        rng = np.random.default_rng(0)
        matrices = rng.normal(size=(4, 3, 3))
        vectors = rng.normal(size=(4, 3, 1))
        expected = sum(a @ x for a, x in zip(matrices, vectors))
        result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
        self.assertEqual(result.shape, (3, 1))
        np.testing.assert_allclose(result, expected)

    def test_single_pair_and_lists(self):
        result = sum_prod(MatrixVectorBatchInput([[[1, 2], [3, 4]]], [[[1], [1]]]))
        np.testing.assert_array_equal(result, [[3], [7]])

    def test_identity(self):
        matrices = [np.eye(2), 2 * np.eye(2)]
        vectors = [np.array([[1], [2]]), np.array([[3], [4]])]
        np.testing.assert_array_equal(sum_prod(MatrixVectorBatchInput(matrices, vectors)), [[7], [10]])


class BinarizeTest(unittest.TestCase):
    def test_basic(self):
        matrix = np.array([[0.1, 0.5], [0.6, 1.0]])
        np.testing.assert_array_equal(binarize(BinarizeInput(matrix)), [[0, 0], [1, 1]])

    def test_threshold_is_strict_and_source_unchanged(self):
        matrix = np.array([[1, 2, 3]])
        result = binarize(BinarizeInput(matrix, threshold=2))
        np.testing.assert_array_equal(result, [[0, 0, 1]])
        np.testing.assert_array_equal(matrix, [[1, 2, 3]])

    def test_negative_threshold(self):
        np.testing.assert_array_equal(binarize(BinarizeInput([[-2, -1, 0]], -1.5)), [[0, 1, 1]])


class UniqueTest(unittest.TestCase):
    matrix = np.array([[1, 2, 2], [3, 3, 3], [1, 4, 2]])

    def test_rows(self):
        self.assertEqual(unique_rows(MatrixInput(self.matrix)), [[1, 2], [3], [1, 2, 4]])

    def test_columns(self):
        self.assertEqual(unique_columns(MatrixInput(self.matrix)), [[1, 3], [2, 3, 4], [2, 3]])

    def test_single_row(self):
        self.assertEqual(unique_rows(MatrixInput([[5, 5]])), [[5]])
        self.assertEqual(unique_columns(MatrixInput([[5, 5]])), [[5], [5]])


class MatrixStatisticsTest(unittest.TestCase):
    def test_shapes_and_values(self):
        stats = matrix_statistics(RandomMatrixInput(rows=4, columns=6, mean=3, std=2, seed=1))
        self.assertEqual(stats.matrix.shape, (4, 6))
        np.testing.assert_allclose(stats.row_means, stats.matrix.mean(axis=1))
        np.testing.assert_allclose(stats.column_means, stats.matrix.mean(axis=0))
        np.testing.assert_allclose(stats.row_variances, stats.matrix.var(axis=1))
        np.testing.assert_allclose(stats.column_variances, stats.matrix.var(axis=0))

    def test_seed_reproducibility(self):
        first = matrix_statistics(RandomMatrixInput(3, 3, seed=7))
        second = matrix_statistics(RandomMatrixInput(3, 3, seed=7))
        np.testing.assert_array_equal(first.matrix, second.matrix)

    def test_distribution_parameters(self):
        stats = matrix_statistics(RandomMatrixInput(500, 500, mean=10, std=3))
        self.assertAlmostEqual(stats.matrix.mean(), 10, delta=0.05)
        self.assertAlmostEqual(stats.matrix.std(), 3, delta=0.05)


class ChessTest(unittest.TestCase):
    def test_example(self):
        expected = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
        np.testing.assert_array_equal(chess(ChessInput(3, 4, 0, 1)), expected)

    def test_single_cell_and_line(self):
        np.testing.assert_array_equal(chess(ChessInput(1, 1, 5, 7)), [[5]])
        np.testing.assert_array_equal(chess(ChessInput(1, 3, 5, 7)), [[5, 7, 5]])


class DrawTest(unittest.TestCase):
    red, black = (255, 0, 0), (0, 0, 0)

    def test_rectangle(self):
        image = draw_rectangle(RectangleInput(2, 2, 4, 6, self.red, self.black))
        self.assertEqual(image.shape, (4, 6, 3))
        mask = np.all(image == self.red, axis=2)
        expected = np.zeros((4, 6), dtype=bool)
        expected[1:3, 2:4] = True
        np.testing.assert_array_equal(mask, expected)

    def test_rectangle_fills_image(self):
        image = draw_rectangle(RectangleInput(3, 3, 3, 3, self.red, self.black))
        self.assertTrue(np.all(image == self.red))

    def test_ellipse(self):
        image = draw_ellipse(EllipseInput(2, 1, 5, 7, self.red, self.black))
        self.assertEqual(image.shape, (5, 7, 3))
        mask = np.all(image == self.red, axis=2)
        self.assertTrue(mask[2, 3])
        self.assertTrue(mask[2, 1] and mask[2, 5])
        self.assertTrue(mask[1, 3] and mask[3, 3])
        self.assertFalse(mask[2, 0] or mask[0, 3])
        self.assertFalse(mask[1, 1])
        np.testing.assert_array_equal(mask, mask[::-1, ::-1])


class TimeSeriesTest(unittest.TestCase):
    def test_basic(self):
        stats = analyze_time_series(TimeSeriesInput([1, 3, 2, 5, 4, 4, 1], window=3))
        values = np.array([1, 3, 2, 5, 4, 4, 1])
        self.assertAlmostEqual(stats.mean, values.mean())
        self.assertAlmostEqual(stats.variance, values.var())
        self.assertAlmostEqual(stats.std, values.std())
        np.testing.assert_array_equal(stats.local_maxima_indices, [1, 3])
        np.testing.assert_array_equal(stats.local_minima_indices, [2])
        np.testing.assert_allclose(stats.moving_average, [2, 10 / 3, 11 / 3, 13 / 3, 3])

    def test_plateau_and_endpoints(self):
        stats = analyze_time_series(TimeSeriesInput([5, 1, 1, 5], window=4))
        self.assertEqual(len(stats.local_maxima_indices), 0)
        self.assertEqual(len(stats.local_minima_indices), 0)
        np.testing.assert_allclose(stats.moving_average, [3])

    def test_window_one(self):
        stats = analyze_time_series(TimeSeriesInput([2, 4], window=1))
        np.testing.assert_allclose(stats.moving_average, [2, 4])

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            analyze_time_series(TimeSeriesInput([1, 2], window=3))


class OneHotTest(unittest.TestCase):
    def test_example(self):
        expected = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
        np.testing.assert_array_equal(one_hot(OneHotInput([0, 2, 3, 0])), expected)

    def test_explicit_class_count(self):
        np.testing.assert_array_equal(one_hot(OneHotInput([1], class_count=3)), [[0, 1, 0]])

    def test_all_zero_labels(self):
        np.testing.assert_array_equal(one_hot(OneHotInput([0, 0])), [[1], [1]])

    def test_label_out_of_range(self):
        with self.assertRaises(ValueError):
            one_hot(OneHotInput([3], class_count=2))


if __name__ == "__main__":
    unittest.main()
