"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in np.asarray(matrix)]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(column).tolist() for column in np.asarray(matrix).T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(mean, std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def plot_matrix_histograms(
    statistics: MatrixStatistics, path: str | None = None, limit: int = 4,
) -> None:
    """Строит гистограммы значений для первых limit строк и limit столбцов матрицы.

    Если передан path, график сохраняется в файл, иначе показывается в окне.
    """
    import matplotlib.pyplot as plt

    matrix = statistics.matrix
    count = min(limit, max(matrix.shape))
    bins = np.linspace(matrix.min(), matrix.max(), 25)
    figure, axes = plt.subplots(2, count, figsize=(3.5 * count, 6), squeeze=False, sharex=True)
    groups = [
        ("Строка", matrix, statistics.row_means, statistics.row_variances, "tab:blue"),
        ("Столбец", matrix.T, statistics.column_means, statistics.column_variances, "tab:orange"),
    ]
    for axes_row, (name, vectors, means, variances, color) in zip(axes, groups):
        for index, axis in enumerate(axes_row):
            if index >= len(vectors):
                axis.set_visible(False)
                continue
            axis.hist(vectors[index], bins=bins, color=color, edgecolor="white")
            axis.set_title(f"{name} {index}: μ={means[index]:.2f}, σ²={variances[index]:.2f}", fontsize=10)
    figure.suptitle(f"Гистограммы значений матрицы {matrix.shape[0]}×{matrix.shape[1]}")
    figure.tight_layout()
    if path is None:
        plt.show()
    else:
        figure.savefig(path, dpi=120)
        plt.close(figure)


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    row_indices, column_indices = np.indices((rows, columns))
    return np.where((row_indices + column_indices) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color
    top = max((image_height - height) // 2, 0)
    left = max((image_width - width) // 2, 0)
    image[top:top + height, left:left + width] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color
    y, x = np.indices((image_height, image_width))
    x0, y0 = (image_width - 1) / 2, (image_height - 1) / 2
    mask = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)
    if not 1 <= window <= len(values):
        raise ValueError("Window must be between 1 and the series length")
    middle, previous, following = values[1:-1], values[:-2], values[2:]
    maxima = np.nonzero((middle > previous) & (middle > following))[0] + 1
    minima = np.nonzero((middle < previous) & (middle < following))[0] + 1
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=float(values.mean()),
        variance=float(values.var()),
        std=float(values.std()),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    if labels.size and (labels.min() < 0 or labels.max() >= class_count):
        raise ValueError("Labels must be in range [0, class_count)")
    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result
