import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CALIBRATION_METHODS = (
    "far",
    "eer",
    "youden",
    "cost",
    "percentile",
)


def calculate_rates(
    threshold: float,
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray | None,
) -> dict[str, float | None]:
    """Calculate FAR, FRR and TPR for a distance threshold."""

    frr = float(np.mean(genuine_distances > threshold))
    far = None

    if impostor_distances is not None:
        far = float(np.mean(impostor_distances <= threshold))

    return {
        "far": far,
        "frr": frr,
        "tpr": 1.0 - frr,
    }


def calibrate_threshold(
    genuine_distances: list[float] | np.ndarray,
    impostor_distances: list[float] | np.ndarray | None = None,
    method: str = "far",
    target_far: float = 0.01,
    percentile: float = 95.0,
    false_accept_cost: float = 10.0,
    false_reject_cost: float = 1.0,
) -> dict[str, Any]:
    """Calculate a threshold using the selected calibration method."""

    genuine = validate_distances(
        genuine_distances,
        "genuine_distances",
    )
    impostor = None

    if impostor_distances is not None:
        impostor = validate_distances(
            impostor_distances,
            "impostor_distances",
        )

    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Use one of: {', '.join(CALIBRATION_METHODS)}."
        )

    if method == "percentile":
        threshold = calibrate_by_percentile(genuine, percentile)
    else:
        if impostor is None:
            raise ValueError(
                f"The '{method}' method requires impostor distances."
            )

        if method == "far":
            threshold = calibrate_by_target_far(
                genuine,
                impostor,
                target_far,
            )
        elif method == "eer":
            threshold = calibrate_by_eer(genuine, impostor)
        elif method == "youden":
            threshold = calibrate_by_youden(genuine, impostor)
        else:
            threshold = calibrate_by_cost(
                genuine,
                impostor,
                false_accept_cost,
                false_reject_cost,
            )

    rates = calculate_rates(threshold, genuine, impostor)

    return {
        "method": method,
        "threshold": float(threshold),
        "far": rates["far"],
        "frr": rates["frr"],
        "tpr": rates["tpr"],
        "genuine_pairs": int(genuine.size),
        "impostor_pairs": int(impostor.size) if impostor is not None else 0,
    }


def calibrate_by_target_far(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    target_far: float,
) -> float:
    """Choose the largest threshold that does not exceed target FAR."""

    if not 0.0 <= target_far <= 1.0:
        raise ValueError("target_far must be between 0 and 1.")

    valid_results = []

    for threshold in create_threshold_candidates(
        genuine_distances,
        impostor_distances,
    ):
        rates = calculate_rates(
            threshold,
            genuine_distances,
            impostor_distances,
        )
        if rates["far"] <= target_far:
            valid_results.append(
                (
                    rates["tpr"],
                    threshold,
                )
            )

    if not valid_results:
        return float(np.nextafter(impostor_distances.min(), -np.inf))

    _, threshold = max(valid_results)
    return float(threshold)


def calibrate_by_eer(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
) -> float:
    """Choose the threshold where FAR and FRR are closest."""

    results = []

    for threshold in create_threshold_candidates(
        genuine_distances,
        impostor_distances,
    ):
        rates = calculate_rates(
            threshold,
            genuine_distances,
            impostor_distances,
        )
        difference = abs(rates["far"] - rates["frr"])
        average_error = (rates["far"] + rates["frr"]) / 2.0
        results.append(
            (
                difference,
                average_error,
                threshold,
            )
        )

    return float(min(results)[2])


def calibrate_by_youden(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
) -> float:
    """Choose the threshold that maximizes TPR minus FAR."""

    results = []

    for threshold in create_threshold_candidates(
        genuine_distances,
        impostor_distances,
    ):
        rates = calculate_rates(
            threshold,
            genuine_distances,
            impostor_distances,
        )
        youden_index = rates["tpr"] - rates["far"]
        results.append(
            (
                youden_index,
                -rates["far"],
                threshold,
            )
        )

    return float(max(results)[2])


def calibrate_by_cost(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    false_accept_cost: float,
    false_reject_cost: float,
) -> float:
    """Choose the threshold that minimizes the configured error cost."""

    if false_accept_cost < 0 or false_reject_cost < 0:
        raise ValueError("Error costs cannot be negative.")
    if false_accept_cost == 0 and false_reject_cost == 0:
        raise ValueError("At least one error cost must be greater than zero.")

    results = []

    for threshold in create_threshold_candidates(
        genuine_distances,
        impostor_distances,
    ):
        rates = calculate_rates(
            threshold,
            genuine_distances,
            impostor_distances,
        )
        total_cost = (
            false_accept_cost * rates["far"]
            + false_reject_cost * rates["frr"]
        )
        results.append(
            (
                total_cost,
                rates["far"],
                threshold,
            )
        )

    return float(min(results)[2])


def calibrate_by_percentile(
    genuine_distances: np.ndarray,
    percentile: float,
) -> float:
    """Choose a percentile of the genuine distance distribution."""

    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100.")

    return float(np.percentile(genuine_distances, percentile))


def create_threshold_candidates(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
) -> np.ndarray:
    """Create threshold values between all observed distances."""

    values = np.unique(
        np.concatenate(
            [
                genuine_distances,
                impostor_distances,
            ]
        )
    )

    if values.size == 1:
        return values

    midpoints = (values[:-1] + values[1:]) / 2.0
    lower = np.nextafter(values[0], -np.inf)
    upper = np.nextafter(values[-1], np.inf)
    return np.sort(
        np.concatenate(
            (
                [lower],
                values,
                midpoints,
                [upper],
            )
        )
    )


def validate_distances(
    distances: list[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    vector = np.asarray(distances, dtype=np.float64).reshape(-1)

    if vector.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains invalid values.")

    return vector


def load_distances(input_path: str) -> tuple[list[float], list[float] | None]:
    with Path(input_path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "genuine_distances" not in data:
        raise ValueError("Input JSON requires 'genuine_distances'.")

    return (
        data["genuine_distances"],
        data.get("impostor_distances"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a facial-recognition distance threshold."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="JSON file containing genuine and impostor distances.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=CALIBRATION_METHODS,
        default="far",
        help="Threshold calibration method.",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=0.01,
        help="Maximum FAR accepted by the far method.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Genuine percentile used by the percentile method.",
    )
    parser.add_argument(
        "--false-accept-cost",
        type=float,
        default=10.0,
        help="Cost assigned to accepting an impostor.",
    )
    parser.add_argument(
        "--false-reject-cost",
        type=float,
        default=1.0,
        help="Cost assigned to rejecting a genuine user.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional path where the calibration result is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    genuine_distances, impostor_distances = load_distances(
        args.input_path
    )
    result = calibrate_threshold(
        genuine_distances=genuine_distances,
        impostor_distances=impostor_distances,
        method=args.calibration_method,
        target_far=args.target_far,
        percentile=args.percentile,
        false_accept_cost=args.false_accept_cost,
        false_reject_cost=args.false_reject_cost,
    )

    output = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )
    print(output)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            output + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
