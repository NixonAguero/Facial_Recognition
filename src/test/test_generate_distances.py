import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils.generate_distances import (
    create_distance_pairs,
    discover_labeled_images,
    extract_identity_from_filename,
    generate_distances,
)


def fake_embedding(image_path: Path) -> list[float]:
    identity = image_path.parent.name
    identity_index = {
        "Ian": 0,
        "Nixon": 1,
        "Cristel": 2,
    }[identity]
    embedding = [0.0] * 512
    embedding[identity_index] = 1.0
    return embedding


class DistanceGenerationTests(unittest.TestCase):
    def test_extracts_identity_from_flat_filename(self):
        self.assertEqual(
            extract_identity_from_filename("ian_02"),
            "ian",
        )

    def test_discovers_images_from_identity_folders(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            create_image_files(
                root,
                {
                    "Ian": 2,
                    "Nixon": 1,
                },
            )

            samples, mode = discover_labeled_images(
                root,
                label_mode="folders",
            )

        self.assertEqual(mode, "folders")
        self.assertEqual(len(samples), 3)
        self.assertEqual(
            {sample["identity"] for sample in samples},
            {"Ian", "Nixon"},
        )

    def test_creates_each_pair_once(self):
        samples = [
            {
                "identity": "Ian",
                "image": "Ian/1.png",
                "embedding": [1.0] + [0.0] * 511,
            },
            {
                "identity": "Ian",
                "image": "Ian/2.png",
                "embedding": [1.0] + [0.0] * 511,
            },
            {
                "identity": "Nixon",
                "image": "Nixon/1.png",
                "embedding": [0.0, 1.0] + [0.0] * 510,
            },
        ]

        genuine, impostor = create_distance_pairs(samples)

        self.assertEqual(len(genuine), 1)
        self.assertEqual(len(impostor), 2)

    def test_generates_expected_pair_counts_and_json(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset"
            output_path = Path(temporary_directory) / "distances.json"
            create_image_files(
                root,
                {
                    "Ian": 3,
                    "Nixon": 2,
                    "Cristel": 1,
                },
            )

            result = generate_distances(
                dataset_path=root,
                output_path=output_path,
                label_mode="folders",
                embedding_generator=fake_embedding,
            )
            saved_result = json.loads(
                output_path.read_text(encoding="utf-8")
            )

        self.assertEqual(result["processed_images"], 6)
        self.assertEqual(result["genuine_pair_count"], 4)
        self.assertEqual(result["impostor_pair_count"], 11)
        self.assertEqual(
            len(result["genuine_distances"]),
            4,
        )
        self.assertEqual(
            len(result["impostor_distances"]),
            11,
        )
        self.assertEqual(saved_result["metric"], "cosine_distance")


def create_image_files(
    root: Path,
    identity_counts: dict[str, int],
) -> None:
    for identity, count in identity_counts.items():
        identity_folder = root / identity
        identity_folder.mkdir(parents=True, exist_ok=True)

        for index in range(1, count + 1):
            (identity_folder / f"foto{index}.png").write_bytes(b"test")


if __name__ == "__main__":
    unittest.main()
