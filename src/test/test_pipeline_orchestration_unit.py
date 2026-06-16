import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from src.api import register
from src.engine import RetinaFace
from src.pipelines import hybrid_pipeline, standard_pipeline
from src.utils.normalizer import (
    calculate_embedding_centroid,
    l2_normalize_embedding,
)


FACE = {
    "bbox": {"x1": 1.0, "y1": 2.0, "x2": 10.0, "y2": 12.0},
    "confidence": 0.99,
    "sharpness": 120.0,
    "bbox_area_px": 90.0,
    "landmarks": [
        {"name": "right_eye", "x": 30.0, "y": 40.0},
        {"name": "left_eye", "x": 70.0, "y": 40.0},
        {"name": "nose", "x": 50.0, "y": 60.0},
        {"name": "mouth_right", "x": 35.0, "y": 80.0},
        {"name": "mouth_left", "x": 65.0, "y": 80.0},
    ],
    "crop": np.zeros((112, 112, 3), dtype=np.float32),
}
EMBEDDING = [1.0] + [0.0] * 511


class FakeAnomalyDetector:
    def __init__(self, score, threshold=0.05):
        self._score = score
        self.threshold = threshold

    def score(self, embedding):
        return self._score


class NormalizerTests(unittest.TestCase):
    def test_l2_normalizes_512_dimensions(self):
        normalized = l2_normalize_embedding([2.0] + [0.0] * 511)

        self.assertEqual(len(normalized), 512)
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0)

    def test_rejects_wrong_dimension(self):
        with self.assertRaises(ValueError):
            l2_normalize_embedding([1.0, 2.0])

    def test_centroid_is_l2_normalized(self):
        embeddings = [
            [1.0] + [0.0] * 511,
            [0.0, 1.0] + [0.0] * 510,
        ]

        centroid = calculate_embedding_centroid(embeddings)

        self.assertEqual(len(centroid), 512)
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0)


class RetinaFaceTests(unittest.TestCase):
    def test_returns_normalized_face_and_five_landmarks(self):
        normalized_crop = np.ones((120, 90, 3), dtype=np.float32)
        facial_area = {
            "x": 10,
            "y": 10,
            "w": 100,
            "h": 100,
            "right_eye": (35, 40),
            "left_eye": (80, 40),
            "nose": (58, 62),
            "mouth_right": (42, 85),
            "mouth_left": (75, 85),
        }
        image = np.indices((120, 120)).sum(axis=0) % 2
        image = np.repeat(image[:, :, None], 3, axis=2)
        image = (image * 255).astype(np.uint8)

        with patch.object(
            RetinaFace.DeepFace,
            "extract_faces",
            return_value=[
                {
                    "face": normalized_crop,
                    "facial_area": facial_area,
                    "confidence": 0.99,
                }
            ],
        ):
            result = RetinaFace.detect_faces(image)

        self.assertEqual(len(result), 1)
        self.assertIs(result[0]["crop"], normalized_crop)
        self.assertEqual(len(result[0]["landmarks"]), 5)


class StandardPipelineTests(unittest.TestCase):
    def test_sign_up_requires_images_with_filenames(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)

        with self.assertRaises(standard_pipeline.FacePipelineError):
            standard_pipeline.sign_up(
                images=[],
                filenames=[],
                full_name="Test User",
                external_code="USER-1",
                enrollment_strategy="multi",
            )

        with self.assertRaises(standard_pipeline.FacePipelineError):
            standard_pipeline.sign_up(
                images=[image],
                filenames=[],
                full_name="Test User",
                external_code="USER-1",
                enrollment_strategy="multi",
            )

    def test_sign_up_requires_exactly_one_face_per_image(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)

        for detected_faces in ([], [FACE, FACE]):
            with (
                self.subTest(detected_faces=len(detected_faces)),
                patch.object(
                    standard_pipeline,
                    "detect_faces",
                    return_value=detected_faces,
                ),
            ):
                with self.assertRaises(standard_pipeline.FacePipelineError):
                    standard_pipeline.sign_up(
                        images=[image],
                        filenames=["face.png"],
                        full_name="Test User",
                        external_code="USER-1",
                        enrollment_strategy="multi",
                    )

    def test_sign_up_delegates_database_work_to_save_registration(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        saved_result = {
            "profile": {"id": "profile-1"},
            "pipeline": "standard",
        }

        with (
            patch.object(
                standard_pipeline,
                "detect_faces",
                return_value=[FACE],
            ),
            patch.object(
                standard_pipeline,
                "generate_arcface_embedding",
                return_value=EMBEDDING,
            ),
            patch.object(
                standard_pipeline,
                "save_registration",
                return_value=saved_result,
            ) as save_registration,
        ):
            result = standard_pipeline.sign_up(
                images=[image],
                filenames=["face.png"],
                full_name="Test User",
                external_code="USER-1",
                enrollment_strategy="multi",
            )

        save_registration.assert_called_once()
        self.assertEqual(result, saved_result)

    def test_sign_up_multi_saves_one_embedding_per_image(self):
        images = [
            np.zeros((120, 120, 3), dtype=np.uint8),
            np.ones((120, 120, 3), dtype=np.uint8),
        ]

        with (
            patch.object(
                standard_pipeline,
                "detect_faces",
                return_value=[FACE],
            ),
            patch.object(
                standard_pipeline,
                "generate_arcface_embedding",
                return_value=EMBEDDING,
            ),
            patch.object(
                standard_pipeline,
                "get_or_create_profile",
                return_value={"id": "profile-1"},
            ),
            patch.object(
                standard_pipeline,
                "upload_face_image",
                side_effect=["storage/one.png", "storage/two.png"],
            ),
            patch.object(
                standard_pipeline,
                "save_face_image",
                side_effect=[{"id": "image-1"}, {"id": "image-2"}],
            ),
            patch.object(
                standard_pipeline,
                "save_face_embedding",
                side_effect=[
                    {"id": "embedding-1", "embedding": EMBEDDING},
                    {"id": "embedding-2", "embedding": EMBEDDING},
                ],
            ) as save_embedding,
        ):
            result = standard_pipeline.sign_up(
                images=images,
                filenames=["one.png", "two.png"],
                full_name="Test User",
                external_code="USER-1",
                enrollment_strategy="multi",
            )

        self.assertEqual(result["processed_images"], 2)
        self.assertEqual(result["stored_embeddings"], 2)
        self.assertEqual(
            save_embedding.call_args_list[0].kwargs["image_id"],
            "image-1",
        )
        self.assertEqual(
            save_embedding.call_args_list[1].kwargs["image_id"],
            "image-2",
        )
        self.assertNotIn("embedding", result["face_embeddings"][0])

    def test_sign_up_centroid_saves_one_normalized_embedding(self):
        images = [
            np.zeros((120, 120, 3), dtype=np.uint8),
            np.ones((120, 120, 3), dtype=np.uint8),
        ]
        second_embedding = [0.0, 1.0] + [0.0] * 510

        with (
            patch.object(
                standard_pipeline,
                "detect_faces",
                return_value=[FACE],
            ),
            patch.object(
                standard_pipeline,
                "generate_arcface_embedding",
                side_effect=[EMBEDDING, second_embedding],
            ),
            patch.object(
                standard_pipeline,
                "get_or_create_profile",
                return_value={"id": "profile-1"},
            ),
            patch.object(
                standard_pipeline,
                "upload_face_image",
                side_effect=["storage/one.png", "storage/two.png"],
            ),
            patch.object(
                standard_pipeline,
                "save_face_image",
                side_effect=[{"id": "image-1"}, {"id": "image-2"}],
            ),
            patch.object(
                standard_pipeline,
                "save_face_embedding",
                return_value={"id": "embedding-1"},
            ) as save_embedding,
        ):
            result = standard_pipeline.sign_up(
                images=images,
                filenames=["one.png", "two.png"],
                full_name="Test User",
                external_code="USER-1",
                enrollment_strategy="centroid",
            )

        stored_embedding = save_embedding.call_args.kwargs["embedding"]
        self.assertEqual(result["stored_embeddings"], 1)
        self.assertIsNone(save_embedding.call_args.kwargs["image_id"])
        self.assertAlmostEqual(
            float(np.linalg.norm(stored_embedding)),
            1.0,
        )

    def test_sign_in_searches_with_normalized_embedding(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)

        with (
            patch.object(
                standard_pipeline,
                "detect_faces",
                return_value=[FACE],
            ),
            patch.object(
                standard_pipeline,
                "generate_arcface_embedding",
                return_value=[2.0] + [0.0] * 511,
            ),
            patch.object(
                standard_pipeline,
                "match_face_embedding",
                return_value={"matched": True},
            ) as search,
        ):
            result = standard_pipeline.sign_in(image=image)

        searched_embedding = search.call_args.kwargs["embedding"]
        self.assertAlmostEqual(
            float(np.linalg.norm(searched_embedding)),
            1.0,
        )
        self.assertTrue(result["matched"])
        self.assertEqual(result["pipeline"], "standard")


class HybridAnalysisTests(unittest.TestCase):
    def test_accepts_genuine_embedding(self):
        anomaly_detector = FakeAnomalyDetector(score=0.01)

        with patch.object(
            hybrid_pipeline,
            "reduce_embedding",
            return_value=[0.0] * 32,
        ):
            analysis = hybrid_pipeline.analyze_embedding(
                EMBEDDING,
                anomaly_detector,
            )

        self.assertEqual(analysis["umap_dimensions"], 32)
        self.assertEqual(analysis["anomaly_score"], 0.01)

    def test_rejects_anomalous_embedding(self):
        anomaly_detector = FakeAnomalyDetector(score=0.10)

        with patch.object(
            hybrid_pipeline,
            "reduce_embedding",
            return_value=[0.0] * 32,
        ):
            with self.assertRaises(hybrid_pipeline.AnomalyDetectedError):
                hybrid_pipeline.analyze_embedding(
                    EMBEDDING,
                    anomaly_detector,
                )


class CliAdapterTests(unittest.TestCase):
    def test_register_selects_standard_pipeline(self):
        args = Namespace(
            image_path="unused.png",
            method="standard",
            full_name="Test User",
            external_code="USER-1",
            enrollment_strategy="multi",
        )

        with (
            TemporaryDirectory() as temporary_directory,
            patch.object(
                standard_pipeline,
                "sign_up",
                return_value={"pipeline": "standard"},
            ) as sign_up,
        ):
            image_path = Path(temporary_directory) / "face.png"
            cv2.imwrite(
                str(image_path),
                np.zeros((10, 10, 3), dtype=np.uint8),
            )
            args.image_path = str(image_path)
            result = register.register_user(args)

        sign_up.assert_called_once()
        self.assertEqual(result["pipeline"], "standard")


if __name__ == "__main__":
    unittest.main()
