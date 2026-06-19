import os
import cv2
from pathlib import Path


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def capture_full_frames(
    person_name: str,
    max_images: int = 100,
    save_every_n_frames: int = 5,
    camera_index: int = 0
):
    output_dir = Path(PROJECT_ROOT) / "dataset" / "own_faces" / "frames" / person_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    saved_count = 0
    frame_count = 0

    print("[INFO] Camera opened.")
    print("[INFO] Press 'q' to stop.")
    print("[INFO] Move your face slowly: front, left, right, up, down, different lighting.")

    while saved_count < max_images:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Could not read frame.")
            break

        frame_count += 1

        cv2.imshow("Capturing Frames", frame)

        if frame_count % save_every_n_frames == 0:
            saved_count += 1

            filename = f"{person_name}_frame_{saved_count:03d}.jpg"
            save_path = output_dir / filename

            success = cv2.imwrite(str(save_path), frame)

            if success:
                print(f"[SAVED] {save_path}")
            else:
                print(f"[ERROR] Could not save {save_path}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"[DONE] Saved {saved_count} frames in {output_dir}")


if __name__ == "__main__":
    capture_full_frames(
        person_name="ian",
        max_images=100,
        save_every_n_frames=5,
        camera_index=0
    )