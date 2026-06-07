from pathlib import Path

from pipelines.register_face_pipeline import register_face
from pipelines.identify_face_pipeline import identify_face


SIGN_UP_IMAGE = Path(__file__).parent.parent.parent / "dataset" / "sign_up" / "user3.png"
LOGIN_IMAGE = Path(__file__).parent.parent.parent / "dataset" / "sign_in" / "user0.png"


print("Registering face...")

registration_result = register_face(
    full_name="Test User 0",
    external_code="TEST_USER_0",
    image_path=str(SIGN_UP_IMAGE),
)

print("Registration completed.")
print("Profile ID:", registration_result["profile"]["id"])
print("Image ID:", registration_result["face_image"]["id"])
print("Embedding ID:", registration_result["face_embedding"]["id"])


print("\nIdentifying face...")

identification_result = identify_face(
    image_path=str(LOGIN_IMAGE),
    threshold=0.40,
)

print("Identification result:")
print(identification_result)

if identification_result["matched"]:
    print("Access accepted.")
    print("Matched profile:", identification_result["best_match"]["full_name"])
    print("Distance:", identification_result["distance"])
else:
    print("Access rejected.")