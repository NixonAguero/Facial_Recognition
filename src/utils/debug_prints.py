def print_match_results(result: dict) -> None:
    """
    Prints face matching results in a clear way for threshold calibration.
    """

    print("\n" + "=" * 80)
    print("FACE MATCHING RESULTS")
    print("=" * 80)

    print(f"Matched:   {result.get('matched')}")
    print(f"Threshold: {result.get('threshold')}")
    print(f"Distance:  {result.get('distance')}")

    if result.get("reason"):
        print(f"Reason:    {result['reason']}")

    matches = result.get("matches", [])

    if not matches:
        print("\nNo matches returned from database.")
        print("=" * 80)
        return

    print("\nRanked matches:")
    print("-" * 80)
    print(f"{'#':<4} {'Name':<20} {'Distance':<12} {'Decision':<12} {'Profile ID'}")
    print("-" * 80)

    threshold = result.get("threshold", 0.40)

    for index, match in enumerate(matches, start=1):
        distance = match.get("distance")
        full_name = match.get("full_name", "Unknown")
        profile_id = match.get("profile_id", "N/A")

        decision = "ACCEPT" if distance <= threshold else "REJECT"

        print(
            f"{index:<4} "
            f"{full_name:<20} "
            f"{distance:<12.6f} "
            f"{decision:<12} "
            f"{profile_id}"
        )

    print("-" * 80)

    best_match = result.get("best_match")

    if best_match:
        print("\nBest match:")
        print(f"Name:       {best_match.get('full_name')}")
        print(f"Profile ID: {best_match.get('profile_id')}")
        print(f"Image ID:   {best_match.get('image_id')}")
        print(f"Distance:   {best_match.get('distance'):.6f}")

        if best_match.get("distance") <= threshold:
            print("Final decision: ACCEPTED")
        else:
            print("Final decision: REJECTED")

    print("=" * 80 + "\n")