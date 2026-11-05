"""Prints which assignment sub-tasks have produced evidence on disk.

Run from the project root:
    python check_progress.py

It only checks things that leave a file behind. DagsHub runs, Docker images,
the cloud deployment and your screenshots cannot be seen from here, so those
are listed at the end as manual checks.
"""

import os


def exists(path):
    return os.path.exists(path)


def any_matching(folder, must_contain):
    if not os.path.isdir(folder):
        return []
    found = []
    for name in os.listdir(folder):
        keep = True
        for piece in must_contain:
            if piece not in name:
                keep = False
        if keep:
            found.append(name)
    return found


def report(label, marks, done, note):
    if done:
        mark = "[x]"
    else:
        mark = "[ ]"
    line = mark + " " + label.ljust(34) + str(marks).rjust(5) + "  " + note
    print(line)
    if done:
        return marks
    return 0


def main():
    print("")
    print("VisionGuard progress, evidence found on disk")
    print("=" * 74)

    earned = 0
    total = 0

    checks = []
    checks.append(("1.1 environment", 1, exists("requirements.txt") and exists(".venv"), "screenshot still needed"))
    checks.append(("1.2 dataset audit", 3, exists("artifacts/dataset_inventory.csv"), ""))
    checks.append(("1.3 visual EDA", 1.5, exists("artifacts/eda_samples.png"), ""))
    checks.append(("1.4 DVC", 1.5, exists("data/raw.dvc") and exists(".dvc/config.local"), "toy lab + push are manual"))
    checks.append(("2.1 input pipeline", 2, exists("artifacts/augmented_batch.png"), ""))
    checks.append(("2.2 baseline CNN", 3, len(any_matching("artifacts", ["run1", "test_metrics"])) > 0, ""))

    transfer_done = len(any_matching("artifacts", ["mnv2", "test_metrics"])) > 0
    checks.append(("2.3 transfer learning", 3, transfer_done, ""))
    checks.append(("2.4 training discipline", 1, len(any_matching("artifacts", ["curves"])) > 0, ""))
    checks.append(("2.5 evaluation", 2, len(any_matching("artifacts", ["confusion"])) > 0, ""))
    checks.append(("2.6 error analysis", 1, len(any_matching("artifacts", ["errors.png"])) > 0, ""))
    checks.append(("2.7 keras tuner", 2, exists("artifacts/tuner_best.json"), ""))
    checks.append(("3.1 mlflow local toy", 2, exists("mlruns"), ""))
    checks.append(("3.5 six+ runs", 2, len(any_matching("artifacts", ["test_metrics"])) >= 6, ""))
    checks.append(("3.6 champion model", 1.5, exists("models/champion.keras"), ""))
    checks.append(("Task 5 database trail", 8, exists("visionguard.db"), "needs real uploads, not the test db"))

    for label, marks, done, note in checks:
        earned = earned + report(label, marks, done, note)
        total = total + marks

    print("=" * 74)
    print("Checkable here: " + str(earned) + " of " + str(total) + " marks have evidence on disk")
    print("")
    print("Cannot be checked from here, verify these yourself:")
    print("  3.2 3.3 3.4  open DagsHub, Experiments tab, count your runs")
    print("  4.x          the three error pages and the health check screenshots")
    print("  6.x          run `docker images` and `docker compose ps`")
    print("  7.x 8.x      is your API URL live, and do you have the Postman shots")
    print("  report       is there a PDF, and does every row above have a caption")
    print("")
    print("Remember: a file here is only worth marks once it is IN the report,")
    print("captioned and explained. The files are the raw material, not the grade.")
    print("")


if __name__ == "__main__":
    main()
