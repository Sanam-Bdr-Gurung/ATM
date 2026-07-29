from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tabs_guitar_dp import dp_tab_mapping


def main() -> None:
    open_midi = [40, 45, 50, 55, 59, 64]

    events = [
        {"t_on": 0.0, "t_off": 0.2, "midi": 40},
        {"t_on": 1.0, "t_off": 1.2, "midi": 45},
        {"t_on": 2.0, "t_off": 2.2, "midi": 50},
        {"t_on": 3.0, "t_off": 3.2, "midi": 55},
        {"t_on": 4.0, "t_off": 4.2, "midi": 59},
        {"t_on": 5.0, "t_off": 5.2, "midi": 64},
    ]

    expected = [
        {"t_on": 0.0, "string": 6, "fret": 0},
        {"t_on": 1.0, "string": 5, "fret": 0},
        {"t_on": 2.0, "string": 4, "fret": 0},
        {"t_on": 3.0, "string": 3, "fret": 0},
        {"t_on": 4.0, "string": 2, "fret": 0},
        {"t_on": 5.0, "string": 1, "fret": 0},
    ]

    actual = dp_tab_mapping(
        events,
        open_midi=open_midi,
        max_fret=20,
    )

    assert actual == expected, (
        f"\nExpected: {expected}\n"
        f"Actual:   {actual}"
    )

    print("Standard guitar string-number mapping passed.")


if __name__ == "__main__":
    main()