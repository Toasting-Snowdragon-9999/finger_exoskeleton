"""
Calibration of EMG amplitudes (work in progress, not used yet).
"""

# pylint: disable=too-few-public-methods  # placeholders until calibration is implemented


class RVC:
    """
    Reference voluntary contraction: an alternative to MVC.

    Use a defined, submaximal task as 100 %, such as "hold a firm grip" or "lift 1 kg".
    It is easier and more repeatable for patients, but the scale is relative to that
    task, not to the person's true maximum. The deleted calibration did this: a
    "reference grip", not a maximum.
    """

    def __init__(self) -> None:
        pass


class Percentile:
    """Percentile-based normalisation (to be written)."""

    def __init__(self) -> None:
        pass


class Calib:
    """The calibration methods."""

    def __init__(self) -> None:
        self.mvc = RVC()
        self.perc = Percentile()
