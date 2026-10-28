from dataclasses import replace

import matplotlib
import pytest

import applied_config
from source.config import AppConfig
from source.sensor_mock import SyntheticSource

# Render figures off-screen so tests never open plot windows
matplotlib.use("Agg")


@pytest.fixture(name="config")
def fixture_config() -> AppConfig:
    """The applied settings with the four-sensor montage."""
    return applied_config.build_config(applied_config.FOUR_SENSORS)


def offline_synthetic(config: AppConfig) -> SyntheticSource:
    """A synthetic source that generates as fast as possible."""
    return SyntheticSource(replace(config.synthetic, realtime=False), config.channels)
