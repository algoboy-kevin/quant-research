def pytest_configure(config):
    config.addinivalue_line("markers", "slow: calibration runs on generated worlds (tens of seconds)")
