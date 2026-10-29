"""Smoke tests: run with  python -m pytest tests -q   (or  python tests/test_app.py)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def _score(at):
    html = next(m.value for m in at.markdown if 'class="hero"' in m.value)
    return int(re.search(r'class="hero">(\d+)%', html).group(1))


def _run():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    return at


def test_app_loads_without_errors():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 4


def test_high_risk_and_low_risk_customers_differ():
    at = _run()
    risky = _score(at)

    # Make the customer very safe: long tenure, two-year contract, no fibre
    at.slider[0].set_value(60)
    at.selectbox[0].set_value("Two year")
    at.selectbox[3].set_value("DSL")     # Internet service
    at.slider[1].set_value(40.0)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    safe = _score(at)
    print(f"   risky customer {risky}% vs safe customer {safe}%")
    assert risky > 50 and safe < 20


def test_no_internet_customer_does_not_crash():
    at = _run()
    at.selectbox[3].set_value("No")      # Internet service = No
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_no_phone_customer_does_not_crash():
    at = _run()
    at.selectbox[4].set_value("No")      # Phone service = No
    at.run()
    assert not at.exception, [e.value for e in at.exception]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
