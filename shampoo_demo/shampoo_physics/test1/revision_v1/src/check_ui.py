"""Headless Streamlit interaction checks; no server or browser is left running."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
from research_core import ROOT, write_json

app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
assert not app.exception, [str(e) for e in app.exception]
app.button(key="check_study").click().run()
assert not app.exception, [str(e) for e in app.exception]
assert not any("예측 보류" in w.value for w in app.warning)
unknown = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
unknown.session_state["study_recipe"] = {"edited_rows": {0: {"ingredient": "NEW_SUPPLIER_UNVERIFIED"}}, "added_rows": [], "deleted_rows": []}
unknown.button(key="check_study").click().run()
assert not unknown.exception, [str(e) for e in unknown.exception]
assert any("예측 보류" in w.value for w in unknown.warning)
write_json(ROOT / "results/ui_verification.json", {"status": "PASS",
    "checks": ["Streamlit initial rendering with research figures", "Supported source-assumption candidate flow", "Unknown ingredient edit produces HOLD"],
    "browser_visual_inspection": False, "method": "streamlit.testing.v1.AppTest"})
print("UI interaction checks passed.")
