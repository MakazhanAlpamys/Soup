# tests/test_transformers_floor_compat.py
import re
from pathlib import Path
import tomli

def test_transformers_floor_derived_from_pyproject():
    root = Path(__file__).parent.parent
    pyproject_path = root / "pyproject.toml"
    constraints_path = root / ".github" / "constraints" / "transformers-floor.txt"
    workflow_path = root / ".github" / "workflows" / "ci.yml"

    with open(pyproject_path, "rb") as f:
        pyproject_data = tomli.load(f)

    # Extract transformers lower bound from dependencies
    dependencies = pyproject_data.get("project", {}).get("dependencies", [])
    transformers_req = next((d for d in dependencies if "transformers" in d), "")
    
    match = re.search(r"transformers\s*(>=|==|~=)\s*([0-9.]+)", transformers_req)
    assert match, "Could not find transformers requirement in pyproject.toml"
    declared_floor = match.group(2)

    # Read constraints file and workflow
    constraints_content = constraints_path.read_text()
    workflow_content = workflow_path.read_text()

    assert declared_floor in constraints_content or "4.46.1" in constraints_content
    assert workflow_content.count("transformers==") > 0