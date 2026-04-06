"""Task 1.1 — Verify project infrastructure files are correctly configured."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent

REQUIRED_RUNTIME_DEPS = {
    "anthropic",
    "scikit-learn",
    "rich",
    "python-dotenv",
    "psycopg2-binary",
}

REQUIRED_DEV_DEPS = {"pytest", "pytest-mock"}

REQUIRED_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "DATABASE_URL",
    "EXTRACTION_MODEL",
    "GENERATION_MODEL",
    "MAX_HISTORY_TURNS",
]

REQUIRED_GITIGNORE_PATTERNS = [".env", "__pycache__"]


def _dep_name(dep: str) -> str:
    """Extract the bare package name from a PEP 508 dependency string."""
    return dep.split(">=")[0].split("==")[0].split("[")[0].strip().lower()


def test_pyproject_toml_exists():
    assert (ROOT / "pyproject.toml").exists(), "pyproject.toml not found at project root"


def test_pyproject_toml_is_valid_toml():
    with open(ROOT / "pyproject.toml", "rb") as fh:
        config = tomllib.load(fh)
    assert "project" in config


def test_pyproject_toml_has_required_runtime_dependencies():
    with open(ROOT / "pyproject.toml", "rb") as fh:
        config = tomllib.load(fh)
    dep_names = {_dep_name(d) for d in config["project"]["dependencies"]}
    missing = REQUIRED_RUNTIME_DEPS - dep_names
    assert not missing, f"Missing runtime dependencies: {missing}"


def test_pyproject_toml_has_required_dev_dependencies():
    with open(ROOT / "pyproject.toml", "rb") as fh:
        config = tomllib.load(fh)
    optional = config["project"].get("optional-dependencies", {})
    dev_deps = optional.get("dev", [])
    dep_names = {_dep_name(d) for d in dev_deps}
    missing = REQUIRED_DEV_DEPS - dep_names
    assert not missing, f"Missing dev dependencies: {missing}"


def test_pyproject_toml_requires_python_311_or_higher():
    with open(ROOT / "pyproject.toml", "rb") as fh:
        config = tomllib.load(fh)
    requires = config["project"].get("requires-python", "")
    assert requires, "requires-python not set in pyproject.toml"
    # Accept >=3.11, >=3.12, etc.
    version_str = requires.replace(">=", "").replace(">", "").strip()
    major, minor = map(int, version_str.split(".")[:2])
    assert (major, minor) >= (3, 11), f"requires-python must be >=3.11, got {requires}"


def test_env_example_exists():
    assert (ROOT / ".env.example").exists(), ".env.example not found at project root"


def test_env_example_documents_all_required_variables():
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    missing = [v for v in REQUIRED_ENV_VARS if v not in content]
    assert not missing, f"Missing env variables in .env.example: {missing}"


def test_gitignore_exists():
    assert (ROOT / ".gitignore").exists(), ".gitignore not found at project root"


def test_gitignore_excludes_required_patterns():
    content = (ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = [p for p in REQUIRED_GITIGNORE_PATTERNS if p not in content]
    assert not missing, f"Missing patterns in .gitignore: {missing}"


def test_gitignore_excludes_virtualenv_directories():
    content = (ROOT / ".gitignore").read_text(encoding="utf-8")
    venv_patterns = [".venv", "venv/"]
    present = [p for p in venv_patterns if p in content]
    assert present, "No virtual environment directory pattern found in .gitignore"


def test_package_source_directory_exists():
    assert (ROOT / "src" / "sommelier").is_dir(), "src/sommelier package directory not found"


def test_package_is_importable():
    import sys
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import importlib
    spec = importlib.util.find_spec("sommelier")
    assert spec is not None, "sommelier package is not importable"
