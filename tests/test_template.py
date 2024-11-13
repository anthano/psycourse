import pytask
from _pytask.outcomes import ExitCode

from psycourse import config
from psycourse.config import ROOT


def test_pytask_build(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "BLD", tmp_path / "bld")

    session = pytask.build(
        config=ROOT / "pyproject.toml",
        force=True,
    )
    assert session.exit_code == ExitCode.OK