import json

from openscenesense.cli import main


def test_schema_command(capsys):
    assert main(["schema"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["properties"]["schema_version"]["const"] == "1.2"


def test_schema_flag_alias(capsys):
    assert main(["--schema"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["properties"]["schema_version"]["const"] == "1.2"


def test_check_command_reports_environment(capsys):
    exit_code = main(["check"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ffmpeg"]
    assert payload["ffprobe"]
    assert exit_code == 0
