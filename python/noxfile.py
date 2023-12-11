import os

import nox
from nox.command import CommandFailed
from nox.sessions import Session

SHELL_COMPLETION_DESC = """
# Shell Completion

- Bash:
  `eval "$(register-python-argcomplete nox)"`
- Zsh:
  ``` shell
  # To activate completions for zsh you need to have
  # bashcompinit enabled in zsh:
  autoload -U bashcompinit
  bashcompinit

  # Afterwards you can enable completion for Nox:
  eval "$(register-python-argcomplete nox)"
  ```

For more shells refer to:
[Shell Completion](https://nox.thea.codes/en/stable/usage.html#shell-completion)
"""
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


class chdir:
    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        self.cwd = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, *_):
        os.chdir(self.cwd)


@nox.session(venv_backend="none")
def shell_completion(session: Session):
    session.run("echo", SHELL_COMPLETION_DESC, silent=True)


@nox.session(venv_backend="none")
def format(session: Session):
    try:
        session.run("taplo", "fmt", "pyproject.toml")
    except CommandFailed:
        session.warn(
            "Seems that `taplo` is not found, skip formatting pyproject.toml. "
            "(Refer to https://taplo.tamasfe.dev/ for information on how to install "
            "`taplo`)"
        )

    items = ["npblob", "tests", "noxfile.py"]

    session.run("autoflake", *items)
    session.run("isort", *items)
    session.run("ruff", "format", *items)
