"""The logger finds the repository root in a git worktree, where .git is a file."""
import os

from experiments.nodes.experiment_logger import _find_repo_root


def test_repo_root_is_found_through_a_git_file(tmp_path):
    (tmp_path / '.git').write_text('gitdir: /elsewhere/.git/worktrees/x\n')
    nested = tmp_path / 'logs' / 'run'
    nested.mkdir(parents=True)
    assert _find_repo_root(str(nested)) == str(tmp_path)


def test_repo_root_is_found_through_a_git_directory(tmp_path):
    (tmp_path / '.git').mkdir()
    nested = tmp_path / 'a' / 'b'
    nested.mkdir(parents=True)
    assert _find_repo_root(str(nested)) == os.path.abspath(tmp_path)
