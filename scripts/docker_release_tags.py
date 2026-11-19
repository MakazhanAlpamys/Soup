"""Only the highest stable release tag may publish the Docker image's latest tag."""

from __future__ import annotations

import argparse
import os
import re
import subprocess


def is_latest_release(release_tag: str) -> bool:
    """Compare against all fetched tags using Git's numeric version ordering."""
    result = subprocess.run(
        ["git", "tag", "--sort=-v:refname"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    newest = next(
        (
            tag
            for tag in result.stdout.splitlines()
            if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag)
        ),
        None,
    )
    return release_tag == newest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="The vX.Y.Z release tag being published")
    args = parser.parse_args()
    latest = is_latest_release(args.tag)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"latest={str(latest).lower()}\n")


if __name__ == "__main__":
    main()
