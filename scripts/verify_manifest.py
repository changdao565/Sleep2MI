from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "MANIFEST.sha256"
ENTRY_PATTERN = re.compile(r"^(?P<digest>[0-9a-fA-F]{64})  (?P<path>.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_manifest_path(relative_path: str) -> Path:
    portable = PurePosixPath(relative_path)
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"Unsafe manifest path: {relative_path}")
    resolved = (REPOSITORY_ROOT / Path(*portable.parts)).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Manifest path leaves repository: {relative_path}") from error
    return resolved


def load_manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        MANIFEST_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        match = ENTRY_PATTERN.fullmatch(raw_line)
        if match is None:
            raise ValueError(f"Malformed manifest line {line_number}: {raw_line!r}")
        relative_path = match.group("path")
        if relative_path in entries:
            raise ValueError(f"Duplicate manifest path: {relative_path}")
        resolve_manifest_path(relative_path)
        entries[relative_path] = match.group("digest").lower()
    return entries


def git_release_files() -> set[str] | None:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None
    return {
        path
        for path in completed.stdout.decode("utf-8").split("\0")
        if path and path != MANIFEST_PATH.name
    }


def write_manifest() -> int:
    release_files = git_release_files()
    if release_files is None:
        print("Manifest generation requires a Git checkout.", file=sys.stderr)
        return 1

    missing = [path for path in sorted(release_files) if not resolve_manifest_path(path).is_file()]
    if missing:
        print(
            "Cannot hash non-file release entries:\n" + "\n".join(missing),
            file=sys.stderr,
        )
        return 1

    lines = [
        f"{sha256(resolve_manifest_path(relative_path))}  {relative_path}"
        for relative_path in sorted(release_files)
    ]
    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} manifest entries.")
    return 0


def verify_manifest() -> int:
    try:
        entries = load_manifest()
    except (OSError, ValueError) as error:
        print(f"Manifest error: {error}", file=sys.stderr)
        return 1

    errors: list[str] = []
    for relative_path, expected_digest in sorted(entries.items()):
        path = resolve_manifest_path(relative_path)
        if not path.is_file():
            errors.append(f"Missing file: {relative_path}")
            continue
        actual_digest = sha256(path)
        if actual_digest != expected_digest:
            errors.append(
                f"Checksum mismatch: {relative_path}\n"
                f"  expected {expected_digest}\n"
                f"  actual   {actual_digest}"
            )

    release_files = git_release_files()
    if release_files is not None:
        manifest_files = set(entries)
        for relative_path in sorted(release_files - manifest_files):
            errors.append(f"Release file missing from manifest: {relative_path}")
        for relative_path in sorted(manifest_files - release_files):
            errors.append(f"Manifest entry is not a release file: {relative_path}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    coverage = " with complete Git coverage" if release_files is not None else ""
    print(f"Verified {len(entries)} manifest entries{coverage}.")
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}:
        print("Usage: python scripts/verify_manifest.py [--write]")
        return 0
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        return write_manifest()
    if len(sys.argv) != 1:
        print("Usage: python scripts/verify_manifest.py [--write]", file=sys.stderr)
        return 2
    return verify_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
