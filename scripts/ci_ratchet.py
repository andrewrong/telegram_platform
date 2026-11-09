#!/usr/bin/env python3
"""CI size ratchet (design dev-constraints.md part 4.2). Standard library only.

A source file that did not exist at the base is at most `new` lines; a file over
`cap` lines at HEAD must be listed in docs/design/ratchet.json and stay under its
ceiling; a ceiling is never raised. Test and generated files are exempt.
Usage: ci_ratchet.py --base <ref> [--config docs/design/ratchet.json]
"""
import argparse
import json
import os
import subprocess
import sys

CONFIG = "docs/design/ratchet.json"
EXEMPT_EXT = (".lock", ".json", ".sql", ".md", ".html", ".svg")
EXEMPT_PREFIX = ("web/src/generated/", "tests/")


def exempt(path):
    base = os.path.basename(path)
    if base.startswith("test_") and base.endswith(".py"):
        return True
    if path.endswith((".test.ts", "_test.go")) or base == "pnpm-lock.yaml":
        return True
    if path.startswith(EXEMPT_PREFIX) or "/tests/" in path:
        return True
    if path.startswith("web/dist"):
        return True
    return path.endswith(EXEMPT_EXT)


def count_lines(text):
    return text.count("\n") + (0 if text == "" or text.endswith("\n") else 1)


def defaults(cfg):
    return {
        "cap": cfg.get("cap", 800),
        "new": cfg.get("new", 600),
        "ceilings": dict(cfg.get("ceilings") or {}),
    }


def check(files, cur, old=None):
    """files: {path: (lines_at_head, existed_at_base)}. Returns (messages, checked)."""
    msgs, checked = [], 0
    for path in sorted(files):
        if exempt(path):
            continue
        checked += 1
        n, existed = files[path]
        if not existed:
            if n > cur["new"]:
                msgs.append(
                    f"ratchet: {path} is new with {n} lines; "
                    f"a new file is at most {cur['new']} lines"
                )
            continue
        if n <= cur["cap"]:
            continue
        ceiling = cur["ceilings"].get(path)
        if ceiling is None:
            msgs.append(
                f"ratchet: {path} is {n} lines, over {cur['cap']}, and has no ceiling in {CONFIG}"
            )
        elif n > ceiling:
            msgs.append(
                f"ratchet: {path} is {n} lines, over its ceiling {ceiling} "
                "(split it by a design first)"
            )
    if old is not None:
        for name in sorted(cur["ceilings"]):
            was = old["ceilings"].get(name, old["cap"])
            if cur["ceilings"][name] > was:
                msgs.append(
                    f"ratchet: ceiling of {name} raised from {was} to {cur['ceilings'][name]}; "
                    "only the owner raises a ceiling"
                )
    return msgs, checked


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return (p.returncode == 0, p.stdout)


def run(repo, base, config=CONFIG):
    ok, mb = git(repo, "merge-base", base, "HEAD")
    mb = mb.strip()
    if not ok or not mb:
        raise SystemExit(f"ratchet: no merge base of {base} and HEAD")
    ok, numstat = git(repo, "diff", "--numstat", "--no-renames", mb, "HEAD")
    if not ok:
        raise SystemExit(f"ratchet: cannot list the files changed since {mb}")
    try:
        with open(os.path.join(repo, config), encoding="utf-8") as fh:
            cur = defaults(json.load(fh))
    except OSError as err:
        raise SystemExit(f"ratchet: cannot read {config}: {err}")
    except ValueError as err:
        raise SystemExit(f"ratchet: {config} is not valid: {err}")
    old = None
    ok, raw = git(repo, "show", f"{mb}:{config}")
    if ok:
        try:
            old = defaults(json.loads(raw))
        except ValueError:
            raise SystemExit(f"ratchet: the base copy of {config} is not valid")
    files = {}
    for line in numstat.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3 or parts[0] == "-":  # binary files are skipped
            continue
        ok, head = git(repo, "show", f"HEAD:{parts[2]}")
        if not ok:  # deleted
            continue
        existed, _ = git(repo, "cat-file", "-e", f"{mb}:{parts[2]}")
        files[parts[2]] = (count_lines(head), existed)
    return check(files, cur, old)


def main(argv=None):
    ap = argparse.ArgumentParser(description=(
            "CI size ratchet: new files at most `new` lines, "
            "files over `cap` only under a ceiling."
        )
    )
    ap.add_argument("--base", required=True, help="base ref, for example origin/main")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--repo", default=".")
    args = ap.parse_args(argv)
    msgs, checked = run(args.repo, args.base, args.config)
    if msgs:
        print("\n".join(msgs))
        return 1
    print(f"ratchet: ok ({checked} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
