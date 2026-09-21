from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
DKUCC = HERE.parent
CONNECTION = DKUCC / "ssh_connection.json"
LOCAL_ROOT = HERE / "events" / "short_recovery_sub150_r3"
STATUS = HERE / "bridge_short_recovery_status.json"

REMOTE_ROOTS = {
    "smoke": "/work/qt28/moss/results/short-recovery-engineering/sr-v2-sub-150-20260916-r3-smoke/G2/sub/seed-0",
    "train": "/work/qt28/moss/results/short-recovery-sr-v2-sub-150-20260916-r3/sub/seed-0",
    "error_repeat_control": "/work/qt28/moss/results/error-repeat-controller-seed1-20260916",
    "error_repeat_smoke": "/work/qt28/moss/results/short-recovery-engineering/sr-v3-error-repeat-seed1-20260916-smoke/G2/sub_repeat/seed-1",
    "error_repeat_smoke_r3": "/work/qt28/moss/results/short-recovery-engineering/sr-v3-error-repeat-seed1-20260916-smoke-r3/G2/sub_repeat/seed-1",
    "error_repeat_train": "/work/qt28/moss/results/short-recovery-sr-v3-error-repeat-seed1-20260916/sub_repeat/seed-1",
    "base_error_repeat_control": "/work/qt28/moss/results/base-error-repeat-controller-seed1-20260916",
    "base_error_repeat_smoke": "/work/qt28/moss/results/short-recovery-engineering/sr-v3-base-error-repeat-seed1-20260916-smoke/G2/sub_repeat/seed-1",
    "base_error_repeat_train": "/work/qt28/moss/results/short-recovery-sr-v3-base-error-repeat-seed1-20260916/sub_repeat/seed-1",
}


def ssh_prefix() -> list[str]:
    cfg = json.loads(CONNECTION.read_text(encoding="utf-8-sig"))
    return [
        r"C:\Program Files\Git\usr\bin\ssh.exe",
        "-T",
        "-O",
        "proxy",
        "-S",
        cfg["control_path"],
        "-o",
        "BatchMode=yes",
        "qt28@dkucc-login-01.rc.duke.edu",
    ]


def remote(command: str, *, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ssh_prefix() + [command],
        check=True,
        capture_output=True,
        text=not binary,
        timeout=120,
    )


def discover() -> list[dict]:
    roots = " ".join(f"'{root}'" for root in REMOTE_ROOTS.values())
    command = (
        f"find {roots} -type f -name 'events.out.tfevents.*' "
        "-printf '%p\\t%s\\t%T@\\n' 2>/dev/null || true"
    )
    rows = []
    for line in remote(command).stdout.splitlines():
        path, size, mtime = line.split("\t")
        rows.append({"path": path, "size": int(size), "mtime": float(mtime)})
    return rows


def local_path(remote_path: str) -> Path:
    for label, root in REMOTE_ROOTS.items():
        prefix = root.rstrip("/") + "/"
        if remote_path.startswith(prefix):
            relative = PurePosixPath(remote_path[len(prefix) :])
            return LOCAL_ROOT / label / Path(*relative.parts)
    raise ValueError(f"unexpected remote path: {remote_path}")


def sync_one(item: dict) -> bool:
    destination = local_path(item["path"])
    if destination.exists() and destination.stat().st_size == item["size"]:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = remote(f"cat -- '{item['path']}'", binary=True)
    if len(result.stdout) != item["size"]:
        raise RuntimeError(
            f"size changed during copy for {item['path']}: "
            f"expected {item['size']}, received {len(result.stdout)}"
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(result.stdout)
    os.replace(temporary, destination)
    return True


def write_status(**payload: object) -> None:
    payload.update({"pid": os.getpid(), "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, STATUS)


def main() -> None:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            items = discover()
            changed = sum(sync_one(item) for item in items)
            write_status(
                state="running",
                remote_files=len(items),
                copied_or_updated=changed,
                local_root=str(LOCAL_ROOT),
                tensorboard_url="http://127.0.0.1:6006",
            )
        except Exception as exc:
            write_status(state="retrying", error=repr(exc))
        time.sleep(20)


if __name__ == "__main__":
    main()
