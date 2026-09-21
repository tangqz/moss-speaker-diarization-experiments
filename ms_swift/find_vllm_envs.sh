#!/usr/bin/env bash
set -u
ROOT=/work/qt28/moss/envs
find "$ROOT" -maxdepth 3 -type f -path '*/bin/python' -print 2>/dev/null | while IFS= read -r py; do
  case "$py" in
    *vllm*|*moss*)
      version=$(timeout 20 "$py" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)
      [[ -z "$version" ]] || printf '%s\t%s\n' "$version" "$py"
      ;;
  esac
done

find /work/qt28/moss -maxdepth 5 -type f \( -name 'python' -o -name 'python3' \) -path '*vllm*' -print 2>/dev/null | while IFS= read -r py; do
  version=$(timeout 20 "$py" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)
  [[ -z "$version" ]] || printf '%s\t%s\n' "$version" "$py"
done
