#!/usr/bin/env bash
# 在项目根目录创建 <学号>_project_phase2/ 并复制五个 .py。截图与 zip 见 phase2.md「提交要求」。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
STUDENT_ID="${1:-}"

usage() {
  echo "用法: $0 <学号>"
  exit 1
}

if [[ -z "$STUDENT_ID" || "$STUDENT_ID" == -* ]]; then
  usage
fi

if ! [[ "$STUDENT_ID" =~ ^[0-9]+$ ]]; then
  echo "错误: 学号应为数字，例如 5213（对应目录名 ${STUDENT_ID}_project_phase2）。" >&2
  exit 1
fi

NAME="${STUDENT_ID}_project_phase2"
OUT="$ROOT/$NAME"

FILES=(
  "tiny_inference/cache.py"
  "tiny_inference/manual_attention.py"
  "tiny_inference/manual_linear.py"
  "tiny_inference/manual_qwen3_5.py"
  "tiny_inference/manual_decoding.py"
)

for f in "${FILES[@]}"; do
  if [[ ! -f "$ROOT/$f" ]]; then
    echo "错误: 缺少文件 $ROOT/$f" >&2
    exit 1
  fi
done

rm -rf "$OUT"
mkdir -p "$OUT"
for f in "${FILES[@]}"; do
  cp -a "$ROOT/$f" "$OUT/"
done

echo "已创建: $OUT"
echo "请将 test_phase2.py 的 Speed Comparison 截图放入该文件夹，并压缩为 zip 后上传到 canvas"
