#!/bin/bash
# DIU 论文构建脚本
# 运行环境：Git Bash / WSL / macOS Terminal
# 依赖：pandoc, xelatex（TeX Live 或 MiKTeX）

cd "$(dirname "$0")"

FILES="DIU-01-intro.md \
       DIU-02-math.md \
       DIU-03-manifold.md \
       DIU-04-rfp-levels.md \
       DIU-05-theorems.md \
       DIU-06-operational.md \
       DIU-07-related-work.md \
       DIU-wip-supplements.md \
       DIU-appendix-a.md \
       DIU-appendix-b.md"

echo "=== 编译 PDF ==="
pandoc meta.yaml $FILES \
  --pdf-engine=xelatex \
  --toc \
  -o DIU.pdf
echo "PDF 输出：DIU.pdf"

echo ""
echo "=== 编译 Word ==="
pandoc meta.yaml $FILES \
  --toc \
  -o DIU.docx
echo "Word 输出：DIU.docx"

echo ""
echo "=== 完成 ==="
