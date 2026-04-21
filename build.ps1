Set-Location $PSScriptRoot

$files = @(
    "src/DIU-01-intro.md",
    "src/DIU-02-math.md",
    "src/DIU-03-manifold.md",
    "src/DIU-04-rfp-levels.md",
    "src/DIU-05-theorems.md",
    "src/DIU-06-operational.md",
    "src/DIU-07-related-work.md",
    "src/DIU-wip-supplements.md",
    "src/DIU-appendix-a.md",
    "src/DIU-appendix-b.md",
    "src/DIU-references.md"
)

New-Item -ItemType Directory -Force -Path "dist" | Out-Null

Write-Host "=== 编译 PDF ===" -ForegroundColor Cyan
pandoc meta.yaml @files --pdf-engine=xelatex --toc -o dist/DIU.pdf
if ($LASTEXITCODE -eq 0) {
    Write-Host "PDF 输出成功：dist/DIU.pdf" -ForegroundColor Green
} else {
    Write-Host "PDF 编译失败，请检查 xelatex 是否安装" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 编译 LaTeX（arXiv 投稿用）===" -ForegroundColor Cyan
pandoc meta-arxiv.yaml @files --standalone --toc -o dist/DIU.tex
if ($LASTEXITCODE -eq 0) {
    Write-Host "LaTeX 输出成功：dist/DIU.tex" -ForegroundColor Green
} else {
    Write-Host "LaTeX 编译失败" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 编译 Word ===" -ForegroundColor Cyan
pandoc meta.yaml @files --toc -o dist/DIU.docx
if ($LASTEXITCODE -eq 0) {
    Write-Host "Word 输出成功：dist/DIU.docx" -ForegroundColor Green
} else {
    Write-Host "Word 编译失败" -ForegroundColor Red
}

# ── 英文版 ──────────────────────────────────────────
$enFiles = @(
    "en/DIU-EN-01-intro.md",
    "en/DIU-EN-02-math.md",
    "en/DIU-EN-03-manifold.md",
    "en/DIU-EN-04-rfp-levels.md",
    "en/DIU-EN-05-theorems.md",
    "en/DIU-EN-06-operational.md",
    "en/DIU-EN-07-related-work.md",
    "en/DIU-EN-wip-supplements.md",
    "en/DIU-EN-appendix-a.md",
    "en/DIU-EN-appendix-b.md",
    "en/DIU-EN-disclosure.md",
    "en/DIU-EN-references.md"
)

Write-Host ""
Write-Host "=== 编译英文 PDF ===" -ForegroundColor Cyan
pandoc en/meta-en.yaml @enFiles --pdf-engine=xelatex --toc -o dist/DIU-EN.pdf
if ($LASTEXITCODE -eq 0) {
    Write-Host "英文 PDF 输出成功：dist/DIU-EN.pdf" -ForegroundColor Green
} else {
    Write-Host "英文 PDF 编译失败" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 编译英文 Word ===" -ForegroundColor Cyan
pandoc en/meta-en.yaml @enFiles --toc -o dist/DIU-EN.docx
if ($LASTEXITCODE -eq 0) {
    Write-Host "英文 Word 输出成功：dist/DIU-EN.docx" -ForegroundColor Green
} else {
    Write-Host "英文 Word 编译失败" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 编译英文 LaTeX（期刊投稿用）===" -ForegroundColor Cyan
pandoc en/meta-en.yaml @enFiles --standalone --toc -o dist/DIU-EN.tex
if ($LASTEXITCODE -eq 0) {
    Write-Host "英文 LaTeX 输出成功：dist/DIU-EN.tex" -ForegroundColor Green
} else {
    Write-Host "英文 LaTeX 编译失败" -ForegroundColor Red
}
