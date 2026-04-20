Set-Location $PSScriptRoot

$files = @(
    "DIU-01-intro.md",
    "DIU-02-math.md",
    "DIU-03-manifold.md",
    "DIU-04-rfp-levels.md",
    "DIU-05-theorems.md",
    "DIU-06-operational.md",
    "DIU-07-related-work.md",
    "DIU-wip-supplements.md",
    "DIU-appendix-a.md",
    "DIU-appendix-b.md",
    "DIU-references.md"
)

Write-Host "=== 编译 PDF ===" -ForegroundColor Cyan
pandoc meta.yaml @files --pdf-engine=xelatex --toc -o DIU.pdf
if ($LASTEXITCODE -eq 0) {
    Write-Host "PDF 输出成功：DIU.pdf" -ForegroundColor Green
} else {
    Write-Host "PDF 编译失败，请检查 xelatex 是否安装" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 编译 Word ===" -ForegroundColor Cyan
pandoc meta.yaml @files --toc -o DIU.docx
if ($LASTEXITCODE -eq 0) {
    Write-Host "Word 输出成功：DIU.docx" -ForegroundColor Green
} else {
    Write-Host "Word 编译失败" -ForegroundColor Red
}
