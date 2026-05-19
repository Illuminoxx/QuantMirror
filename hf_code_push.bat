@echo off
echo.
echo ========================================
echo   Push to HF Spaces
echo ========================================
echo.
cd D:\sentimentEdge\hf-sentimentedge
git add .
git commit -m "code update" 2>nul || echo Nothing new to commit, skipping...
git push origin main --force
echo.
echo ========================================
echo   Done! HF Spaces updated.
echo ========================================
pause