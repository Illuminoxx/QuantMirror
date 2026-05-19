@echo off
echo.
echo ========================================
echo   Push to GitHub
echo ========================================
echo.
cd D:\sentimentEdge
git config lfs.https://github.com/Illuminoxx/QuantMirror.git/info/lfs.locksverify false
git config --global http.proxy ""
git config --global https.proxy ""
git add .
git commit -m "code update" 2>nul || echo Nothing new to commit, skipping...
git push origin main --force
echo.
echo ========================================
echo   Done! GitHub updated.
echo ========================================
pause