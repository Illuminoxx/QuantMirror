@echo off
echo.
echo ========================================
echo   Push to GitHub - SentimentEdge
echo ========================================
echo.

cd D:\sentimentEdge

:: ── Proxy and LFS config ──
git config lfs.https://github.com/Illuminoxx/QuantMirror.git/info/lfs.locksverify false
git config --global http.proxy ""
git config --global https.proxy ""

:: ── Remove tracked files that should be ignored ──
git rm --cached *.bat 2>nul
git rm --cached *.docx 2>nul
git rm -r --cached backend/cache/ 2>nul
git rm -r --cached __pycache__/ 2>nul

:: ── Stage only what matters ──
git add backend/
git add pages/
git add app.js
git add index.html
git add requirements.txt
git add style.css
git add Dockerfile
git add README.md


:: ── Commit and push ──
git commit -m "code update" 2>nul || echo Nothing new to commit, skipping...
git push origin main --force

echo.
echo ========================================
echo   Done! GitHub updated.
echo ========================================
pause