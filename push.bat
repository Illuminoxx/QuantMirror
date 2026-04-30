@echo off
echo.
echo ========================================
echo   SentimentEdge — Code Deploy
echo ========================================

echo.
echo [1/3] Syncing metrics...
copy /Y D:\sentimentEdge\backend\model_metrics.json D:\sentimentEdge\hf-sentimentedge\backend\

echo.
echo [2/3] Pushing to GitHub...
cd D:\sentimentEdge
git add .
git commit -m "update"
git push

echo.
echo [3/3] Pushing to HF Spaces...
cd D:\sentimentEdge\hf-sentimentedge
git add .
git commit -m "update"
git push origin main

echo.
echo ========================================
echo   Done!
echo ========================================
pause