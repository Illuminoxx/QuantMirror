@echo off
echo.
echo ========================================
echo   SentimentEdge — Model Sync
echo ========================================
echo.
echo [1/3] Copying model to HF Spaces...
copy /Y D:\sentimentEdge\backend\rf_model.joblib D:\sentimentEdge\hf-sentimentedge\backend\rf_model.joblib
echo.
echo [2/3] Pushing to GitHub...
cd D:\sentimentEdge
git config lfs.https://github.com/Illuminoxx/QuantMirror.git/info/lfs.locksverify false
git config --global http.proxy ""
git config --global https.proxy ""
git add .gitignore
git add backend\rf_model.joblib
git commit -m "update trained model" 2>nul || echo Nothing new to commit, skipping...
git push origin main --force
echo [3/3] Pushing to HF Spaces...
cd D:\sentimentEdge\hf-sentimentedge
git add backend\rf_model.joblib --force
git commit -m "update trained model"
git push origin main --force
echo.
echo ========================================
echo   Done! New model live on HF Spaces.
echo ========================================
pause