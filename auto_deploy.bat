@echo off
echo ===================================================
echo 🚀 INITIATING CLOUD DEPLOYMENT...
echo ===================================================
echo.

echo 1. Locking in the DeepFashion2 model weights...
git add fashion.pt

echo 2. Packaging the rest of your API code...
git add .

echo 3. Committing changes to history...
git commit -m "Automated deployment via batch script"

echo 4. Uploading 50MB to GitHub (Please wait 10-30 seconds)...
git push

echo.
echo ===================================================
echo ✅ UPLOAD COMPLETE!
echo Google Cloud Build is now deploying your API.
echo You can close this window.
echo ===================================================
pause
