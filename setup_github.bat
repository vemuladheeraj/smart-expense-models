@echo off
echo === GitHub Repository Setup for TransactionModel ===
echo.

echo Step 1: Create a new repository on GitHub:
echo 1. Go to https://github.com/new
echo 2. Repository name: TransactionModel
echo 3. Description: Enhanced SMS Transactional Classifier with Multi-Task Learning
echo 4. Make it Public or Private as you prefer
echo 5. DO NOT initialize with README, .gitignore, or license
echo 6. Click 'Create repository'
echo.

echo Step 2: After creating the repository, copy the repository URL and run:
echo git remote add origin ^<YOUR_REPOSITORY_URL^>
echo git branch -M main
echo git push -u origin main
echo.

echo Step 3: Create a release with model artifacts:
echo 1. Go to your repository on GitHub
echo 2. Click 'Releases' on the right side
echo 3. Click 'Create a new release'
echo 4. Tag version: v1.0.0
echo 5. Release title: Initial Release - SMS Transactional Classifier
echo 6. Description: Copy from README_ENHANCED.md
echo 7. Upload model files as assets:
echo    - artifacts/sms_classifier.tflite
echo    - artifacts_enhanced/sms_classifier.tflite
echo    - artifacts_multi_task/sms_multi_task.tflite
echo    - artifacts_enhanced/tokenizer.spm
echo    - artifacts_enhanced/labels.json
echo 8. Click 'Publish release'
echo.

echo Step 4: Verify the setup:
echo git remote -v
echo git status
echo.

echo Your models will be available as downloadable assets in the GitHub release!
echo Users can download the TFLite models directly from the release page.
pause
