# PowerShell script to set up GitHub repository and create releases
# Run this script after creating a GitHub repository

Write-Host "=== GitHub Repository Setup for TransactionModel ===" -ForegroundColor Green
Write-Host ""

Write-Host "Step 1: Create a new repository on GitHub:" -ForegroundColor Yellow
Write-Host "1. Go to https://github.com/new" -ForegroundColor Cyan
Write-Host "2. Repository name: TransactionModel" -ForegroundColor Cyan
Write-Host "3. Description: Enhanced SMS Transactional Classifier with Multi-Task Learning" -ForegroundColor Cyan
Write-Host "4. Make it Public or Private as you prefer" -ForegroundColor Cyan
Write-Host "5. DO NOT initialize with README, .gitignore, or license" -ForegroundColor Cyan
Write-Host "6. Click 'Create repository'" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 2: After creating the repository, copy the repository URL and run:" -ForegroundColor Yellow
Write-Host "git remote add origin <YOUR_REPOSITORY_URL>" -ForegroundColor Cyan
Write-Host "git branch -M main" -ForegroundColor Cyan
Write-Host "git push -u origin main" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 3: Create a release with model artifacts:" -ForegroundColor Yellow
Write-Host "1. Go to your repository on GitHub" -ForegroundColor Cyan
Write-Host "2. Click 'Releases' on the right side" -ForegroundColor Cyan
Write-Host "3. Click 'Create a new release'" -ForegroundColor Cyan
Write-Host "4. Tag version: v1.0.0" -ForegroundColor Cyan
Write-Host "5. Release title: Initial Release - SMS Transactional Classifier" -ForegroundColor Cyan
Write-Host "6. Description: Copy from README_ENHANCED.md" -ForegroundColor Cyan
Write-Host "7. Upload model files as assets:" -ForegroundColor Cyan
Write-Host "   - artifacts/sms_classifier.tflite" -ForegroundColor Cyan
Write-Host "   - artifacts_enhanced/sms_classifier.tflite" -ForegroundColor Cyan
Write-Host "   - artifacts_multi_task/sms_multi_task.tflite" -ForegroundColor Cyan
Write-Host "   - artifacts_enhanced/tokenizer.spm" -ForegroundColor Cyan
Write-Host "   - artifacts_enhanced/labels.json" -ForegroundColor Cyan
Write-Host "8. Click 'Publish release'" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 4: Verify the setup:" -ForegroundColor Yellow
Write-Host "git remote -v" -ForegroundColor Cyan
Write-Host "git status" -ForegroundColor Cyan
Write-Host ""

Write-Host "Your models will be available as downloadable assets in the GitHub release!" -ForegroundColor Green
Write-Host "Users can download the TFLite models directly from the release page." -ForegroundColor Green
