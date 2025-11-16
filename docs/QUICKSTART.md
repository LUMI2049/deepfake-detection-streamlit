# ⚡ Quick Start Guide - Five-Minute Deployment

Comprehensive instructions for rapid deployment of the DeepFake Detection System.

---

## 🚀 Step 1: Create GitHub Repository (2 minutes)

### 1.1 Navigate to GitHub
Access: [github.com/new](https://github.com/new)

### 1.2 Repository Creation
- **Repository name:** `deepfake-detection-streamlit`
- **Description:** `AI-Generated Image Detection for AICS 2025`
- **Visibility:** ✅ **Public** (required for Streamlit Cloud free tier)
- **Initialization options:** Do NOT enable any initialization settings
- Select **"Create repository"**

### 1.3 Push Code to Remote Repository

Execute the following commands sequentially in Terminal:

```bash
cd /Users/dr.sam/Desktop/deepfake-detection-streamlit

git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection-streamlit.git

git branch -M main

git push -u origin main
```

**Important:** Replace `YOUR_USERNAME` with the appropriate GitHub username.

✅ **Repository Setup Complete** - Code is now available on GitHub.

---

## 🌐 Step 2: Deploy on Streamlit Cloud (3 minutes)

### 2.1 Access Streamlit Cloud
Navigate to: [share.streamlit.io](https://share.streamlit.io/)

### 2.2 Authentication
Select **"Continue with GitHub"** and authorize Streamlit to access GitHub account.

### 2.3 Application Deployment Configuration
1. Select **"New app"** (located in top-right corner)
2. Complete deployment form:
   - **Repository:** `YOUR_USERNAME/deepfake-detection-streamlit`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** `deepfake-detection-aics2025` (or preferred custom URL)
3. Select **"Deploy!"**

### 2.4 Build Process Monitoring
⏳ **Estimated duration:** 3-4 minutes

Expected build log messages:
```
✅ Installing dependencies...
✅ Starting app...
✅ Model downloading... (one-time operation)
✅ Your app is live!
```

---

## 🎉 Step 3: Application Verification

1. **Access deployment URL:**
   ```
   https://deepfake-detection-aics2025.streamlit.app
   ```

2. **Upload test image**

3. **Initiate analysis** by selecting "Analyze Image"

4. **Verify detection results**

---

## ✅ Deployment Verification Checklist

- ✅ Code successfully pushed to GitHub
- ✅ Application deployed on Streamlit Cloud
- ✅ Application loads without errors
- ✅ Image upload functionality operational
- ✅ Prediction system functions correctly
- ✅ Deployment URL shared with team members

---

## 🆘 Troubleshooting

### GitHub Push Authentication Error
```bash
# If authentication error occurs, utilize GitHub Personal Access Token
# Navigate to: github.com/settings/tokens → Generate new token
# Use token as password when authentication is requested
```

### Streamlit Build Failure
- Review build logs for specific error messages
- Common issue: Repository name contains typographical error
- Resolution: Verify all configuration settings and reinitiate deployment

### Application Performance Issues
- **Initial load:** 30-60 seconds (model download) - This is expected behavior
- **Subsequent inference:** 2-3 seconds per image
- If performance issues persist after multiple attempts, consult Streamlit Cloud status page

---

## 📞 Additional Resources

1. Comprehensive deployment guide: `DEPLOYMENT.md`
2. System overview: `README.md`
3. Community support: [Streamlit Community Forum](https://discuss.streamlit.io/)

---

<div align="center">
  <strong>🎯 Deployment Complete - Ready for AICS 2025</strong>
  <br>
  <sub>Total deployment time: approximately 5 minutes</sub>
</div>
