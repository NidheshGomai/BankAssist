# Deployment Guide - Streamlit Cloud

## Prerequisites
- GitHub account
- Streamlit account (free at https://streamlit.io)
- HuggingFace API token

## Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/BankAssist.git
git push -u origin main
```

## Step 2: Create HuggingFace Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Copy the token

## Step 3: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click "New app"
3. Connect your GitHub repository
4. Select branch: `main`
5. Set main file path: `app.py`
6. Click "Deploy"

## Step 4: Add Secrets
After deployment:
1. Go to your app settings (hamburger menu → Settings)
2. Go to "Secrets" tab
3. Add your HuggingFace token:
```
HF_TOKEN = "your_token_here"
```
4. Save

## Step 5: Access Your App
Your app will be live at:
`https://share.streamlit.io/YOUR_USERNAME/BankAssist`

## Environment Variables in Streamlit Cloud
- Secrets are automatically available via `streamlit secrets` 
- Update `app.py` to use:
```python
hf_token = st.secrets.get("HF_TOKEN")
login(token=hf_token)
```

## Limitations & Notes
- Free tier has resource limits (memory, CPU)
- Large model downloads may be slow on first load
- Consider caching models after first load
- You can add GitHub Actions for automated deployments

## Troubleshooting
- **Models not loading**: Check HuggingFace token in secrets
- **Out of memory**: Streamlit Cloud has ~1GB RAM limit; may need paid tier for larger models
- **Slow startup**: First deployment caches models; subsequent runs are faster

## Monitoring
- Check logs: Go to app settings → Manage app → View logs
- Monitor usage: Streamlit provides analytics in the dashboard
