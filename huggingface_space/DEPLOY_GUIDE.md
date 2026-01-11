# 🚀 Deploy to Hugging Face Spaces - Step by Step

This guide will help you deploy your empathetic chatbot to Hugging Face Spaces with **free GPU**.

---

## Prerequisites

1. ✅ Hugging Face account ([sign up here](https://huggingface.co/join))
2. ✅ Your model uploaded to HuggingFace Hub (from the notebook)

---

## Step 1: Update the Model ID

Edit `app.py` and replace `YOUR_USERNAME` with your actual HuggingFace username:

```python
MODEL_ID = "YOUR_USERNAME/empathetic-qwen3-8b"  # <-- Change this!
```

Also update the footer in `app.py`:
```python
Made with 💚 | <a href="https://github.com/YOUR_USERNAME">GitHub</a> | 
Model: <a href="https://huggingface.co/YOUR_USERNAME/empathetic-qwen3-8b">empathetic-qwen3-8b</a>
```

And update `README.md`:
```yaml
models:
  - YOUR_USERNAME/empathetic-qwen3-8b
```

---

## Step 2: Create a New Space

1. Go to: https://huggingface.co/new-space

2. Fill in the details:
   - **Owner**: Your username
   - **Space name**: `empathetic-friend` (or whatever you like)
   - **License**: Apache 2.0
   - **SDK**: Gradio
   - **Hardware**: **T4 small** (free GPU!) or T4 medium for better performance

3. Click **Create Space**

---

## Step 3: Upload Files

### Option A: Web Upload (Easiest)

1. In your new Space, click **Files** tab
2. Click **+ Add file** → **Upload files**
3. Upload these 3 files from `huggingface_space/` folder:
   - `app.py`
   - `requirements.txt`
   - `README.md`

### Option B: Git (Recommended for updates)

```bash
# Clone the Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/empathetic-friend
cd empathetic-friend

# Copy your files
copy ..\huggingface_space\* .

# Push to HuggingFace
git add .
git commit -m "Initial deployment"
git push
```

---

## Step 4: Wait for Build

1. Go to your Space: `https://huggingface.co/spaces/YOUR_USERNAME/empathetic-friend`
2. Watch the **Logs** tab for build progress
3. Wait 5-10 minutes for the model to download and load

---

## Step 5: Test It!

Once the status shows **Running**, your chatbot is live!

Share your Space URL with anyone:
```
https://huggingface.co/spaces/YOUR_USERNAME/empathetic-friend
```

---

## Troubleshooting

### "Out of memory" Error

1. Go to Space **Settings**
2. Change Hardware to **T4 medium** or **A10G small**
3. Click **Apply**

### "Model not found" Error

Make sure:
1. Your model is uploaded and public on HuggingFace
2. The `MODEL_ID` in `app.py` matches exactly

### Slow Responses

- T4 small: ~10-20 seconds per response
- T4 medium: ~5-10 seconds per response
- A10G: ~2-5 seconds per response

Consider upgrading hardware for faster responses.

### Build Fails

Check the Logs tab for errors. Common issues:
- Missing dependency in requirements.txt
- Syntax error in app.py
- Model access issues (make sure model is public)

---

## Embedding in Your Website

Add your Space to any website with an iframe:

```html
<iframe
    src="https://YOUR_USERNAME-empathetic-friend.hf.space"
    frameborder="0"
    width="100%"
    height="700"
></iframe>
```

---

## Custom Domain (Optional)

1. Go to Space **Settings**
2. Add your custom domain
3. Configure DNS as instructed

---

## Cost

| Hardware | Cost | Speed |
|----------|------|-------|
| T4 small | **Free** (community GPU) | ~15 sec/response |
| T4 medium | $0.60/hr | ~8 sec/response |
| A10G small | $1.05/hr | ~3 sec/response |

The **T4 small** is free but may have queues during peak times.

---

## Next Steps

1. ⭐ Star your Space to help others find it
2. 📝 Add more examples to the interface
3. 🎨 Customize the CSS for your brand
4. 📊 Check Space analytics for usage stats

---

🎉 **Congratulations!** Your empathetic chatbot is now live on the web!

