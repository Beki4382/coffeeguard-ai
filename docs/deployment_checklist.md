# Deployment Checklist

- Create a Hugging Face Space using the Gradio SDK.
- Upload `app.py`, `src/`, `requirements.txt`, `models/best_model.pt`, and `models/label_map.json`.
- Confirm the app starts without dependency errors.
- Test one image from each class.
- Test one poor-quality image.
- Test one invalid/non-leaf image.
- Add the working app URL to the final report and README.
