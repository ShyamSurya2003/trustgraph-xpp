# Deployment

## MongoDB Atlas

1. Create an Atlas cluster.
2. Create a database user.
3. Allow network access from Render.
4. Copy the connection string.
5. Set these Render environment variables:

```text
MONGODB_URI=mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB=trustgraph_xpp
```

The API stores each `/api/assessment` response in the `assessments` collection when `MONGODB_URI` is present.

## Render Backend

Use `render.yaml` from the project root, or create a Web Service manually:

```text
Root Directory: backend
Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
Python: 3.10.11
```

After deployment, copy the backend URL, for example:

```text
https://trustgraph-xpp-api.onrender.com
```

## Vercel Frontend

Create a Vercel project with:

```text
Root Directory: frontend
Framework: Vite
Build Command: npm run build
Output Directory: dist
```

Set:

```text
VITE_API_URL=https://YOUR_RENDER_BACKEND_URL
```

Then deploy. The final frontend URL will look like:

```text
https://trustgraph-xpp.vercel.app
```
