# ⚡ Python URL Summerizer
Request format:
{
 "content":"url" 
} 

Takes in a url and automatically summerizes it's contents. Returns a json structure in the format
{
  "title":"url title", 
  "content" :"url summary" 
} 

# ⚡ Python Starter Function

A simple starter function. Edit `src/main.py` to get started and create something awesome! 🚀

## 🧰 Usage

### GET /

- Returns a "Hello, World!" message.

**Response**

Sample `200` Response:

```text
Hello, World!
```

### POST, PUT, PATCH, DELETE /

- Returns a "Learn More" JSON response.

**Response**

Sample `200` Response:

```json
{
  "motto": "Build like a team of hundreds_",
  "learn": "https://appwrite.io/docs",
  "connect": "https://appwrite.io/discord",
  "getInspired": "https://builtwith.appwrite.io"
}
```

## ⚙️ Configuration

| Setting           | Value                             |
|-------------------|-----------------------------------|
| Runtime           | Python (3.9)                      |
| Entrypoint        | `src/main.py`                     |
| Build Commands    | `pip install -r requirements.txt` |
| Permissions       | `any`                             |
| Timeout (Seconds) | 15                                |

## 🔒 Environment Variables

No environment variables required.
