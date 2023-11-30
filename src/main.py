from appwrite.client import Client
import os,json
from .worker import eatUrl

# This is your Appwrite function
# It's executed each time we get a request
def main(context):
    # Why not try the Appwrite SDK?
    #
    # client = (
    #     Client()
    #     .set_endpoint("https://cloud.appwrite.io/v1")
    #     .set_project(os.environ["APPWRITE_FUNCTION_PROJECT_ID"])
    #     .set_key(os.environ["APPWRITE_API_KEY"])
    # )

    # You can log messages to the console
    context.log("Hello, Logs!")

    # If something goes wrong, log an error
    context.error("Hello, Errors!")

    # The `ctx.req` object contains the request data
    if context.req.method == "GET":
        # Send a response with the res object helpers
        # `ctx.res.send()` dispatches a string back to the client
        return context.res.send("Hello, World!")
    #if context.req.method == "POST":
    post_body = json.dumps(context.req.body)
    url_link = post_body["content"]
    url_sum_data = eatUrl(url_link)
    return context.res.json(url_sum_data)
    

    # `ctx.res.json()` is a handy helper for sending JSON
    #return context.res.json(
    #    {
    #        "motto": "Build like a team of hundreds_",
    #        "learn": "https://appwrite.io/docs",
    #        "connect": "https://appwrite.io/discord",
    #        "getInspired": "https://builtwith.appwrite.io",
    #    }
    #)
