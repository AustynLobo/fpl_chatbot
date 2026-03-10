import json
import boto3
import csv
import io
import urllib.request
import urllib.parse
import os
import boto3

dynamodb = boto3.resource("dynamodb")
table    = dynamodb.Table(os.environ["DYNAMODB_TABLE"])

MAX_HISTORY = 10  # keep last 10 messages per user

S3_BUCKET = os.environ["S3_BUCKET"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

FPL_KEYWORDS = [
    "player", "midfielder", "defender", "forward", "goalkeeper",
    "captain", "transfer", "price", "fixture", "fdr", "points",
    "gw", "gameweek", "buy", "sell", "bench", "squad", "team",
    "best", "value", "cheap", "recommend", "who", "should", "vice captain", 
    "triple captain", "team"
]

def get_history(chat_id):
    try:
        response = table.get_item(Key={"chat_id": str(chat_id)})
        return response.get("Item", {}).get("messages", [])
    except Exception:
        return []


def save_history(chat_id, messages):
    try:
        # keep only last MAX_HISTORY messages to avoid token limits
        messages = messages[-MAX_HISTORY:]
        table.put_item(Item={
            "chat_id"  : str(chat_id),
            "messages" : messages
        })
    except Exception:
        pass

def is_fpl_related(message):
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in FPL_KEYWORDS)


def get_latest_predictions():
    s3 = boto3.client("s3")
    
    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix="predictions/fpl_best_by_position_"
    )
    files = [obj["Key"] for obj in response.get("Contents", [])]
    latest = sorted(files)[-1]
    
    obj = s3.get_object(Bucket=S3_BUCKET, Key=latest)
    content = obj["Body"].read().decode("utf-8")
    
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    
    lines = []
    current_pos = None
    for row in rows:
        if row["Pos"] != current_pos:
            current_pos = row["Pos"]
            lines.append(f"\n{current_pos}:")
        lines.append(
            f"  {row['Player']:<20} Price: £{row['Price(£m)']}  "
            f"PredPts: {row['PredPts']}  FDR: {row['FDR']}  "
            f"Home: {row['Home']}  Value: {row['Value']}"
        )
    
    gw = latest.split("gw")[1].replace(".csv", "")
    return f"GW{gw} Predictions:\n" + "\n".join(lines)


def ask_claude(user_message, predictions_context, history):
    # build message list from history + current message
    messages = history + [
        {
            "role": "user",
            "content": f"FPL data:\n{predictions_context}\n\nQuestion: {user_message}"
        }
    ]

    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "system": (
            "You are an FPL (Fantasy Premier League) assistant. "
            "Answer questions using the prediction data provided. "
            "Be concise and helpful. Always mention player prices "
            "and predicted points when recommending players. "
            "Keep responses under 200 words as this is a Telegram chat."
        ),
        "messages": messages
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01"
        }
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read())
        return result["content"][0]["text"]


def send_telegram_message(chat_id, text):
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }).encode()

    req = urllib.request.Request(
        f"{TELEGRAM_API}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read())


def send_typing_action(chat_id):
    payload = json.dumps({
        "chat_id": chat_id,
        "action": "typing"
    }).encode()

    req = urllib.request.Request(
        f"{TELEGRAM_API}/sendChatAction",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=5)


def lambda_handler(event, context):
    try:
        body         = json.loads(event.get("body", "{}"))
        message      = body.get("message", {})
        chat_id      = message.get("chat", {}).get("id")
        user_message = message.get("text", "")

        if not chat_id or not user_message:
            return {"statusCode": 200, "body": "ok"}

        if user_message == "/start":
            # clear history on /start so conversation resets
            save_history(chat_id, [])
            send_telegram_message(
                chat_id,
                "👋 Welcome to the *FPL Predictor Bot*!\n\n"
                "Ask me anything about this gameweek:\n"
                "• Who are the best value midfielders?\n"
                "• Which defenders have easy fixtures?\n"
                "• Who should I captain this week?"
            )
            return {"statusCode": 200, "body": "ok"}

        if not is_fpl_related(user_message):
            send_telegram_message(
                chat_id,
                "I only answer FPL related questions! Try asking:\n"
                "• Who should I captain this week?\n"
                "• Best value midfielders?\n"
                "• Which defenders have easy fixtures?"
            )
            return {"statusCode": 200, "body": "ok"}

        send_typing_action(chat_id)

        # load conversation history
        history = get_history(chat_id)

        # get predictions and ask Claude with full history
        predictions = get_latest_predictions()
        answer      = ask_claude(user_message, predictions, history)

        # update history with this exchange
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": answer})
        save_history(chat_id, history)

        send_telegram_message(chat_id, answer)

        return {"statusCode": 200, "body": "ok"}

    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error"  : f"HTTP {e.code}",
                "detail" : error_body
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }