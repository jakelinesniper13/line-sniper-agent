# send_telegram_smoke_test.py
import os, sys, json, time
from urllib import request

BOT = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")

def send(text: str) -> bool:
    if not BOT or not CHAT:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return False
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    data = json.dumps({"chat_id": CHAT, "text": text}).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            ok = bool(body.get("ok"))
            print("Telegram OK" if ok else f"Telegram FAIL: {body}")
            return ok
    except Exception as e:
        print(f"Telegram FAIL: {e}")
        return False

if __name__ == "__main__":
    msg = os.getenv("MSG", f"LineSniper smoke test ✅ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.exit(0 if send(msg) else 1)
