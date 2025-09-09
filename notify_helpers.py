# notify_helpers.py

def notify_opps(agent, opps):
    """
    For each opportunity dict in `opps`, format with the agent's existing
    formatter and send via the agent's existing Telegram method.
    """
    if not opps:
        print("No opportunities to notify.")
        return 0
    sent = 0
    for opp in opps:
        try:
            msg = agent._format_notification_message(opp)
            if agent.send_telegram_notification(msg):
                sent += 1
        except Exception as e:
            print(f"Notify error: {e}")
    print(f"Telegram: sent {sent}/{len(opps)}")
    return sent
