def notify_opps(agent, opps):
    """
    For each opportunity dict in `opps`, format and send a Telegram message.
    Uses the agent's existing _format_notification_message() and send_telegram_notification().
    Returns the number of successful notifications.
    """
    if not opps:
        print("No opportunities to notify.")
        return 0
    sent = 0
    for opp in opps:
        try:
            msg = agent._format_notification_message(opp['event'], opp['probability'], opp['expected_value'])
            if agent.send_telegram_notification(msg):
                sent += 1
        except Exception as e:
            print(f"Notify error: {e}")
    print(f"Telegram: sent {sent}/{len(opps)}")
    return sent
