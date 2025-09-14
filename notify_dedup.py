import random

# Set to keep track of notified picks
already_notified = set()

def notify_opps(agent, opps):
    """
    For each opportunity dict in `opps`, construct a custom Telegram message with dynamic commentary
    based on expected value. Avoid sending duplicate notifications for the same pick (event, bookmaker, expected value).
    Returns the number of successful notifications.
    """
    if not opps:
        print("No opportunities to notify.")
        return 0
    sent = 0
    for opp in opps:
        try:
            # Build unique key to avoid duplicates across runs
            key = (opp.get('event'), opp.get('bookmaker'), opp.get('expected_value'))
            if key in already_notified:
                continue
            event = opp.get('event')
            prob = opp.get('probability')
            ev = opp.get('expected_value')
            kelly = opp.get('kelly_fraction')
            stake = opp.get('recommended_bet')
            # Choose a comment based on magnitude of EV
            if ev is not None:
                if ev >= 0.75:
                    phrases = ["Massive edge here!", "This pick looks extremely valuable.", "High-return opportunity detected."]
                elif ev >= 0.50:
                    phrases = ["Great value pick with solid upside.", "Edge looks promising here.", "This wager stands out from the crowd."]
                elif ev >= 0.25:
                    phrases = ["Not a jackpot, but still worthwhile.", "Moderate edge – a nice addition.", "Steady gain potential on this pick."]
                else:
                    phrases = ["Small edge; proceed wisely.", "Marginal value – bet cautiously.", "Every bit counts, but keep stakes modest."]
                comment = random.choice(phrases)
            else:
                comment = ""
            lines = [f"LineSniper pick: {event}"]
            if prob is not None:
                lines.append(f"Win probability: {prob*100:.1f}%")
            if ev is not None:
                lines.append(f"Expected value: +{ev:.3f} units")
            if kelly is not None and stake is not None:
                lines.append(f"Recommended bet: {kelly*100:.1f}% of bankroll ({stake:.2f} units)")
            if comment:
                lines.append(comment)
            message = "\n".join(lines)
            if agent.send_telegram_notification(message):
                sent += 1
                # mark as notified
                already_notified.add(key)
        except Exception as e:
            print(f"Notify error: {e}")
    print(f"Telegram: sent {sent}/{len(opps)}")
    return sent
