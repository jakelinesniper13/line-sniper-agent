from linesniper_agent import LineSniperAgent
import os


def main() -> None:
    """Run the LineSniperAgent demo and send Telegram notifications."""
    agent = LineSniperAgent(bankroll=1000.0)

    # Hypothetical picks
    picks_data = [
        {"event": "Eagles vs Giants (moneyline on Eagles)", "probability": 0.60, "odds": 1.90},
        {"event": "Yankees vs Red Sox (run line -1.5)", "probability": 0.45, "odds": 2.40},
        {"event": "Lakers vs Bulls (over 215.5 total)", "probability": 0.52, "odds": 1.95},
        {"event": "Packers vs Bears (moneyline on Bears)", "probability": 0.35, "odds": 3.20},
    ]

    opportunities = agent.find_positive_ev_picks(picks_data)
    if opportunities:
        # Print results
        header = f"{'Event':40s}  {'Prob':>5s}  {'Odds':>6s}  {'ImpProb':>7s}  {'EV':>6s}  {'Kelly':>5s}  {'Bet':>7s}"
        print(header)
        print("-" * len(header))
        for opp in opportunities:
            print(
                f"{opp['event'][:40]:40s}  "
                f"{opp['probability']:.2f}  {opp['odds']:.2f}  {opp['implied_probability']:.2f}  "
                f"{opp['expected_value']:.3f}  {opp['kelly_fraction']:.2f}  ${opp['recommended_bet']:.2f}"
            )
            try:
                msg = agent._format_notification_message(opp)
                sent = agent.send_telegram_notification(msg)
                print(f"Telegram notification {'sent' if sent else 'failed'} for {opp['event']}")
            except Exception as e:
                print(f"Error sending Telegram notification for {opp['event']}: {e}")
    else:
        print("No positive EV opportunities were found in the synthetic dataset.")

    # Analyse a real dataset if available
    real_file = os.path.join(os.path.dirname(__file__), 'prices.csv')
    if os.path.exists(real_file):
        try:
            print("\nReal dataset analysis (top 5 opportunities):")
            real_opps = agent.analyse_prices_file(real_file, top_n=5)
            if real_opps:
                header = f"{'Team':25s}  {'Bookmaker':20s}  {'EV':>6s}  {'Kelly':>5s}  {'Bet':>7s}"
                print(header)
                print('-' * len(header))
                for opp in real_opps:
                    print(
                        f"{opp['team'][:25]:25s}  {opp['bookmaker'][:20]:20s}  {opp['expected_value']:.3f}  {opp['kelly_fraction']:.2f}  ${opp['recommended_bet']:.2f}"
                    )
                    try:
                        # Compose simple notification message for real dataset
                        msg = (
                            f"{opp['team']} at {opp['bookmaker']} has EV {opp['expected_value']:.3f}, "
                            f"Kelly {opp['kelly_fraction']:.2f}, bet ${opp['recommended_bet']:.2f}"
                        )
                        sent = agent.send_telegram_notification(msg)
                        print(f"Telegram notification {'sent' if sent else 'failed'} for {opp['team']}")
                    except Exception as e:
                        print(f"Error sending Telegram notification for {opp['team']}: {e}")
            else:
                print("No positive EV opportunities found in the real dataset.")
        except Exception as e:
            print(f"Error analysing real dataset: {e}")


if __name__ == '__main__':
    main()
