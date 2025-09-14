from linesniper_agent import LineSniperAgent
from notify_custom import notify_opps
import time


def main():
    agent = LineSniperAgent(bankroll=1000.0)
    # Synthetic demonstration picks (mock games)
    picks_data = [
        {'event': 'Eagles vs Giants', 'probability': 0.61, 'odds': 1.8,
         'implied_probability': 0.55,
         'expected_value': 0.06, 'kelly_fraction': 0.11, 'recommended_bet': 0.05},
        {'event': 'Cowboys vs Redskins', 'probability': 0.58, 'odds': 1.9,
         'implied_probability': 0.53,
         'expected_value': 0.05, 'kelly_fraction': 0.10, 'recommended_bet': 0.05},
        {'event': 'Patriots vs Jets', 'probability': 0.52, 'odds': 2.1,
         'implied_probability': 0.47,
         'expected_value': 0.05, 'kelly_fraction': 0.05, 'recommended_bet': 0.05},
    ]
    opportunities = agent.find_positive_ev_picks(picks_data)
    if opportunities:
        for opp in opportunities:
            print(f"{opp['event']}: EV {opp['expected_value']:.2f}, Kelly {opp['kelly_fraction']:.2f}, Bet {opp['recommended_bet']:.2f}")
        notify_opps(agent, opportunities)
    else:
        print("No positive EV opportunities found in synthetic data.")


if __name__ == '__main__':
    while True:
        main()
        # Sleep for 15 minutes (900 seconds)
        time.sleep(900)
