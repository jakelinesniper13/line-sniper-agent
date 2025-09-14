from linesniper_agent import LineSniperAgent
from notify_custom import notify_opps
import os


def main():
    agent = LineSniperAgent(bankroll=1000.0)
    # Synthetic demonstration picks
    picks_data = [
        {'event': 'Eagles vs Giants', 'probability': 0.61, 'odds': 1.8, 'implied_probability': 0.55,
         'expected_value': 0.06, 'kelly_fraction': 0.11, 'recommended_bet': 0.05},
        {'event': 'Cowboys vs Redskins', 'probability': 0.58, 'odds': 1.9, 'implied_probability': 0.53,
         'expected_value': 0.05, 'kelly_fraction': 0.10, 'recommended_bet': 0.05},
        {'event': 'Patriots vs Jets', 'probability': 0.52, 'odds': 2.1, 'implied_probability': 0.47,
         'expected_value': 0.05, 'kelly_fraction': 0.05, 'recommended_bet': 0.05},
    ]

    # Find positive EV opportunities in synthetic data
    opportunities = agent.find_positive_ev_picks(picks_data)
    # Print and notify synthetic opportunities
    if opportunities:
        for opp in opportunities:
            print(f"{opp['event']}: EV {opp['expected_value']:.2f}, Kelly {opp['kelly_fraction']:.2f}, Bet {opp['recommended_bet']:.2f}")
        notify_opps(agent, opportunities)
    else:
        print("No positive EV opportunities found in synthetic data.")

    # Analyze real dataset if present
    csv_path = os.path.join('data', 'prices.csv')
    if os.path.exists(csv_path):
        try:
            real_opps = agent.analyse_prices_file(csv_path)
            if real_opps:
                for opp in real_opps:
                    bookmaker = opp.get('bookmaker', 'Unknown')
                    print(f"{opp['event']} via {bookmaker}: EV {opp['expected_value']:.2f}, Kelly {opp['kelly_fraction']:.2f}")
                notify_opps(agent, real_opps)
            else:
                print("No positive EV opportunities found in real data.")
        except Exception as e:
            print(f"Error analyzing real data: {e}")
    else:
        print(f"File {csv_path} not found.")


if __name__ == '__main__':
    main()
