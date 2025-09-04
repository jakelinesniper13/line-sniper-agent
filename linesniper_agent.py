"""
linesniper_agent.py
====================

This module defines a simple `LineSniperAgent` class which can be used to
analyse sports betting lines and identify wagers with a positive expected
value (EV).  The design is intentionally straightforward; it does **not**
interact with any sportsbooks or place bets, and is therefore compliant
with the restrictions on gambling‑related financial transactions.  Instead,
the agent takes user‑provided probability estimates and bookmaker odds,
computes the implied probability, expected value and recommended bet size
using the Kelly criterion, and exposes helper methods for inspecting
potential opportunities.

The name “LineSniper” was inspired by the concept of “sniping” mispriced
betting lines: identifying prices that appear favourable relative to your
own probability estimates.  While this tool can help illustrate the
mathematics behind such decisions, it should not be misconstrued as
financial advice.  As always, gambling carries inherent risk and should
be approached responsibly.

Example usage
-------------

>>> from linesniper_agent import LineSniperAgent
>>> agent = LineSniperAgent(bankroll=1000)
>>> picks = [
...     {"event": "Team A vs Team B", "probability": 0.55, "odds": 2.10},
...     {"event": "Team C vs Team D", "probability": 0.40, "odds": 3.00},
... ]
>>> opportunities = agent.find_positive_ev_picks(picks)
>>> for opp in opportunities:
...     print(opp)

Running this script directly (`python linesniper_agent.py`) will
demonstrate the agent on a small, hard‑coded dataset.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import os
import requests


@dataclass
class Pick:
    """Represents a single betting opportunity.

    Attributes
    ----------
    event : str
        Human‑readable description of the sporting event.
    probability : float
        Your estimated probability of the outcome occurring (between 0 and 1).
    odds : float
        Decimal odds offered by the bookmaker.  Decimal odds represent the
        total payout, including stake, on a 1‑unit wager.  For example,
        `2.5` means that a winning bet returns 2.5 units (1.5 units profit).
    """

    event: str
    probability: float
    odds: float

    def implied_probability(self) -> float:
        """Compute the implied probability from the offered decimal odds.

        The implied probability is the bookmaker's assessment of the chance of
        the outcome occurring, derived from the odds.  It is calculated as
        1 / odds.  Note that bookmakers typically build a margin into the odds,
        so the implied probabilities of all outcomes will sum to more than 1.

        Returns
        -------
        float
            The implied probability (between 0 and 1).
        """
        return 1.0 / self.odds

    def expected_value(self) -> float:
        """Compute the expected value per unit stake.

        The expected value (EV) of a wager measures the average return
        assuming your estimated probability is correct.  It is calculated as:

        EV = (probability * (odds - 1)) - ((1 - probability) * 1)

        A positive EV indicates that the bet is, on average, profitable in
        the long run.  A negative EV suggests the opposite.

        Returns
        -------
        float
            The expected profit (or loss) per unit staked.  A value of 0.05
            means that for every 1 unit staked, you expect to earn 0.05 units
            on average.
        """
        win_profit = self.odds - 1.0  # profit if the bet wins
        lose_cost = 1.0  # cost if the bet loses (stake lost)
        return (self.probability * win_profit) - ((1.0 - self.probability) * lose_cost)

    def kelly_fraction(self) -> float:
        """Compute the Kelly criterion bet fraction.

        The Kelly criterion suggests how much of your bankroll to stake on a
        wager when you have a perceived edge.  It maximises the expected
        logarithm of wealth and is given by:

            f* = (bp - q) / b

        where b = odds - 1, p = probability, q = 1 - p.  The fraction is
        capped between 0 and 1 for practicality (no leveraging or shorting).

        Returns
        -------
        float
            Fraction of the bankroll that should be staked according to the
            Kelly criterion.  Values <= 0 indicate that no bet should be placed.
        """
        b = self.odds - 1.0
        p = self.probability
        q = 1.0 - p
        numerator = (b * p) - q
        if b == 0:
            return 0.0
        kelly = numerator / b
        # Constrain the fraction to [0, 1] to avoid over‑betting or shorting
        return max(0.0, min(1.0, kelly))


class LineSniperAgent:
    """A simple agent for identifying positive EV betting opportunities.

    Parameters
    ----------
    bankroll : float, optional
        Initial bankroll available for staking.  Used solely for computing
        recommended bet sizes via the Kelly criterion.  Defaults to 1000.
    """

    def __init__(self, bankroll: float = 1000.0, telegram_token: str | None = None, telegram_chat_id: str | None = None) -> None:
        """Initialise the LineSniperAgent.

        Parameters
        ----------
        bankroll : float, optional
            Initial bankroll available for staking.  Defaults to 1000.
        telegram_token : str, optional
            Telegram bot token used for sending notifications.  If not
            provided, the agent will try to read the `TELEGRAM_BOT_TOKEN`
            environment variable.
        telegram_chat_id : str, optional
            Telegram chat ID used for sending notifications.  If not
            provided, the agent will try to read the `TELEGRAM_CHAT_ID`
            environment variable.
        """
        self.bankroll = bankroll
        # store Telegram credentials if provided or fetch from environment
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")

        # Placeholder for a trained NFL model and break‑even threshold for point spreads.  The
        # model will be set by calling :meth:`train_nfl_model`.  The break‑even
        # probability reflects the fact that most point‑spread wagers are priced
        # at approximately –110 on both sides, which implies a win rate of
        # roughly 52.38 % is needed just to break even【916453135453690†L294-L323】.  It is
        # computed as 1 / (1 + 100/110) and will be used when scanning for
        # positive expected value opportunities.
        self.nfl_model = None  # type: Any
        self.nfl_break_even = 1.0 / (1.0 + 100.0 / 110.0)

        # Track predictions for future evaluation.  When scanning upcoming games
        # the agent stores the details of each prediction (e.g. favourite team,
        # event name, predicted probability, line and date).  After games
        # conclude, these records can be compared with actual results to
        # determine whether the prediction was correct.
        self._prediction_log: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------
    # Internal helpers for formatting notifications
    #
    def _format_notification_message(self, event: str, probability: float, expected_value: float) -> str:
        """Create a more engaging notification message.

        Parameters
        ----------
        event : str
            Human‑readable description of the event (e.g. "Team A vs Team B (spread)").
        probability : float
            Predicted probability that the bet wins (between 0 and 1).
        expected_value : float
            Expected value per unit stake.

        Returns
        -------
        str
            A formatted message including emojis and a concise summary of the
            edge.  The message avoids encouraging gambling behaviour but
            highlights the value opportunity.
        """
        # Convert probability to percent and expected value to three decimals
        prob_pct = probability * 100.0
        ev_units = expected_value
        # Craft a dry, humorous message without emojis.  The tone is
        # intentionally understated.  Feel free to adjust the phrasing to
        # suit your own sense of humour.
        lines = [
            f"LineSniper pick: {event}",
            f"Win probability: {prob_pct:.1f}%",
            f"Expected value: +{ev_units:.3f} units",
        ]
        # Add a tongue‑in‑cheek comment based on the edge
        if ev_units >= 0.5:
            comment = "This one might even make your accountant smile."
        elif ev_units >= 0.2:
            comment = "Not exactly thrilling, but money is money."
        else:
            comment = "Hey, at least the math checks out."
        lines.append(comment)
        return "\n".join(lines)

    def analyse_pick(self, pick: Pick) -> Dict[str, Any]:
        """Analyse a single pick and return computed metrics.

        Parameters
        ----------
        pick : Pick
            The betting opportunity to analyse.

        Returns
        -------
        dict
            A dictionary containing the event name, estimated probability,
            offered odds, implied probability, expected value per unit,
            Kelly fraction and recommended stake (bankroll * Kelly fraction).
        """
        implied_prob = pick.implied_probability()
        ev = pick.expected_value()
        kelly_frac = pick.kelly_fraction()
        recommended_bet = kelly_frac * self.bankroll
        return {
            "event": pick.event,
            "probability": pick.probability,
            "odds": pick.odds,
            "implied_probability": implied_prob,
            "expected_value": ev,
            "kelly_fraction": kelly_frac,
            "recommended_bet": recommended_bet,
        }

    def find_positive_ev_picks(self, picks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and analyse picks with positive expected value.

        Accepts a list of dictionaries, each with keys `event`, `probability`
        and `odds`.  For each entry, a `Pick` instance is constructed and
        analysed.  Only those with an expected value > 0 are returned.

        Parameters
        ----------
        picks : list of dict
            Each dict must contain the keys `event`, `probability` and
            `odds`.

        Returns
        -------
        list of dict
            A list of analysis dictionaries (as produced by
            :meth:`analyse_pick`) for picks with positive expected value.
        """
        analysed: List[Dict[str, Any]] = []
        for p in picks:
            try:
                pick = Pick(event=p["event"], probability=float(p["probability"]), odds=float(p["odds"]))
            except (KeyError, ValueError) as e:
                # Skip malformed entries
                continue
            result = self.analyse_pick(pick)
            if result["expected_value"] > 0:
                analysed.append(result)
        return analysed

    # -------------------------------------------------------------------------
    # Real data utilities
    #
    # The following helper methods extend the LineSniperAgent with the ability to
    # process a dataset of bookmaker prices.  They are separated from the core
    # EV/Kelly logic to keep the base functionality self‑contained and easy to
    # test.  These methods are optional and are only used when analysing
    # datasets such as the MLB moneyline prices provided in this repository.

    @staticmethod
    def american_to_decimal(american: float) -> float:
        """Convert American odds to decimal odds.

        American odds express the amount a bettor must wager or will win on a
        $100 stake.  Positive numbers indicate the potential profit on a $100
        bet; negative numbers indicate how much must be staked to win $100.

        Parameters
        ----------
        american : float
            The American odds (e.g., -150, +200).

        Returns
        -------
        float
            The equivalent decimal odds.  For example, +150 becomes 2.50 and
            -150 becomes 1.6667.
        """
        if american > 0:
            return 1 + (american / 100.0)
        elif american < 0:
            return 1 + (100.0 / abs(american))
        else:
            # Zero odds are undefined; return 0 to signal an issue
            return 0.0

    def analyse_prices_file(self, file_path: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Analyse a CSV file of moneyline prices to find potential value bets.

        This method reads a CSV file with columns corresponding to the
        structure of the MLB `prices.csv` dataset.  It then performs the
        following steps:

        1. Convert the American prices for each team to decimal odds and
           calculate their implied probabilities.
        2. For each game and team, compute the consensus probability by
           averaging the implied probabilities across all bookmakers.
        3. Treat the consensus probability as the bettor's estimate and
           compute the EV and Kelly fraction for each bookmaker's price.
        4. Filter for opportunities with positive expected value and return
           the top `top_n` picks sorted by descending EV.

        Parameters
        ----------
        file_path : str
            Path to the CSV file to analyse.
        top_n : int, optional
            Number of top opportunities to return.  Defaults to 10.

        Returns
        -------
        list of dict
            A list of dictionaries describing each positive EV opportunity.
        """
        import pandas as pd
        # Load the dataset
        df = pd.read_csv(file_path)
        # Guard against unexpected column names
        required_cols = {"game_id", "bookmaker", "team_name_1", "team_1_price", "team_name_2", "team_2_price"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"The prices file is missing required columns. Expected at least {required_cols}, "
                f"found {set(df.columns)}"
            )
        # Melt the dataset so each row represents a single team price
        team1 = df[["game_id", "commence_time_pst", "bookmaker", "team_name_1", "team_1_price"]].copy()
        team1 = team1.rename(columns={"team_name_1": "team_name", "team_1_price": "price"})
        team2 = df[["game_id", "commence_time_pst", "bookmaker", "team_name_2", "team_2_price"]].copy()
        team2 = team2.rename(columns={"team_name_2": "team_name", "team_2_price": "price"})
        prices = pd.concat([team1, team2], ignore_index=True)
        # Convert price to decimal odds and implied probability
        prices["decimal_odds"] = prices["price"].apply(lambda x: self.american_to_decimal(float(x)))
        # Remove rows with zero or invalid odds
        prices = prices[prices["decimal_odds"] > 1.0].copy()
        prices["implied_prob"] = 1.0 / prices["decimal_odds"]
        # Compute consensus probability per game and team by averaging implied probabilities
        consensus = (
            prices.groupby(["game_id", "team_name"])["implied_prob"]
            .mean()
            .reset_index()
            .rename(columns={"implied_prob": "consensus_prob"})
        )
        # Merge consensus back into prices
        prices = prices.merge(consensus, on=["game_id", "team_name"], how="left")
        # Analyse each row with positive EV
        opportunities = []
        for _, row in prices.iterrows():
            prob = row["consensus_prob"]
            odds = row["decimal_odds"]
            pick = Pick(event=f"{row['team_name']} vs Opponent", probability=prob, odds=odds)
            result = self.analyse_pick(pick)
            # Only consider positive EV opportunities
            if result["expected_value"] > 0:
                result.update(
                    {
                        "game_id": row["game_id"],
                        "bookmaker": row["bookmaker"],
                        "team": row["team_name"],
                        "american_odds": row["price"],
                        "commence_time": row["commence_time_pst"],
                    }
                )
                opportunities.append(result)
        # Sort by expected value descending and return top N
        opportunities.sort(key=lambda x: x["expected_value"], reverse=True)
        return opportunities[:top_n]

    # ---------------------------------------------------------------------
    # Telegram notification support
    #
    def send_telegram_notification(self, message: str) -> bool:
        """Send a notification message via Telegram.

        This method uses the Telegram Bot API to send a text message to
        the configured chat.  Both a bot token and a chat ID must be
        provided either during initialisation or via environment
        variables (`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`).

        Parameters
        ----------
        message : str
            The text message to send.

        Returns
        -------
        bool
            ``True`` if the message was sent successfully, ``False`` otherwise.
        """
        if not self.telegram_token or not self.telegram_chat_id:
            raise ValueError(
                "Telegram token or chat ID not set. Provide them to the constructor or as environment variables."
            )
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {"chat_id": self.telegram_chat_id, "text": message}
        try:
            response = requests.post(url, data=payload, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    # ---------------------------------------------------------------------
    # Upcoming game predictions and evaluation
    #
    # In addition to analysing historical datasets, the LineSniperAgent can
    # evaluate lines on future games.  The method below accepts a list of
    # dictionaries describing upcoming games with point spreads and totals.
    # The agent predicts the probability that the favoured side covers
    # using either the trained NFL model or a simple logistic transformation
    # and returns only those with positive expected value relative to –110
    # pricing.  Each prediction is logged for later evaluation.

    def predict_upcoming_spreads(
        self,
        lines: List[Dict[str, Any]],
        use_model: bool = True,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Predict and rank upcoming point‑spread opportunities.

        Parameters
        ----------
        lines : list of dict
            Each dictionary should describe a game with the following keys:

            ``home_team`` (str)
                Name of the home team.
            ``away_team`` (str)
                Name of the away team.
            ``spread_favorite`` (float)
                The point spread quoted for the favourite.  Negative
                numbers indicate the home team is favoured; positive numbers
                indicate the away team is favoured.
            ``over_under`` (float), optional
                Total points line for the game.  Required when
                ``use_model`` is True so the NFL model can use it as a
                predictor.  When absent or when ``use_model`` is False the
                method falls back to a logistic transformation of the spread.
            ``event_date`` (str), optional
                Human‑readable date/time of the game.  Used only for
                record keeping.

        use_model : bool, optional
            If True and a trained NFL model is available, use it to
            estimate cover probabilities.  Otherwise, apply a logistic
            transformation of the spread.  Defaults to True.
        top_n : int, optional
            Maximum number of positive EV opportunities to return.  Defaults
            to 10.

        Returns
        -------
        list of dict
            A list of positive EV picks sorted by expected value.  Each
            dictionary contains the event description, predicted probability,
            expected value, favourite team, spread and an optional
            pre‑formatted Telegram notification URL.  An empty list is
            returned if no picks have positive EV.
        """
        import math
        import urllib.parse

        # Decimal odds for –110 lines
        dec_odds = 1.0 + 100.0 / 110.0
        win_profit = dec_odds - 1.0

        picks: List[Dict[str, Any]] = []
        for game in lines:
            try:
                spread = float(game["spread_favorite"])
            except Exception:
                # Skip games without a valid spread
                continue
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            if not home_team or not away_team:
                continue
            # Determine which side is favoured by the spread
            # Negative spread → home is favoured; positive → away is favoured
            if spread < 0:
                fav_team = home_team
                opp_team = away_team
            elif spread > 0:
                fav_team = away_team
                opp_team = home_team
            else:
                # Pick‑em games: treat home team as favourite
                fav_team = home_team
                opp_team = away_team

            # Estimate probability the favourite covers
            if use_model and self.nfl_model is not None:
                # Use trained model.  Require over/under line for the feature
                ou = game.get("over_under")
                if ou is None:
                    # Fall back to logistic transformation when over/under is missing
                    use_model = False
                else:
                    try:
                        ou_val = float(ou)
                        prob_cover = float(
                            self.nfl_model.predict_proba([[spread, ou_val]])[0, 1]
                        )
                    except Exception:
                        # If prediction fails, fall back
                        use_model = False
            if not use_model:
                # Logistic transformation heuristic: tune parameter k to 0.18 like in
                # our college football method.  Use the negative of the spread
                # because negative spreads favour the home team.
                k = 0.18
                prob_cover = 1.0 / (1.0 + math.exp(-k * (-spread)))

            # Skip unrealistic probabilities
            if prob_cover <= 0.01 or prob_cover >= 0.99:
                continue

            # Compute expected value per unit stake for a –110 wager
            ev = (prob_cover * win_profit) - ((1.0 - prob_cover) * 1.0)
            if ev <= 0:
                continue

            event = f"{fav_team} vs {opp_team} (spread)"
            # Create a personality‑rich notification message
            message = self._format_notification_message(event, prob_cover, ev)
            # Build notification URL if Telegram credentials are set
            url = None
            if self.telegram_token and self.telegram_chat_id:
                # Use quote rather than quote_plus to percent‑encode all
                # special characters.  Telegram appears to reject messages
                # containing unencoded punctuation or plus signs, so a full
                # percent encoding is safer.
                encoded = urllib.parse.quote(message, safe="")
                url = (
                    f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    f"?chat_id={self.telegram_chat_id}&text={encoded}"
                )
                # Attempt to push the notification immediately.  In
                # environments with outbound network access this will send
                # the message to the configured Telegram chat.  Failures are
                # silently ignored to avoid interrupting the scanning loop.
                try:
                    # Use requests.post rather than GET to conform to
                    # Telegram API best practices.  Pass both chat_id and
                    # text in the data payload to avoid length limits on
                    # query strings.
                    import requests
                    payload = {"chat_id": self.telegram_chat_id, "text": message}
                    # Use a short timeout to prevent hanging if the network
                    # is unreachable.  Ignore the response status; if it
                    # fails, the user can still send the URL manually.
                    requests.post(
                        f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                        data=payload,
                        timeout=5,
                    )
                except Exception:
                    pass
            pick_record = {
                "event": event,
                "probability": prob_cover,
                "expected_value": ev,
                "favorite_team": fav_team,
                "opponent": opp_team,
                "spread": spread,
                "model_used": use_model,
                "message": message,
                "notification_url": url,
                "event_date": game.get("event_date"),
            }
            picks.append(pick_record)
            # Log the prediction for future evaluation
            self._prediction_log.append(pick_record)
        # Sort by expected value and return top N
        picks.sort(key=lambda x: x["expected_value"], reverse=True)
        return picks[:top_n]

    def evaluate_prediction(self, prediction: Dict[str, Any], final_home_score: int, final_away_score: int) -> bool:
        """Evaluate whether a spread prediction was correct.

        Given a prediction dictionary produced by :meth:`predict_upcoming_spreads` and the
        final scores of the game, this method computes whether the favourite
        covered the spread.  It returns ``True`` if the prediction was
        successful and ``False`` otherwise.

        Parameters
        ----------
        prediction : dict
            A prediction dictionary returned by :meth:`predict_upcoming_spreads`.
        final_home_score : int
            The final score of the home team.
        final_away_score : int
            The final score of the away team.

        Returns
        -------
        bool
            True if the favourite covered, False otherwise.  In the event of
            a push, returns False (no win).
        """
        try:
            fav = prediction["favorite_team"]
            opp = prediction["opponent"]
            spread = float(prediction["spread"])
        except Exception:
            raise ValueError("Prediction dictionary is missing required keys.")

        # Determine which team was home and away by checking names
        # Note: because we do not store which team was home in the prediction,
        # this evaluation assumes the calling code knows which score belongs
        # to which team.  The final scores must correspond to the home and
        # away teams passed to :meth:`predict_upcoming_spreads`.
        # Compute margin of victory for the favourite
        # Negative spread means home is favoured, positive means away is favoured
        if spread < 0:
            # Home favoured
            margin = final_home_score - final_away_score
            # Favourite covers if margin > abs(spread)
            return margin > abs(spread)
        elif spread > 0:
            # Away favoured
            margin = final_away_score - final_home_score
            return margin > abs(spread)
        else:
            # Pick‑em: favourite covers if it wins outright
            return final_home_score != final_away_score and (
                (fav == prediction.get("home_team") and final_home_score > final_away_score) or
                (fav == prediction.get("away_team") and final_away_score > final_home_score)
            )


    # ---------------------------------------------------------------------
    # Spread‑based analysis for NFL and College Football
    #
    # Point spread wagers are commonly offered at around –110 American odds on
    # both sides, meaning a bettor risks $110 to win $100 regardless of which
    # team they choose.  Because the NFL dataset we bundle with this project
    # (`nfl_betting_df.csv`) contains point spreads and estimated win
    # probabilities rather than moneyline odds, the methods below
    # approximate EV calculations by assuming standard –110 pricing.  For
    # college football we apply a simple logistic transformation to the
    # spread to estimate the probability of the home team covering.  These
    # heuristics are meant for educational purposes only; they will not
    # perfectly reflect bookmaker models.

    def _decimal_from_american(self, american_odds: float) -> float:
        """Internal helper to convert American odds to decimal odds.

        This mirrors :meth:`american_to_decimal` but is reused internally.
        It accepts both integers and floats and returns 0 for invalid odds.
        """
        try:
            american = float(american_odds)
        except Exception:
            return 0.0
        return self.american_to_decimal(american)

    def analyse_nfl_spread_file(self, file_path: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Analyse an NFL point‑spread dataset for potential value bets.

        The supplied CSV is expected to follow the structure of
        `nfl_betting_df.csv` from this repository.  It should include
        columns like `team_home`, `team_away`, `spread_favorite`,
        `home_exp_win_pct` and `away_exp_win_pct`.  Because the file does
        not contain actual moneyline prices, this method assumes both
        sides of the spread are priced at –110 (decimal 1.9091).  The
        probability of the favoured side covering is taken directly from
        the provided `home_exp_win_pct` or `away_exp_win_pct` columns.
        
        Parameters
        ----------
        file_path : str
            Path to the NFL spread CSV.
        top_n : int, optional
            Maximum number of positive EV picks to return.  Defaults to 10.

        Returns
        -------
        list of dict
            A list of analysis dictionaries for the top positive EV
            opportunities.  If the file is missing required columns or
            yields no positive EV bets, an empty list is returned.
        """
        import pandas as pd
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read {file_path}: {e}") from e
        required = {"team_home", "team_away", "spread_favorite", "home_exp_win_pct", "away_exp_win_pct"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"NFL spread file missing required columns. Expected at least {required}, found {set(df.columns)}"
            )
        # Standard decimal odds for –110 point spread wagers
        dec_odds = self._decimal_from_american(-110)
        picks = []
        for _, row in df.iterrows():
            spread = row["spread_favorite"]
            home_prob = row["home_exp_win_pct"]
            away_prob = row["away_exp_win_pct"]
            # Determine which team is favoured: negative spread means home is favoured
            if pd.isna(spread) or pd.isna(home_prob) or pd.isna(away_prob):
                continue
            # Choose the side with greater expected win probability
            if home_prob > away_prob:
                team = row["team_home"]
                prob = float(home_prob)
            else:
                team = row["team_away"]
                prob = float(away_prob)
            # Skip nonsensical probabilities and extreme edges.  Realistic
            # probability estimates should lie strictly between 0 and 1 and
            # avoid values close to certainty.
            if prob <= 0 or prob >= 1 or prob <= 0.01 or prob >= 0.99:
                continue
            pick = Pick(event=f"{team} vs Opponent (spread)", probability=prob, odds=dec_odds)
            result = self.analyse_pick(pick)
            if result["expected_value"] > 0:
                result.update(
                    {
                        "team": team,
                        "spread": spread,
                        "probability_used": prob,
                    }
                )
                picks.append(result)
        picks.sort(key=lambda x: x["expected_value"], reverse=True)
        return picks[:top_n]

    def analyse_cfb_spread_file(self, file_path: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Analyse a college football point‑spread dataset for value bets.

        This method is intentionally flexible, as public college football
        spread datasets vary widely in structure.  It expects columns
        `HomeTeam`, `AwayTeam` and `Spread` where the spread is quoted
        from the perspective of the favourite (negative numbers indicate
        the home team is favoured).  Because most college games are
        priced at approximately –110, the odds are assumed to be 1.9091.
        
        Without explicit win probability estimates in the dataset, we
        estimate the probability of the home team covering using a simple
        logistic transformation of the spread:
            p_home = 1 / (1 + exp(-k * ( -spread )))
        where k = 0.18.  This function produces reasonable probabilities
        around typical college spreads (e.g. –7 → ~0.68, –14 → ~0.84).
        The underdog's probability is 1 − p_home.  We then compute EV
        assuming –110 odds on both sides and return picks where the EV is
        positive for the team with the higher estimated probability.
        
        Parameters
        ----------
        file_path : str
            Path to the college football spread CSV.
        top_n : int, optional
            Maximum number of positive EV picks to return.  Defaults to 10.

        Returns
        -------
        list of dict
            A list of analysis dictionaries for the top positive EV picks.
        """
        import pandas as pd
        import math
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read {file_path}: {e}") from e
        # Require minimal columns
        cols_needed = {"HomeTeam", "AwayTeam", "Spread"}
        if not cols_needed.issubset(df.columns):
            raise ValueError(
                f"College football spread file missing required columns {cols_needed}. Found {set(df.columns)}"
            )
        dec_odds = self._decimal_from_american(-110)
        k = 0.18  # logistic scale parameter; tuned heuristically
        opportunities = []
        for _, row in df.iterrows():
            try:
                spread = float(row["Spread"])
            except Exception:
                continue
            # Estimate probability the home team covers the spread
            # Negative spread indicates home team is favoured
            p_home = 1.0 / (1.0 + math.exp(-k * (-spread)))
            p_away = 1.0 - p_home
            home_team = str(row["HomeTeam"])
            away_team = str(row["AwayTeam"])
            # Choose the team with higher estimated probability
            if p_home >= p_away:
                team = home_team
                prob = p_home
            else:
                team = away_team
                prob = p_away
            # Skip extremes; ignore probabilities too close to 0 or 1.
            if prob <= 0 or prob >= 1 or prob <= 0.01 or prob >= 0.99:
                continue
            pick = Pick(event=f"{team} vs Opponent (CFB spread)", probability=prob, odds=dec_odds)
            result = self.analyse_pick(pick)
            if result["expected_value"] > 0:
                result.update(
                    {
                        "team": team,
                        "spread": spread,
                        "probability_used": prob,
                    }
                )
                opportunities.append(result)
        opportunities.sort(key=lambda x: x["expected_value"], reverse=True)
        return opportunities[:top_n]

    # ---------------------------------------------------------------------
    # Machine learning for NFL point‑spread predictions
    #
    # To make the agent "learn" from historical data and improve over time,
    # we provide a pair of methods to train a logistic regression model on
    # historical NFL point‑spread data and then scan that same dataset for
    # positive EV opportunities.  Logistic regression is a common technique for
    # predicting binary outcomes in sports modelling【916453135453690†L294-L323】.  Here, we use
    # it to estimate the probability that the favourite covers the spread based
    # on the spread itself and the over/under line.  Rows where the result
    # of the wager was a push (encoded as 2 in `favorite_covered`) are
    # discarded prior to training.

    def train_nfl_model(self, file_path: str) -> float:
        """Train a logistic regression model on historical NFL spread data.

        The CSV specified by ``file_path`` should match the structure of
        ``nfl_betting_df.csv`` included with this project.  It must contain
        columns ``spread_favorite``, ``over_under_line`` and ``favorite_covered``.
        The latter is expected to be 0 if the favourite failed to cover,
        1 if the favourite covered and 2 if the game resulted in a push.  Rows
        where ``favorite_covered`` equals 2 are excluded from the training set,
        and any rows with missing feature values are dropped.  The method
        trains a logistic regression model using scikit‑learn, stores it in
        ``self.nfl_model`` and returns the area under the ROC curve (AUC) on
        a 20 % holdout for reference.  A higher AUC indicates better ability
        to discriminate between covers and non‑covers.

        Parameters
        ----------
        file_path : str
            Path to the NFL betting CSV.

        Returns
        -------
        float
            The AUC of the trained model on a holdout set.  If evaluation
            cannot be performed (e.g. because the holdout contains only a
            single class), the method returns ``float('nan')``.
        """
        import pandas as pd  # local import to avoid heavy dependency at module load
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        # Load and clean the dataset
        df = pd.read_csv(file_path)
        required_cols = {"spread_favorite", "over_under_line", "favorite_covered"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"NFL betting file missing required columns {required_cols}. Found {set(df.columns)}"
            )
        # Drop pushes (favorite_covered == 2) and rows with missing data
        df = df[(df["favorite_covered"].isin([0, 1])) & df["spread_favorite"].notna() & df["over_under_line"].notna()]
        if df.empty:
            raise ValueError("No valid rows remaining after filtering pushes and missing data.")
        X = df[["spread_favorite", "over_under_line"]].astype(float)
        y = df["favorite_covered"].astype(int)
        # Split off a portion of the data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        # Try to compute AUC; if the test set lacks both classes, return NaN
        auc = float('nan')
        try:
            proba = model.predict_proba(X_test)[:, 1]
            if len(set(y_test)) > 1:
                auc = roc_auc_score(y_test, proba)
        except Exception:
            pass
        # Store the trained model on the agent
        self.nfl_model = model
        return auc

    def scan_nfl_ml(self, file_path: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Use the trained NFL model to find positive EV point‑spread bets.

        This method assumes that ``self.nfl_model`` has been trained by
        :meth:`train_nfl_model`.  It reads the same data file, iterates over
        each row and uses the model to predict the probability that the
        favourite covers.  If that probability exceeds the break‑even
        threshold implied by –110 odds (approx. 0.524), the method
        calculates the expected value per unit stake and, if positive,
        constructs a pick dictionary.  Each dictionary contains the event
        description (favourite versus opponent), the predicted probability,
        the expected value, and a pre‑formatted Telegram message string that
        can be sent via :meth:`send_telegram_notification`.

        Parameters
        ----------
        file_path : str
            Path to the NFL betting CSV.
        top_n : int, optional
            Maximum number of positive EV picks to return.  Defaults to 10.

        Returns
        -------
        list of dict
            A list of dictionaries for the top positive EV opportunities.  If
            no positive EV picks are found, an empty list is returned.
        """
        if self.nfl_model is None:
            raise ValueError("NFL model has not been trained. Call train_nfl_model first.")
        import pandas as pd  # local import
        import urllib.parse
        # Load the full dataset (do not drop pushes; they can still produce picks)
        df = pd.read_csv(file_path)
        # Compute constant values for EV
        decimal_odds = 1.0 + 100.0 / 110.0
        win_profit = decimal_odds - 1.0
        picks: List[Dict[str, Any]] = []
        import math
        for _, row in df.iterrows():
            # Require both features to make a prediction.  Convert to float and
            # discard rows containing NaN values, as scikit‑learn cannot
            # handle missing values.
            try:
                spread = float(row["spread_favorite"])
                total = float(row["over_under_line"])
            except Exception:
                continue
            # Skip if either value is NaN
            if math.isnan(spread) or math.isnan(total):
                continue
            # Predict probability that favourite covers
            prob_cover = float(self.nfl_model.predict_proba([[spread, total]])[0, 1])
            # Skip if probability does not exceed break‑even
            if prob_cover <= self.nfl_break_even:
                continue
            # Compute expected value (per unit stake) under –110 pricing
            ev = (prob_cover * win_profit) - ((1.0 - prob_cover) * 1.0)
            if ev <= 0:
                continue
            # Determine the favourite team; if unavailable use home team
            fav = row.get("team_favorite_id")
            home = row.get("team_home")
            away = row.get("team_away")
            # Some entries label a pick‑em game as 'PICK'; use the home team in that case
            if not fav or pd.isna(fav) or str(fav).strip().upper() == "PICK":
                fav_team = home
            else:
                fav_team = fav
            # Determine the opponent
            if fav_team == home:
                opponent = away
            else:
                opponent = home
            event = f"{fav_team} vs {opponent} (spread)"
            # Build a user‑friendly Telegram message
            # Use personality‑rich formatting for the notification message
            message = self._format_notification_message(event, prob_cover, ev)
            # If Telegram credentials are available, include the encoded URL
            url = None
            if self.telegram_token and self.telegram_chat_id:
                # Percent‑encode the message to build a GET URL as a fallback.
                encoded = urllib.parse.quote(message, safe="")
                url = (
                    f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    f"?chat_id={self.telegram_chat_id}&text={encoded}"
                )
                # Attempt to send the notification immediately via POST.
                try:
                    import requests
                    payload = {"chat_id": self.telegram_chat_id, "text": message}
                    requests.post(
                        f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                        data=payload,
                        timeout=5,
                    )
                except Exception:
                    pass
            picks.append(
                {
                    "event": event,
                    "probability": prob_cover,
                    "expected_value": ev,
                    "message": message,
                    "notification_url": url,
                }
            )
        # Sort by expected value descending and return top N picks
        picks.sort(key=lambda x: x["expected_value"], reverse=True)
        return picks[:top_n]


def _demo() -> None:
    """Run a simple demonstration when executed as a script.

    This function defines a small set of hypothetical betting opportunities,
    analyses them using a LineSniperAgent and prints a report to stdout.
    """
    print("LineSniperAgent demonstration:\n")
    agent = LineSniperAgent(bankroll=1000.0)
    picks_data = [
        {"event": "Eagles vs Giants (moneyline on Eagles)", "probability": 0.60, "odds": 1.90},
        {"event": "Yankees vs Red Sox (run line -1.5)", "probability": 0.45, "odds": 2.40},
        {"event": "Lakers vs Bulls (over 215.5 total)", "probability": 0.52, "odds": 1.95},
        {"event": "Packers vs Bears (moneyline on Bears)", "probability": 0.35, "odds": 3.20},
    ]
    # Analyse the hypothetical picks using our core methods
    opportunities = agent.find_positive_ev_picks(picks_data)
    if opportunities:
        # Print results in a simple table
        header = (
            f"{'Event':40s}  {'Prob':>5s}  {'Odds':>6s}  {'ImpProb':>7s}  {'EV':>6s}  {'Kelly':>5s}  {'Bet':>7s}"
        )
        print(header)
        print("-" * len(header))
        for opp in opportunities:
            print(
                f"{opp['event'][:40]:40s}  {opp['probability']:.2f}  {opp['odds']:.2f}  {opp['implied_probability']:.2f}  "
                f"{opp['expected_value']:.3f}  {opp['kelly_fraction']:.2f}  ${opp['recommended_bet']:.2f}"
            )
    else:
        print("No positive EV opportunities were found in the synthetic dataset.")

    # Attempt to analyse a real dataset if available
    import os
    real_file = os.path.join(os.path.dirname(__file__), "prices.csv")
    if os.path.exists(real_file):
        try:
            print("\nReal dataset analysis (top 5 opportunities):")
            real_opps = agent.analyse_prices_file(real_file, top_n=5)
            if real_opps:
                header = (
                    f"{'Team':25s}  {'Bookmaker':20s}  {'EV':>6s}  {'Kelly':>5s}  {'Bet':>7s}"
                )
                print(header)
                print("-" * len(header))
                for opp in real_opps:
                    print(
                        f"{opp['team'][:25]:25s}  {opp['bookmaker'][:20]:20s}  {opp['expected_value']:.3f}  "
                        f"{opp['kelly_fraction']:.2f}  ${opp['recommended_bet']:.2f}"
                    )
            else:
                print("No positive EV opportunities found in the real dataset.")
        except Exception as e:
            print(f"Error analysing real dataset: {e}")


if __name__ == "__main__":
    _demo()