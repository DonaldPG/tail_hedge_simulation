"""
Behavioral Utility Module

Implements utility functions based on behavioral finance principles,
such as Prospect Theory by Kahneman and Tversky. These functions
are used to evaluate portfolio returns in a way that reflects
asymmetric risk preferences (loss aversion).
"""

import numpy as np
from typing import Union

def prospect_theory_utility(
    returns: Union[np.ndarray, float],
    loss_aversion: float = 2.25,
    risk_aversion: float = 0.88
) -> Union[np.ndarray, float]:
    """
    Calculate the utility of returns based on Prospect Theory.

    This function models two key concepts:
    1.  Loss Aversion: Losses feel more painful than equivalent gains feel good.
        This is controlled by the `loss_aversion` parameter (lambda).
    2.  Diminishing Sensitivity: The impact of a change in wealth diminishes
        as wealth moves further from the reference point (0). This is
        controlled by the `risk_aversion` parameter (alpha).

    The formula is a simplified version of the value function from Prospect Theory:
    - For gains (x >= 0): U(x) = x^alpha
    - For losses (x < 0):  U(x) = -lambda * (-x)^alpha

    Args:
        returns: A numpy array or single float of portfolio returns.
        loss_aversion: The multiplier for the pain of losses (lambda).
                       A value of 2.25 means a loss is 2.25x as painful
                       as an equivalent gain.
        risk_aversion: The exponent for risk aversion (alpha). Values < 1
                       indicate risk-averse behavior in the domain of gains
                       and risk-seeking in the domain of losses.

    Returns:
        The calculated utility values for the given returns.
    """
    # Separate returns into gains and losses
    gains = returns.copy()
    losses = returns.copy()
    
    gains[gains < 0] = 0
    losses[losses >= 0] = 0

    # Calculate utility for gains
    utility_gains = gains ** risk_aversion
    
    # Calculate utility for losses
    utility_losses = -loss_aversion * ((-losses) ** risk_aversion)

    return utility_gains + utility_losses

def behavioral_sharpe_ratio(
    returns: np.ndarray,
    loss_aversion: float = 2.25,
    risk_aversion: float = 0.88,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate a Sharpe ratio adjusted for behavioral utility.

    Instead of using the standard deviation of returns as the measure of risk,
    this function uses the standard deviation of the *utility* of returns.
    This penalizes downside volatility more heavily, in line with behavioral
    principles.

    Args:
        returns: Array of portfolio returns.
        loss_aversion: Loss aversion parameter for the utility function.
        risk_aversion: Risk aversion parameter for the utility function.
        risk_free_rate: The risk-free rate of return.

    Returns:
        The behavioral-adjusted Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate
    
    # Calculate the utility of excess returns
    utility_of_returns = prospect_theory_utility(
        excess_returns,
        loss_aversion,
        risk_aversion
    )
    
    # The "reward" is the average utility
    mean_utility = np.mean(utility_of_returns)
    
    # The "risk" is the volatility of that utility
    utility_volatility = np.std(utility_of_returns)
    
    if utility_volatility == 0:
        return np.inf if mean_utility > 0 else 0
        
    return mean_utility / utility_volatility