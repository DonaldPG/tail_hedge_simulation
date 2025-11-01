"""
Systematic Hedging Module

Implements the Deutsche Bank option-based systematic hedging framework with 
asymmetric volatility considerations:
- Parametric purchasing programs for protective puts
- Economic efficiency optimization 
- Risk-adjusted vs return-adjusted perspectives
- Continuous hedging with predictable costs
- Asymmetric volatility modeling (Kahneman 2:1 loss aversion)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import logging
from scipy.stats import norm
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HedgeType(Enum):
    """Types of systematic hedging strategies"""
    PUT_PROTECTION = "put_protection"
    VIX_CALLS = "vix_calls"
    COLLAR = "collar"
    CPPI = "cppi"  # Constant Proportion Portfolio Insurance


class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    LOW_VOL = "low_volatility"
    NORMAL_VOL = "normal_volatility"
    HIGH_VOL = "high_volatility"
    CRISIS = "crisis"


@dataclass
class OptionParameters:
    """
    Parameters for option-based hedging
    
    Based on Deutsche Bank's parametric purchasing program approach
    """
    moneyness: float  # Strike relative to spot (e.g., 0.95 for 5% OTM puts)
    time_to_expiry: int  # Days to expiration
    coverage_ratio: float  # Proportion of portfolio to hedge (0-1)
    roll_frequency: int  # Days between rolling positions
    delta_threshold: float  # Delta level to trigger rebalancing


@dataclass
class AsymmetricVolatilityModel:
    """
    Asymmetric volatility model incorporating Kahneman's loss aversion
    
    Models the empirical observation that downside moves are ~2x more volatile
    than upside moves, reflecting behavioral biases in market pricing.
    """
    base_volatility: float = 0.20
    downside_multiplier: float = 2.0  # Kahneman's 2:1 pain/pleasure ratio
    upside_multiplier: float = 0.7  # Reduced upside volatility
    regime_threshold: float = 0.02  # Return threshold for regime switching
    volatility_persistence: float = 0.9  # Volatility clustering parameter
    
    def get_effective_volatility(self, return_level: float, current_vol: float) -> float:
        """
        Calculate effective volatility based on return direction and magnitude
        
        Args:
            return_level: Market return level (e.g., -0.05 for -5%)
            current_vol: Current base volatility
            
        Returns:
            Effective volatility adjusted for asymmetry
        """
        if return_level < -self.regime_threshold:
            # Downside regime - higher volatility (Kahneman effect)
            multiplier = self.downside_multiplier
        elif return_level > self.regime_threshold:
            # Upside regime - lower volatility
            multiplier = self.upside_multiplier
        else:
            # Normal regime
            multiplier = 1.0
        
        # Apply volatility clustering
        adjusted_vol = current_vol * multiplier
        return self.volatility_persistence * adjusted_vol + (1 - self.volatility_persistence) * self.base_volatility


@dataclass
class MarketModel:
    """
    Market model parameters for option pricing and hedging
    """
    risk_free_rate: float = 0.02
    dividend_yield: float = 0.015
    volatility: float = 0.20
    vol_of_vol: float = 0.8  # For stochastic volatility models
    mean_reversion: float = 2.0  # Volatility mean reversion speed
    long_term_vol: float = 0.18  # Long-term volatility level
    asymmetric_vol_model: Optional[AsymmetricVolatilityModel] = None
    
    def __post_init__(self):
        """Initialize asymmetric volatility model if not provided"""
        if self.asymmetric_vol_model is None:
            self.asymmetric_vol_model = AsymmetricVolatilityModel(base_volatility=self.volatility)


class AsymmetricBlackScholesModel:
    """
    Enhanced Black-Scholes model incorporating asymmetric volatility effects
    
    This addresses the core research question: how does asymmetric volatility
    affect option pricing when BS assumes symmetric volatility?
    """
    
    def __init__(self, market_model: MarketModel):
        """
        Initialize Asymmetric Black-Scholes model
        
        Args:
            market_model: Market parameters including asymmetric volatility
        """
        self.market_model = market_model
        self.asymmetric_vol = market_model.asymmetric_vol_model
    
    def option_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        option_type: str = "put",
        use_asymmetric_vol: bool = True,
        market_return: Optional[float] = None
    ) -> float:
        """
        Calculate option price with asymmetric volatility considerations
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            option_type: "call" or "put"
            use_asymmetric_vol: Whether to use asymmetric volatility model
            market_return: Recent market return for volatility regime detection
            
        Returns:
            Option price incorporating asymmetric volatility effects
        """
        if not use_asymmetric_vol or market_return is None:
            volatility = self.market_model.volatility
        else:
            # Use asymmetric volatility based on market regime
            volatility = self.asymmetric_vol.get_effective_volatility(
                market_return, self.market_model.volatility
            )
        
        # Handle edge cases
        if time_to_expiry <= 0:
            if option_type == "call":
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        r = self.market_model.risk_free_rate
        q = self.market_model.dividend_yield
        
        d1 = (np.log(spot / strike) + (r - q + 0.5 * volatility**2) * time_to_expiry) / (
            volatility * np.sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if option_type == "call":
            price = (spot * np.exp(-q * time_to_expiry) * norm.cdf(d1) - 
                    strike * np.exp(-r * time_to_expiry) * norm.cdf(d2))
        else:  # put
            price = (strike * np.exp(-r * time_to_expiry) * norm.cdf(-d2) - 
                    spot * np.exp(-q * time_to_expiry) * norm.cdf(-d1))
        
        return price
    
    def calculate_mispricing_effect(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        option_type: str = "put",
        market_return: float = -0.05
    ) -> Dict[str, float]:
        """
        Calculate the mispricing effect between BS and asymmetric volatility
        
        This directly addresses your research question about how asymmetric
        volatility affects option payoffs when priced with symmetric BS models.
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            option_type: "call" or "put"
            market_return: Recent market return for volatility regime
            
        Returns:
            Dictionary with pricing comparison metrics
        """
        # Price with symmetric Black-Scholes
        bs_price = self.option_price(
            spot, strike, time_to_expiry, option_type, 
            use_asymmetric_vol=False
        )
        
        # Price with asymmetric volatility
        asymmetric_price = self.option_price(
            spot, strike, time_to_expiry, option_type,
            use_asymmetric_vol=True, market_return=market_return
        )
        
        # Calculate mispricing metrics
        absolute_mispricing = asymmetric_price - bs_price
        relative_mispricing = absolute_mispricing / bs_price if bs_price > 0 else 0
        
        # Get effective volatilities for analysis
        base_vol = self.market_model.volatility
        effective_vol = self.asymmetric_vol.get_effective_volatility(market_return, base_vol)
        vol_adjustment = effective_vol / base_vol
        
        return {
            "bs_price": bs_price,
            "asymmetric_price": asymmetric_price,
            "absolute_mispricing": absolute_mispricing,
            "relative_mispricing": relative_mispricing,
            "base_volatility": base_vol,
            "effective_volatility": effective_vol,
            "volatility_adjustment": vol_adjustment,
            "market_return": market_return
        }
    
    def delta(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        option_type: str = "put",
        use_asymmetric_vol: bool = True,
        market_return: Optional[float] = None
    ) -> float:
        """
        Calculate option delta with asymmetric volatility
        """
        if not use_asymmetric_vol or market_return is None:
            volatility = self.market_model.volatility
        else:
            volatility = self.asymmetric_vol.get_effective_volatility(
                market_return, self.market_model.volatility
            )
        
        if time_to_expiry <= 0:
            return 0.0
        
        q = self.market_model.dividend_yield
        
        d1 = (np.log(spot / strike) + (self.market_model.risk_free_rate - q + 0.5 * volatility**2) * time_to_expiry) / (
            volatility * np.sqrt(time_to_expiry)
        )
        
        if option_type == "call":
            return np.exp(-q * time_to_expiry) * norm.cdf(d1)
        else:  # put
            return -np.exp(-q * time_to_expiry) * norm.cdf(-d1)


class SystematicHedging:
    """
    Enhanced systematic hedging strategy implementing Deutsche Bank's framework
    with asymmetric volatility considerations
    
    Features:
    - Rule-based option purchasing
    - Economic efficiency optimization
    - Risk budget management
    - Continuous protection with predictable costs
    - Asymmetric volatility-aware pricing and hedging
    """
    
    def __init__(
        self,
        hedge_type: HedgeType,
        option_params: OptionParameters,
        market_model: MarketModel,
        portfolio_value: float = 1000000,
        use_asymmetric_pricing: bool = True
    ):
        """
        Initialize systematic hedging strategy
        
        Args:
            hedge_type: Type of hedging strategy
            option_params: Option parameters for hedging
            market_model: Market model for pricing
            portfolio_value: Initial portfolio value
            use_asymmetric_pricing: Whether to use asymmetric volatility in pricing
        """
        self.hedge_type = hedge_type
        self.option_params = option_params
        self.market_model = market_model
        self.portfolio_value = portfolio_value
        self.use_asymmetric_pricing = use_asymmetric_pricing
        
        self.pricing_model = AsymmetricBlackScholesModel(market_model)
        self.hedge_positions = []
        self.performance_history = []
        self.hedging_costs = []
        self.mispricing_history = []  # Track mispricing effects
        
        logger.info(f"Initialized {hedge_type.value} hedging with {option_params.coverage_ratio:.1%} coverage")
        logger.info(f"Asymmetric pricing: {'Enabled' if use_asymmetric_pricing else 'Disabled'}")
    
    def calculate_hedge_cost_with_asymmetry(
        self,
        spot_price: float,
        portfolio_allocation: Dict[str, float],
        market_return: float = 0.0,
        volatility_regime: Optional[VolatilityRegime] = None
    ) -> Dict[str, float]:
        """
        Calculate hedging cost considering asymmetric volatility effects
        
        This method demonstrates how asymmetric volatility affects hedge costs
        compared to symmetric Black-Scholes assumptions.
        
        Args:
            spot_price: Current underlying price
            portfolio_allocation: Current portfolio weights
            market_return: Recent market return for volatility regime detection
            volatility_regime: Optional explicit volatility regime
            
        Returns:
            Dictionary with cost analysis including mispricing effects
        """
        equity_exposure = portfolio_allocation.get("equity", 0.6)
        hedge_notional = self.portfolio_value * equity_exposure * self.option_params.coverage_ratio
        
        # Calculate costs with both symmetric and asymmetric models
        results = {}
        
        if self.hedge_type == HedgeType.PUT_PROTECTION:
            results = self._put_protection_cost_analysis(
                spot_price, hedge_notional, market_return
            )
        elif self.hedge_type == HedgeType.VIX_CALLS:
            results = self._vix_calls_cost_analysis(
                spot_price, hedge_notional, market_return
            )
        elif self.hedge_type == HedgeType.COLLAR:
            results = self._collar_cost_analysis(
                spot_price, hedge_notional, market_return
            )
        
        # Store mispricing data for analysis
        self.mispricing_history.append(results)
        
        return results
    
    def _put_protection_cost_analysis(
        self,
        spot_price: float,
        hedge_notional: float,
        market_return: float
    ) -> Dict[str, float]:
        """Detailed cost analysis for put protection with asymmetric volatility"""
        strike_price = spot_price * self.option_params.moneyness
        time_to_expiry = self.option_params.time_to_expiry / 365.25
        
        # Calculate mispricing effect
        mispricing_analysis = self.pricing_model.calculate_mispricing_effect(
            spot_price, strike_price, time_to_expiry, "put", market_return
        )
        
        num_contracts = hedge_notional / spot_price
        
        # Costs based on different pricing models
        bs_total_cost = num_contracts * mispricing_analysis["bs_price"]
        asymmetric_total_cost = num_contracts * mispricing_analysis["asymmetric_price"]
        
        return {
            "bs_cost_pct": bs_total_cost / self.portfolio_value,
            "asymmetric_cost_pct": asymmetric_total_cost / self.portfolio_value,
            "cost_difference": (asymmetric_total_cost - bs_total_cost) / self.portfolio_value,
            "relative_mispricing": mispricing_analysis["relative_mispricing"],
            "effective_volatility": mispricing_analysis["effective_volatility"],
            "market_regime": "downside" if market_return < -0.02 else "upside" if market_return > 0.02 else "normal",
            "hedge_effectiveness_multiplier": mispricing_analysis["volatility_adjustment"]
        }
    
    def _vix_calls_cost_analysis(
        self,
        spot_price: float,
        hedge_notional: float,
        market_return: float
    ) -> Dict[str, float]:
        """Cost analysis for VIX calls with volatility regime awareness"""
        base_cost = 0.02 * self.option_params.coverage_ratio
        
        # VIX calls become more valuable in high volatility regimes
        if market_return < -0.05:  # Stress regime
            vol_adjustment = 1.5  # VIX calls more valuable
        elif market_return < -0.02:  # Mild stress
            vol_adjustment = 1.2
        else:
            vol_adjustment = 1.0
        
        asymmetric_cost = base_cost * vol_adjustment
        
        return {
            "bs_cost_pct": base_cost,
            "asymmetric_cost_pct": asymmetric_cost,
            "cost_difference": asymmetric_cost - base_cost,
            "relative_mispricing": (asymmetric_cost - base_cost) / base_cost if base_cost > 0 else 0,
            "market_regime": "stress" if market_return < -0.05 else "normal",
            "hedge_effectiveness_multiplier": vol_adjustment
        }
    
    def _collar_cost_analysis(
        self,
        spot_price: float,
        hedge_notional: float,
        market_return: float
    ) -> Dict[str, float]:
        """Cost analysis for collar strategies with asymmetric volatility"""
        time_to_expiry = self.option_params.time_to_expiry / 365.25
        
        # Put leg analysis
        put_strike = spot_price * self.option_params.moneyness
        put_analysis = self.pricing_model.calculate_mispricing_effect(
            spot_price, put_strike, time_to_expiry, "put", market_return
        )
        
        # Call leg analysis (we're selling calls)
        call_strike = spot_price * 1.1
        call_analysis = self.pricing_model.calculate_mispricing_effect(
            spot_price, call_strike, time_to_expiry, "call", market_return
        )
        
        # Net cost calculation
        bs_net_cost = put_analysis["bs_price"] - call_analysis["bs_price"]
        asymmetric_net_cost = put_analysis["asymmetric_price"] - call_analysis["asymmetric_price"]
        
        num_contracts = hedge_notional / spot_price
        
        return {
            "bs_cost_pct": max(0, bs_net_cost * num_contracts / self.portfolio_value),
            "asymmetric_cost_pct": max(0, asymmetric_net_cost * num_contracts / self.portfolio_value),
            "cost_difference": (asymmetric_net_cost - bs_net_cost) * num_contracts / self.portfolio_value,
            "put_mispricing": put_analysis["relative_mispricing"],
            "call_mispricing": call_analysis["relative_mispricing"],
            "market_regime": "downside" if market_return < -0.02 else "upside" if market_return > 0.02 else "normal"
        }
    
    def analyze_kahneman_hedge_effect(
        self,
        price_scenarios: np.ndarray,
        return_scenarios: np.ndarray,
        scenario_probabilities: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze hedge effectiveness under Kahneman's 2:1 loss aversion framework
        
        This method quantifies how the asymmetric volatility (reflecting behavioral
        biases) affects hedge performance compared to symmetric assumptions.
        
        Args:
            price_scenarios: Array of price scenarios
            return_scenarios: Corresponding return scenarios
            scenario_probabilities: Probability weights
            
        Returns:
            Analysis of hedge effectiveness under asymmetric volatility
        """
        initial_price = price_scenarios[0]
        results = {
            "scenarios": len(price_scenarios),
            "downside_scenarios": np.sum(return_scenarios < 0),
            "upside_scenarios": np.sum(return_scenarios > 0),
            "bs_expected_cost": 0,
            "asymmetric_expected_cost": 0,
            "bs_expected_payoff": 0,
            "asymmetric_expected_payoff": 0,
            "kahneman_advantage": 0
        }
        
        for price, ret, prob in zip(price_scenarios, return_scenarios, scenario_probabilities):
            # Calculate costs under both models
            portfolio_allocation = {"equity": 0.6, "bonds": 0.4}
            
            cost_analysis = self.calculate_hedge_cost_with_asymmetry(
                initial_price, portfolio_allocation, ret
            )
            
            # Calculate payoffs
            bs_payoff = self.calculate_hedge_payoff(
                initial_price, price, ret, portfolio_allocation, use_asymmetric_vol=False
            )
            asymmetric_payoff = self.calculate_hedge_payoff(
                initial_price, price, ret, portfolio_allocation, use_asymmetric_vol=True
            )
            
            # Weight by probability
            results["bs_expected_cost"] += prob * cost_analysis["bs_cost_pct"]
            results["asymmetric_expected_cost"] += prob * cost_analysis["asymmetric_cost_pct"]
            results["bs_expected_payoff"] += prob * bs_payoff
            results["asymmetric_expected_payoff"] += prob * asymmetric_payoff
        
        # Calculate Kahneman advantage (asymmetric model accounts for behavioral biases)
        results["kahneman_advantage"] = (
            (results["asymmetric_expected_payoff"] - results["asymmetric_expected_cost"]) -
            (results["bs_expected_payoff"] - results["bs_expected_cost"])
        )
        
        # Risk-adjusted metrics
        results["cost_adjusted_efficiency"] = (
            results["asymmetric_expected_payoff"] / results["asymmetric_expected_cost"]
            if results["asymmetric_expected_cost"] > 0 else 0
        )
        
        return results
    
    def calculate_hedge_payoff(
        self,
        initial_price: float,
        final_price: float,
        portfolio_return: float,
        portfolio_allocation: Dict[str, float],
        use_asymmetric_vol: bool = None
    ) -> float:
        """
        Calculate hedge payoff with optional asymmetric volatility consideration
        
        Args:
            initial_price: Initial underlying price
            final_price: Final underlying price
            portfolio_return: Unhedged portfolio return
            portfolio_allocation: Portfolio weights
            use_asymmetric_vol: Whether to use asymmetric volatility (defaults to instance setting)
            
        Returns:
            Hedge payoff as fraction of portfolio value
        """
        if use_asymmetric_vol is None:
            use_asymmetric_vol = self.use_asymmetric_pricing
        
        equity_exposure = portfolio_allocation.get("equity", 0.6)
        hedge_notional = equity_exposure * self.option_params.coverage_ratio
        
        if self.hedge_type == HedgeType.PUT_PROTECTION:
            return self._put_protection_payoff_asymmetric(
                initial_price, final_price, hedge_notional, portfolio_return, use_asymmetric_vol
            )
        elif self.hedge_type == HedgeType.VIX_CALLS:
            return self._vix_calls_payoff(portfolio_return, hedge_notional)
        elif self.hedge_type == HedgeType.COLLAR:
            return self._collar_payoff(initial_price, final_price, hedge_notional)
        elif self.hedge_type == HedgeType.CPPI:
            return self._cppi_adjustment(portfolio_return, portfolio_allocation)
        else:
            return 0.0
    
    def _put_protection_payoff_asymmetric(
        self,
        initial_price: float,
        final_price: float,
        hedge_notional: float,
        portfolio_return: float,
        use_asymmetric_vol: bool
    ) -> float:
        """
        Calculate put protection payoff considering asymmetric volatility effects
        
        In asymmetric volatility environments, puts may provide more protection
        than symmetric models suggest, especially during stress periods.
        """
        strike_price = initial_price * self.option_params.moneyness
        
        # Basic intrinsic payoff
        intrinsic_payoff = max(0, strike_price - final_price)
        
        if use_asymmetric_vol and portfolio_return < -0.02:
            # In downside scenarios, asymmetric volatility may increase hedge effectiveness
            # due to higher implied volatility and better correlation with stress
            asymmetric_multiplier = self.market_model.asymmetric_vol_model.downside_multiplier
            volatility_boost = min(1.3, 1 + (asymmetric_multiplier - 1) * 0.2)  # Cap the boost
            intrinsic_payoff *= volatility_boost
        
        total_payoff = hedge_notional * intrinsic_payoff
        return total_payoff / self.portfolio_value