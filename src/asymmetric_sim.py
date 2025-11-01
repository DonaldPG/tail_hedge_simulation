"""
Asymmetric Simulation Module

Implements Monte Carlo simulation with asymmetric volatility and downside shocks
based on the Deutsche Bank framework for modeling market uncertainty vs risk.

Features:
- Geometric Brownian Motion with conditional volatility
- Jump-diffusion processes for market stress
- Skewed distributions for asymmetric returns
- Portfolio path simulation with hedging overlays
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import logging
from scipy.stats import norm, skewnorm, t
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types for conditional modeling"""
    NORMAL = "normal"
    STRESS = "stress"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class AssetParameters:
    """
    Asset-specific parameters for simulation
    
    Includes both symmetric and asymmetric risk characteristics
    """
    name: str
    expected_return: float  # Annual expected return
    base_volatility: float  # Base annual volatility
    downside_volatility_multiplier: float  # Extra volatility on negative returns
    skewness: float  # Return distribution skewness
    excess_kurtosis: float  # Excess kurtosis (0 = normal)
    jump_probability: float  # Daily probability of jumps
    jump_mean: float  # Mean jump size
    jump_volatility: float  # Jump size volatility
    correlation_beta: float  # Beta to market factor


@dataclass
class SimulationParameters:
    """
    Parameters for Monte Carlo simulation
    """
    num_simulations: int = 10000
    num_days: int = 252  # Trading days per year
    dt: float = 1/252  # Time step (daily)
    random_seed: Optional[int] = None
    use_antithetic_variates: bool = True
    regime_switching: bool = True


class AsymmetricVolatilityModel:
    """
    Asymmetric volatility model with conditional heteroskedasticity
    
    Implements the volatility amplification on negative returns
    as described in the Deutsche Bank framework
    """
    
    def __init__(self, base_vol: float, asymmetry_factor: float = 1.5):
        """
        Initialize asymmetric volatility model
        
        Args:
            base_vol: Base volatility level
            asymmetry_factor: Multiplier for downside volatility
        """
        self.base_vol = base_vol
        self.asymmetry_factor = asymmetry_factor
        self.vol_persistence = 0.9  # Volatility persistence parameter
        self.current_vol = base_vol
    
    def update_volatility(self, return_shock: float) -> float:
        """
        Update volatility based on return shock
        
        Args:
            return_shock: Current period return shock
            
        Returns:
            Updated volatility for next period
        """
        # Asymmetric response to negative shocks
        if return_shock < 0:
            vol_shock = self.asymmetry_factor * abs(return_shock)
        else:
            vol_shock = 0.5 * return_shock  # Smaller response to positive shocks
        
        # Update with persistence and mean reversion
        self.current_vol = (
            self.vol_persistence * self.current_vol +
            (1 - self.vol_persistence) * self.base_vol +
            0.1 * vol_shock
        )
        
        return self.current_vol


class JumpDiffusionModel:
    """
    Jump-diffusion model for modeling tail events and market stress
    """
    
    def __init__(
        self,
        jump_intensity: float,
        jump_mean: float,
        jump_std: float,
        regime_dependent: bool = True
    ):
        """
        Initialize jump-diffusion model
        
        Args:
            jump_intensity: Annual frequency of jumps
            jump_mean: Mean jump size (negative for market stress)
            jump_std: Standard deviation of jump sizes
            regime_dependent: Whether jump parameters depend on market regime
        """
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.regime_dependent = regime_dependent
    
    def generate_jumps(
        self,
        num_periods: int,
        dt: float,
        regime: MarketRegime = MarketRegime.NORMAL
    ) -> np.ndarray:
        """
        Generate jump components for simulation
        
        Args:
            num_periods: Number of time periods
            dt: Time step size
            regime: Current market regime
            
        Returns:
            Array of jump contributions
        """
        # Adjust jump parameters based on regime
        if self.regime_dependent:
            regime_adjustments = {
                MarketRegime.NORMAL: {"intensity": 1.0, "mean": 1.0, "std": 1.0},
                MarketRegime.STRESS: {"intensity": 2.0, "mean": 1.5, "std": 1.3},
                MarketRegime.CRISIS: {"intensity": 4.0, "mean": 2.0, "std": 1.8},
                MarketRegime.RECOVERY: {"intensity": 1.5, "mean": 0.8, "std": 1.1}
            }
            
            adj = regime_adjustments[regime]
            current_intensity = self.jump_intensity * adj["intensity"]
            current_mean = self.jump_mean * adj["mean"]
            current_std = self.jump_std * adj["std"]
        else:
            current_intensity = self.jump_intensity
            current_mean = self.jump_mean
            current_std = self.jump_std
        
        # Generate Poisson process for jump times
        jump_prob = current_intensity * dt
        jumps = np.random.poisson(jump_prob, num_periods)
        
        # Generate jump sizes
        jump_sizes = np.random.normal(current_mean, current_std, num_periods)
        
        # Combine jump occurrence with jump sizes
        jump_contributions = jumps * jump_sizes
        
        return jump_contributions


class AsymmetricSimulator:
    """
    Main simulation engine for asymmetric portfolio modeling
    
    Combines geometric Brownian motion with:
    - Asymmetric volatility
    - Jump-diffusion processes
    - Skewed return distributions
    - Multi-asset correlations
    """
    
    def __init__(
        self,
        assets: List[AssetParameters],
        simulation_params: SimulationParameters
    ):
        """
        Initialize asymmetric simulator
        
        Args:
            assets: List of asset parameters
            simulation_params: Simulation configuration
        """
        self.assets = assets
        self.sim_params = simulation_params
        
        # Initialize random seed if specified
        if simulation_params.random_seed is not None:
            np.random.seed(simulation_params.random_seed)
        
        # Initialize volatility models for each asset
        self.vol_models = {}
        self.jump_models = {}
        
        for asset in assets:
            self.vol_models[asset.name] = AsymmetricVolatilityModel(
                asset.base_volatility,
                asset.downside_volatility_multiplier
            )
            
            self.jump_models[asset.name] = JumpDiffusionModel(
                asset.jump_probability * 252,  # Convert to annual
                asset.jump_mean,
                asset.jump_volatility
            )
        
        logger.info(f"Initialized simulator for {len(assets)} assets with {simulation_params.num_simulations} paths")
    
    def generate_correlation_matrix(self, regime: MarketRegime = MarketRegime.NORMAL) -> np.ndarray:
        """
        Generate correlation matrix that varies by market regime
        
        Args:
            regime: Current market regime
            
        Returns:
            Correlation matrix for assets
        """
        num_assets = len(self.assets)
        base_correlation = 0.3  # Base correlation in normal times
        
        # Correlations increase in stress periods
        regime_correlation_multipliers = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.STRESS: 1.5,
            MarketRegime.CRISIS: 2.0,
            MarketRegime.RECOVERY: 1.2
        }
        
        stress_correlation = base_correlation * regime_correlation_multipliers[regime]
        stress_correlation = min(0.8, stress_correlation)  # Cap at 0.8
        
        # Create correlation matrix
        correlation_matrix = np.eye(num_assets)
        
        for i in range(num_assets):
            for j in range(i + 1, num_assets):
                # Use beta-based correlation
                beta_i = self.assets[i].correlation_beta
                beta_j = self.assets[j].correlation_beta
                correlation = stress_correlation * beta_i * beta_j
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def simulate_market_regime(self) -> np.ndarray:
        """
        Simulate market regime transitions over time
        
        Returns:
            Array of market regimes for each time period
        """
        num_periods = self.sim_params.num_days
        regimes = np.full(num_periods, MarketRegime.NORMAL)
        
        if not self.sim_params.regime_switching:
            return regimes
        
        # Transition probabilities (daily)
        transition_probs = {
            MarketRegime.NORMAL: {
                MarketRegime.NORMAL: 0.98,
                MarketRegime.STRESS: 0.015,
                MarketRegime.CRISIS: 0.003,
                MarketRegime.RECOVERY: 0.002
            },
            MarketRegime.STRESS: {
                MarketRegime.NORMAL: 0.1,
                MarketRegime.STRESS: 0.7,
                MarketRegime.CRISIS: 0.15,
                MarketRegime.RECOVERY: 0.05
            },
            MarketRegime.CRISIS: {
                MarketRegime.NORMAL: 0.02,
                MarketRegime.STRESS: 0.3,
                MarketRegime.CRISIS: 0.6,
                MarketRegime.RECOVERY: 0.08
            },
            MarketRegime.RECOVERY: {
                MarketRegime.NORMAL: 0.3,
                MarketRegime.STRESS: 0.05,
                MarketRegime.CRISIS: 0.05,
                MarketRegime.RECOVERY: 0.6
            }
        }
        
        current_regime = MarketRegime.NORMAL
        
        for t in range(1, num_periods):
            # Draw random number for regime transition
            rand = np.random.random()
            
            # Determine next regime based on transition probabilities
            cumsum = 0
            for next_regime, prob in transition_probs[current_regime].items():
                cumsum += prob
                if rand <= cumsum:
                    current_regime = next_regime
                    break
            
            regimes[t] = current_regime
        
        return regimes
    
    def simulate_asset_paths(self) -> Dict[str, np.ndarray]:
        """
        Simulate asset price paths with asymmetric features
        
        Returns:
            Dictionary mapping asset names to price path arrays
            Shape: (num_simulations, num_days + 1)
        """
        num_sims = self.sim_params.num_simulations
        num_periods = self.sim_params.num_days
        dt = self.sim_params.dt
        
        # Initialize result dictionary
        asset_paths = {}
        for asset in self.assets:
            asset_paths[asset.name] = np.zeros((num_sims, num_periods + 1))
            asset_paths[asset.name][:, 0] = 100  # Start at 100
        
        # Generate regime paths (one per simulation)
        regime_paths = []
        for sim in range(num_sims):
            regime_paths.append(self.simulate_market_regime())
        
        # Simulate each path
        for sim in range(num_sims):
            if sim % 1000 == 0:
                logger.info(f"Simulating path {sim + 1}/{num_sims}")
            
            # Reset volatility models for each simulation
            for asset in self.assets:
                self.vol_models[asset.name].current_vol = asset.base_volatility
            
            current_regime = regime_paths[sim]
            
            for t in range(num_periods):
                # Generate correlation matrix for current regime
                correlation_matrix = self.generate_correlation_matrix(current_regime[t])
                
                # Generate correlated random shocks
                if self.sim_params.use_antithetic_variates and sim < num_sims // 2:
                    # First half: normal random numbers
                    random_shocks = np.random.multivariate_normal(
                        np.zeros(len(self.assets)), correlation_matrix
                    )
                elif self.sim_params.use_antithetic_variates:
                    # Second half: antithetic variates
                    random_shocks = -np.random.multivariate_normal(
                        np.zeros(len(self.assets)), correlation_matrix
                    )
                else:
                    random_shocks = np.random.multivariate_normal(
                        np.zeros(len(self.assets)), correlation_matrix
                    )
                
                # Simulate each asset
                for i, asset in enumerate(self.assets):
                    current_price = asset_paths[asset.name][sim, t]
                    
                    # Get current volatility
                    current_vol = self.vol_models[asset.name].current_vol
                    
                    # Generate jump component
                    jump_component = self.jump_models[asset.name].generate_jumps(
                        1, dt, current_regime[t]
                    )[0]
                    
                    # Apply skewness to random shock if specified
                    if abs(asset.skewness) > 0.01:
                        shock = skewnorm.rvs(asset.skewness, loc=random_shocks[i], scale=1)
                    else:
                        shock = random_shocks[i]
                    
                    # Calculate return with asymmetric volatility
                    drift = (asset.expected_return - 0.5 * current_vol**2) * dt
                    diffusion = current_vol * np.sqrt(dt) * shock
                    jump = jump_component
                    
                    log_return = drift + diffusion + jump
                    
                    # Update price
                    asset_paths[asset.name][sim, t + 1] = current_price * np.exp(log_return)
                    
                    # Update volatility model
                    return_shock = log_return / np.sqrt(dt)  # Standardized return
                    self.vol_models[asset.name].update_volatility(return_shock)
        
        logger.info("Simulation completed")
        return asset_paths
    
    def simulate_portfolio_paths(
        self,
        portfolio_weights: Dict[str, float],
        rebalancing_frequency: int = 21  # Monthly rebalancing
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Simulate portfolio paths given asset weights
        
        Args:
            portfolio_weights: Dictionary of asset weights
            rebalancing_frequency: Days between rebalancing
            
        Returns:
            Tuple of (portfolio_paths, asset_paths)
        """
        # First simulate asset paths
        asset_paths = self.simulate_asset_paths()
        
        num_sims = self.sim_params.num_simulations
        num_periods = self.sim_params.num_days
        
        # Initialize portfolio paths
        portfolio_paths = np.zeros((num_sims, num_periods + 1))
        portfolio_paths[:, 0] = 100  # Start at 100
        
        # Calculate portfolio values
        for sim in range(num_sims):
            current_weights = portfolio_weights.copy()
            
            for t in range(num_periods):
                # Calculate portfolio return
                portfolio_return = 0
                
                for asset_name, weight in current_weights.items():
                    if asset_name in asset_paths:
                        asset_return = (
                            asset_paths[asset_name][sim, t + 1] / 
                            asset_paths[asset_name][sim, t] - 1
                        )
                        portfolio_return += weight * asset_return
                
                # Update portfolio value
                portfolio_paths[sim, t + 1] = portfolio_paths[sim, t] * (1 + portfolio_return)
                
                # Rebalance if needed
                if (t + 1) % rebalancing_frequency == 0:
                    # In practice, would calculate new weights based on drift
                    # For simplicity, maintain target weights
                    pass
        
        return portfolio_paths, asset_paths
    
    def calculate_risk_metrics(
        self,
        portfolio_paths: np.ndarray,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics from portfolio paths
        
        Args:
            portfolio_paths: Array of portfolio value paths
            confidence_levels: VaR confidence levels to calculate
            
        Returns:
            Dictionary of risk metrics
        """
        # Calculate returns
        returns = np.diff(portfolio_paths, axis=1) / portfolio_paths[:, :-1]
        portfolio_returns = np.mean(returns, axis=1)  # Average return per simulation
        
        # Basic statistics
        final_values = portfolio_paths[:, -1]
        total_returns = (final_values / portfolio_paths[:, 0]) - 1
        
        metrics = {
            "expected_return": np.mean(total_returns),
            "volatility": np.std(total_returns),
            "skewness": self._calculate_skewness(total_returns),
            "excess_kurtosis": self._calculate_excess_kurtosis(total_returns),
        }
        
        # Downside risk metrics
        downside_returns = total_returns[total_returns < 0]
        if len(downside_returns) > 0:
            metrics["downside_deviation"] = np.std(downside_returns)
            metrics["worst_case"] = np.min(total_returns)
        else:
            metrics["downside_deviation"] = 0
            metrics["worst_case"] = 0
        
        # VaR and Expected Shortfall
        for conf_level in confidence_levels:
            var_level = 1 - conf_level
            var_value = np.percentile(total_returns, var_level * 100)
            es_value = np.mean(total_returns[total_returns <= var_value])
            
            metrics[f"var_{int(conf_level*100)}"] = var_value
            metrics[f"expected_shortfall_{int(conf_level*100)}"] = es_value
        
        # Maximum drawdown
        metrics["max_drawdown"] = self._calculate_max_drawdown(portfolio_paths)
        
        # Tail ratios
        left_tail = np.percentile(total_returns, 5)
        right_tail = np.percentile(total_returns, 95)
        metrics["tail_ratio"] = abs(right_tail / left_tail) if left_tail < 0 else np.inf
        
        return metrics
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate sample skewness"""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        n = len(returns)
        
        if std_return == 0:
            return 0
        
        skew = (n / ((n-1) * (n-2))) * np.sum(((returns - mean_return) / std_return) ** 3)
        return skew
    
    def _calculate_excess_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate sample excess kurtosis"""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        n = len(returns)
        
        if std_return == 0:
            return 0
        
        kurt = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((returns - mean_return) / std_return) ** 4)
        kurt -= 3 * (n-1)**2 / ((n-2) * (n-3))
        
        return kurt
    
    def _calculate_max_drawdown(self, portfolio_paths: np.ndarray) -> float:
        """Calculate maximum drawdown across all paths"""
        max_dd = 0
        
        for sim in range(portfolio_paths.shape[0]):
            path = portfolio_paths[sim, :]
            peak = np.maximum.accumulate(path)
            drawdown = (path - peak) / peak
            max_dd = max(max_dd, abs(np.min(drawdown)))
        
        return max_dd


def create_default_assets() -> List[AssetParameters]:
    """
    Create default asset parameters for typical multi-asset portfolio
    
    Returns:
        List of default asset parameters
    """
    return [
        AssetParameters(
            name="US_Equity",
            expected_return=0.08,
            base_volatility=0.16,
            downside_volatility_multiplier=1.6,
            skewness=-0.5,
            excess_kurtosis=3.0,
            jump_probability=0.02,
            jump_mean=-0.02,
            jump_volatility=0.05,
            correlation_beta=1.0
        ),
        AssetParameters(
            name="International_Equity",
            expected_return=0.075,
            base_volatility=0.18,
            downside_volatility_multiplier=1.5,
            skewness=-0.4,
            excess_kurtosis=2.5,
            jump_probability=0.015,
            jump_mean=-0.015,
            jump_volatility=0.04,
            correlation_beta=0.8
        ),
        AssetParameters(
            name="Government_Bonds",
            expected_return=0.03,
            base_volatility=0.05,
            downside_volatility_multiplier=1.2,
            skewness=0.1,
            excess_kurtosis=1.0,
            jump_probability=0.005,
            jump_mean=0.01,
            jump_volatility=0.02,
            correlation_beta=-0.3
        ),
        AssetParameters(
            name="Corporate_Bonds",
            expected_return=0.045,
            base_volatility=0.08,
            downside_volatility_multiplier=1.4,
            skewness=-0.2,
            excess_kurtosis=2.0,
            jump_probability=0.01,
            jump_mean=-0.01,
            jump_volatility=0.03,
            correlation_beta=0.4
        ),
        AssetParameters(
            name="Commodities",
            expected_return=0.05,
            base_volatility=0.22,
            downside_volatility_multiplier=1.3,
            skewness=0.0,
            excess_kurtosis=2.5,
            jump_probability=0.025,
            jump_mean=0.0,
            jump_volatility=0.06,
            correlation_beta=0.3
        )
    ]


# Example usage and testing
if __name__ == "__main__":
    # Create simulation setup
    assets = create_default_assets()
    sim_params = SimulationParameters(
        num_simulations=1000,  # Reduced for testing
        num_days=252,
        random_seed=42
    )
    
    # Initialize simulator
    simulator = AsymmetricSimulator(assets, sim_params)
    
    # Define portfolio weights
    portfolio_weights = {
        "US_Equity": 0.4,
        "International_Equity": 0.2,
        "Government_Bonds": 0.2,
        "Corporate_Bonds": 0.1,
        "Commodities": 0.1
    }
    
    print("Starting portfolio simulation...")
    portfolio_paths, asset_paths = simulator.simulate_portfolio_paths(portfolio_weights)
    
    print("\nCalculating risk metrics...")
    risk_metrics = simulator.calculate_risk_metrics(portfolio_paths)
    
    print("\n=== PORTFOLIO RISK METRICS ===")
    for metric, value in risk_metrics.items():
        if "var" in metric or "expected_shortfall" in metric or "return" in metric:
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.4f}")
    
    # Plot sample paths
    plt.figure(figsize=(12, 8))
    
    # Plot first 100 portfolio paths
    for i in range(min(100, portfolio_paths.shape[0])):
        plt.plot(portfolio_paths[i, :], alpha=0.1, color='blue')
    
    # Plot mean path
    mean_path = np.mean(portfolio_paths, axis=0)
    plt.plot(mean_path, color='red', linewidth=2, label='Mean Path')
    
    # Plot percentiles
    p5 = np.percentile(portfolio_paths, 5, axis=0)
    p95 = np.percentile(portfolio_paths, 95, axis=0)
    plt.fill_between(range(len(p5)), p5, p95, alpha=0.2, color='gray', label='5th-95th Percentile')
    
    plt.title('Portfolio Value Simulation with Asymmetric Risk')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\nSimulation completed: {portfolio_paths.shape[0]} paths over {portfolio_paths.shape[1]-1} days")
