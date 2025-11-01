"""
Alternative Risk Premia Module

Enhanced simulation of alternative risk premia with realistic characteristics:
- Momentum, carry, value, and volatility factors
- Non-normal return distributions with asymmetric features
- Factor correlation dynamics and regime dependence
- Performance attribution and factor decomposition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.stats import skewnorm, t
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskPremiaFactor(Enum):
    """Alternative risk premia factors"""
    MOMENTUM = "momentum"
    CARRY = "carry"
    VALUE = "value"
    VOLATILITY = "volatility"
    QUALITY = "quality"
    SIZE = "size"


@dataclass
class FactorParameters:
    """
    Parameters for individual risk premia factors
    
    Based on empirical research on alternative risk premia characteristics
    """
    name: str
    expected_return: float  # Annual expected excess return
    volatility: float  # Annual volatility
    skewness: float  # Return distribution skewness
    excess_kurtosis: float  # Excess kurtosis (fat tails)
    max_drawdown: float  # Historical maximum drawdown
    persistence: float  # Return persistence (AR(1) coefficient)
    regime_sensitivity: float  # Sensitivity to market regimes


class AlternativeRiskPremia:
    """
    Enhanced alternative risk premia simulation with realistic features
    
    Features:
    - Multiple factor modeling with cross-correlations
    - Regime-dependent behavior
    - Time-varying volatility and correlations
    - Realistic distributional properties
    """
    
    def __init__(self, factors: Optional[List[RiskPremiaFactor]] = None):
        """
        Initialize alternative risk premia model
        
        Args:
            factors: List of factors to include (default: all main factors)
        """
        if factors is None:
            factors = [
                RiskPremiaFactor.MOMENTUM,
                RiskPremiaFactor.CARRY,
                RiskPremiaFactor.VALUE,
                RiskPremiaFactor.VOLATILITY
            ]
        
        self.factors = factors
        self.factor_params = self._create_factor_parameters()
        self.correlation_matrix = self._create_base_correlation_matrix()
        
        logger.info(f"Initialized risk premia model with {len(factors)} factors")
    
    def _create_factor_parameters(self) -> Dict[RiskPremiaFactor, FactorParameters]:
        """Create realistic factor parameters based on empirical research"""
        params = {
            RiskPremiaFactor.MOMENTUM: FactorParameters(
                name="Momentum",
                expected_return=0.08,  # 8% annual excess return
                volatility=0.12,       # 12% annual volatility
                skewness=-0.6,         # Negative skew (crash risk)
                excess_kurtosis=3.0,   # Fat tails
                max_drawdown=0.35,     # 35% max drawdown historically
                persistence=0.15,      # Moderate persistence
                regime_sensitivity=1.2  # More sensitive in stress
            ),
            RiskPremiaFactor.CARRY: FactorParameters(
                name="Carry",
                expected_return=0.06,
                volatility=0.10,
                skewness=-0.8,         # Strong negative skew
                excess_kurtosis=4.0,   # Very fat tails
                max_drawdown=0.25,
                persistence=0.25,      # Higher persistence
                regime_sensitivity=1.4
            ),
            RiskPremiaFactor.VALUE: FactorParameters(
                name="Value",
                expected_return=0.05,
                volatility=0.14,
                skewness=-0.3,         # Mild negative skew
                excess_kurtosis=1.5,
                max_drawdown=0.30,
                persistence=0.05,      # Low persistence
                regime_sensitivity=0.8  # Less sensitive to regimes
            ),
            RiskPremiaFactor.VOLATILITY: FactorParameters(
                name="Volatility",
                expected_return=0.04,
                volatility=0.16,
                skewness=0.2,          # Slight positive skew
                excess_kurtosis=2.0,
                max_drawdown=0.40,
                persistence=0.30,      # High persistence
                regime_sensitivity=1.6  # Very sensitive
            ),
            RiskPremiaFactor.QUALITY: FactorParameters(
                name="Quality",
                expected_return=0.04,
                volatility=0.08,
                skewness=0.1,
                excess_kurtosis=1.0,
                max_drawdown=0.20,
                persistence=0.20,
                regime_sensitivity=0.6  # Defensive
            ),
            RiskPremiaFactor.SIZE: FactorParameters(
                name="Size",
                expected_return=0.03,
                volatility=0.15,
                skewness=-0.4,
                excess_kurtosis=2.5,
                max_drawdown=0.45,
                persistence=0.10,
                regime_sensitivity=1.1
            )
        }
        
        return {factor: params[factor] for factor in self.factors}
    
    def _create_base_correlation_matrix(self) -> np.ndarray:
        """Create base correlation matrix between factors"""
        n_factors = len(self.factors)
        
        # Empirically-based correlations
        base_correlations = {
            (RiskPremiaFactor.MOMENTUM, RiskPremiaFactor.CARRY): 0.15,
            (RiskPremiaFactor.MOMENTUM, RiskPremiaFactor.VALUE): -0.20,
            (RiskPremiaFactor.MOMENTUM, RiskPremiaFactor.VOLATILITY): -0.10,
            (RiskPremiaFactor.CARRY, RiskPremiaFactor.VALUE): 0.05,
            (RiskPremiaFactor.CARRY, RiskPremiaFactor.VOLATILITY): -0.25,
            (RiskPremiaFactor.VALUE, RiskPremiaFactor.VOLATILITY): -0.05,
            (RiskPremiaFactor.QUALITY, RiskPremiaFactor.MOMENTUM): 0.10,
            (RiskPremiaFactor.QUALITY, RiskPremiaFactor.VALUE): 0.30,
            (RiskPremiaFactor.SIZE, RiskPremiaFactor.VALUE): 0.40,
            (RiskPremiaFactor.SIZE, RiskPremiaFactor.MOMENTUM): 0.25
        }
        
        # Create correlation matrix
        correlation_matrix = np.eye(n_factors)
        
        for i, factor_i in enumerate(self.factors):
            for j, factor_j in enumerate(self.factors):
                if i != j:
                    pair = (factor_i, factor_j)
                    reverse_pair = (factor_j, factor_i)
                    
                    if pair in base_correlations:
                        correlation_matrix[i, j] = base_correlations[pair]
                    elif reverse_pair in base_correlations:
                        correlation_matrix[i, j] = base_correlations[reverse_pair]
        
        return correlation_matrix
    
    def simulate_risk_premia(
        self,
        n_periods: int = 252,
        regime_path: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Simulate alternative risk premia with realistic characteristics
        
        Args:
            n_periods: Number of periods to simulate
            regime_path: Market regime for each period (0=normal, 1=stress)
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with factor returns
        """
        if seed is not None:
            np.random.seed(seed)
        
        if regime_path is None:
            regime_path = self._simulate_regime_path(n_periods)
        
        # Initialize return series
        factor_returns = {}
        factor_states = {}  # For persistence modeling
        
        for factor in self.factors:
            factor_returns[factor.value] = np.zeros(n_periods)
            factor_states[factor.value] = 0.0  # Initial state
        
        # Simulate period by period
        for t in range(n_periods):
            current_regime = regime_path[t]
            
            # Adjust correlation matrix for regime
            regime_corr_matrix = self._adjust_correlation_for_regime(
                self.correlation_matrix, current_regime
            )
            
            # Generate correlated random shocks
            random_shocks = np.random.multivariate_normal(
                np.zeros(len(self.factors)), regime_corr_matrix
            )
            
            # Generate returns for each factor
            for i, factor in enumerate(self.factors):
                params = self.factor_params[factor]
                
                # Adjust parameters for regime
                regime_vol_mult = 1.0 + current_regime * (params.regime_sensitivity - 1.0)
                current_vol = params.volatility * regime_vol_mult / np.sqrt(252)  # Daily vol
                
                # Base return with regime adjustment
                base_return = params.expected_return / 252  # Daily expected return
                if current_regime > 0:  # Stress regime
                    base_return *= (2 - params.regime_sensitivity)  # Reduce in stress
                
                # Add persistence from previous period
                persistence_component = params.persistence * factor_states[factor.value]
                
                # Generate skewed and fat-tailed innovations
                if abs(params.skewness) > 0.01 or params.excess_kurtosis > 0.5:
                    # Use skewed t-distribution for fat tails and skewness
                    df = 4 + params.excess_kurtosis  # Degrees of freedom for t-dist
                    innovation = skewnorm.rvs(
                        params.skewness, 
                        loc=random_shocks[i], 
                        scale=1
                    )
                    innovation *= t.rvs(df) / np.sqrt(df / (df - 2))  # Add fat tails
                else:
                    innovation = random_shocks[i]
                
                # Combine components
                factor_return = (
                    base_return + 
                    persistence_component + 
                    current_vol * innovation
                )
                
                factor_returns[factor.value][t] = factor_return
                factor_states[factor.value] = factor_return  # Update state
        
        # Apply asymmetric amplification to negative returns
        for factor in self.factors:
            returns = factor_returns[factor.value]
            params = self.factor_params[factor]
            
            # Amplify negative returns based on skewness
            if params.skewness < 0:
                amplification = 1.0 + abs(params.skewness)
                negative_mask = returns < 0
                factor_returns[factor.value][negative_mask] *= amplification
        
        return pd.DataFrame(factor_returns)
    
    def _simulate_regime_path(self, n_periods: int) -> np.ndarray:
        """Simulate market regime path (0=normal, 1=stress)"""
        regime_path = np.zeros(n_periods)
        
        # Regime transition probabilities
        prob_normal_to_stress = 0.02  # 2% daily probability
        prob_stress_to_normal = 0.10  # 10% daily probability
        
        current_regime = 0  # Start in normal regime
        
        for t in range(1, n_periods):
            if current_regime == 0:  # Normal regime
                if np.random.random() < prob_normal_to_stress:
                    current_regime = 1
            else:  # Stress regime
                if np.random.random() < prob_stress_to_normal:
                    current_regime = 0
            
            regime_path[t] = current_regime
        
        return regime_path
    
    def _adjust_correlation_for_regime(
        self, 
        base_correlation: np.ndarray, 
        regime: float
    ) -> np.ndarray:
        """Adjust correlation matrix based on market regime"""
        # Correlations increase in stress periods
        stress_multiplier = 1.0 + regime * 0.5  # 50% increase in stress
        
        adjusted_corr = base_correlation.copy()
        
        # Increase off-diagonal correlations
        for i in range(len(adjusted_corr)):
            for j in range(len(adjusted_corr)):
                if i != j:
                    adjusted_corr[i, j] *= stress_multiplier
                    # Ensure matrix remains valid
                    adjusted_corr[i, j] = np.clip(adjusted_corr[i, j], -0.95, 0.95)
        
        return adjusted_corr
    
    def factor_attribution(
        self, 
        portfolio_returns: np.ndarray,
        factor_returns: pd.DataFrame,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Perform factor attribution analysis
        
        Args:
            portfolio_returns: Portfolio return series
            factor_returns: Factor return DataFrame
            window: Rolling window for regression
            
        Returns:
            DataFrame with factor loadings and attribution
        """
        results = []
        
        for i in range(window, len(portfolio_returns)):
            # Get window data
            y = portfolio_returns[i-window:i]
            X = factor_returns.iloc[i-window:i].values
            
            # Add constant term for alpha
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            # Regression
            try:
                betas = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                alpha = betas[0]
                factor_betas = betas[1:]
                
                # Calculate R-squared
                y_pred = X_with_const @ betas
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Factor contributions
                current_factor_returns = factor_returns.iloc[i].values
                factor_contributions = factor_betas * current_factor_returns
                
                result = {
                    'date': i,
                    'alpha': alpha,
                    'r_squared': r_squared,
                    'total_factor_return': np.sum(factor_contributions)
                }
                
                # Add individual factor contributions
                for j, factor in enumerate(factor_returns.columns):
                    result[f'{factor}_beta'] = factor_betas[j]
                    result[f'{factor}_contribution'] = factor_contributions[j]
                
                results.append(result)
                
            except np.linalg.LinAlgError:
                # Skip if regression fails
                continue
        
        return pd.DataFrame(results)
    
    def calculate_factor_statistics(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive statistics for each factor
        
        Args:
            factor_returns: Factor return DataFrame
            
        Returns:
            DataFrame with factor statistics
        """
        stats = []
        
        for factor in factor_returns.columns:
            returns = factor_returns[factor].dropna()
            
            if len(returns) > 0:
                # Basic statistics
                stat = {
                    'Factor': factor,
                    'Annual_Return': returns.mean() * 252,
                    'Annual_Volatility': returns.std() * np.sqrt(252),
                    'Sharpe_Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                    'Skewness': returns.skew(),
                    'Excess_Kurtosis': returns.kurtosis(),
                    'VaR_95': np.percentile(returns, 5),
                    'Max_Drawdown': self._calculate_max_drawdown(returns),
                    'Hit_Rate': (returns > 0).mean(),
                    'Worst_Month': returns.min(),
                    'Best_Month': returns.max()
                }
                
                # Downside statistics
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    stat['Downside_Deviation'] = downside_returns.std() * np.sqrt(252)
                    stat['Sortino_Ratio'] = (returns.mean() * 252) / stat['Downside_Deviation'] if stat['Downside_Deviation'] > 0 else 0
                else:
                    stat['Downside_Deviation'] = 0
                    stat['Sortino_Ratio'] = np.inf
                
                stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        return abs(drawdown.min())
    
    def create_optimized_portfolio(
        self,
        factor_returns: pd.DataFrame,
        loss_aversion: float = 2.0
    ) -> Dict[str, float]:
        """
        Create an optimized portfolio using a behavioral utility function.

        This optimization penalizes losses more heavily than gains, reflecting
        loss aversion from behavioral finance (Kahneman & Tversky).

        Args:
            factor_returns: DataFrame of factor returns.
            loss_aversion: The multiplier for penalizing losses (lambda).
                           A value of 2.0 means losses are twice as painful as gains.

        Returns:
            A dictionary of optimized asset weights.
        """
        num_assets = factor_returns.shape[1]

        def behavioral_utility(weights: np.ndarray) -> float:
            """
            Objective function to maximize: behavioral utility.
            We minimize the negative of this utility.
            """
            portfolio_returns = factor_returns.values @ weights
            
            # Apply loss aversion
            utility = np.where(
                portfolio_returns >= 0,
                portfolio_returns,
                portfolio_returns * loss_aversion
            )
            
            # We want to maximize the mean of this utility
            return -np.mean(utility)

        # Constraints: weights sum to 1 and are between 0 and 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess: equal weights
        initial_weights = np.ones(num_assets) / num_assets
        
        # Perform optimization
        result = minimize(
            behavioral_utility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning("Optimization may not have succeeded.")
            logger.warning(result.message)

        optimized_weights = result.x
        
        return dict(zip(factor_returns.columns, optimized_weights))

    def create_factor_portfolios(
        self,
        factor_returns: pd.DataFrame,
        allocation_method: str = "equal_weight"
    ) -> Dict[str, np.ndarray]:
        """
        Create factor-based portfolio allocations
        
        Args:
            factor_returns: Factor return DataFrame
            allocation_method: Method for combining factors
            
        Returns:
            Dictionary of portfolio weights for each factor strategy
        """
        portfolios = {}
        
        if allocation_method == "equal_weight":
            # Equal weight all factors
            equal_weights = np.ones(len(factor_returns.columns)) / len(factor_returns.columns)
            portfolios["Equal_Weight"] = equal_weights
            
        elif allocation_method == "risk_parity":
            # Risk parity weighting
            factor_vols = factor_returns.std()
            inv_vol_weights = 1 / factor_vols
            risk_parity_weights = inv_vol_weights / inv_vol_weights.sum()
            portfolios["Risk_Parity"] = risk_parity_weights.values
            
        elif allocation_method == "momentum":
            # Momentum-based weighting (recent performance)
            recent_returns = factor_returns.tail(21).mean()  # Last month
            momentum_weights = np.maximum(recent_returns, 0)
            if momentum_weights.sum() > 0:
                momentum_weights = momentum_weights / momentum_weights.sum()
            else:
                momentum_weights = np.ones(len(factor_returns.columns)) / len(factor_returns.columns)
            portfolios["Momentum"] = momentum_weights.values
        
        return portfolios


# Legacy function for backward compatibility
def simulate_risk_premia(n_periods=252, seed=42):
    """Legacy function - use AlternativeRiskPremia class for enhanced features"""
    rp_model = AlternativeRiskPremia()
    return rp_model.simulate_risk_premia(n_periods=n_periods, seed=seed)


# Example usage and testing
if __name__ == "__main__":
    # Create risk premia model
    rp_model = AlternativeRiskPremia()
    
    print("=== SIMULATING ALTERNATIVE RISK PREMIA ===")
    factor_returns = rp_model.simulate_risk_premia(n_periods=252*2, seed=42)
    
    print("\n=== FACTOR STATISTICS ===")
    stats = rp_model.calculate_factor_statistics(factor_returns)
    print(stats.round(3))
    
    print("\n=== FACTOR CORRELATIONS ===")
    correlations = factor_returns.corr()
    print(correlations.round(3))
    
    # Test factor attribution
    print("\n=== FACTOR ATTRIBUTION EXAMPLE ===")
    # Create a simple portfolio return series for testing
    portfolio_rets = (factor_returns * [0.3, 0.3, 0.2, 0.2]).sum(axis=1).values
    attribution = rp_model.factor_attribution(portfolio_rets, factor_returns)
    
    if not attribution.empty:
        print(f"Average factor attribution over {len(attribution)} periods:")
        for col in attribution.columns:
            if 'contribution' in col:
                print(f"{col}: {attribution[col].mean():.4f}")
    
    print("\n=== FACTOR PORTFOLIO ALLOCATIONS ===")
    portfolios = rp_model.create_factor_portfolios(factor_returns, "risk_parity")
    for name, weights in portfolios.items():
        print(f"{name}: {dict(zip(factor_returns.columns, weights.round(3)))}")

    print("\n=== BEHAVIORAL OPTIMIZED PORTFOLIO ===")
    optimized_weights = rp_model.create_optimized_portfolio(factor_returns, loss_aversion=2.25)
    print(f"Optimized (Loss Aversion): { {k: f'{v:.3f}' for k, v in optimized_weights.items()} }")
