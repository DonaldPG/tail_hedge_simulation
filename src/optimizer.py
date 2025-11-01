"""
Portfolio Optimizer with Asymmetric Risk Preferences

Enhanced optimizer that incorporates:
- Semi-variance and CVaR for downside risk
- Behavioral utility with loss aversion
- Alternative risk premia integration
- Multi-objective optimization with asymmetric preferences
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
from enum import Enum

class OptimizationMethod(Enum):
    """Portfolio optimization methods"""
    MEAN_VARIANCE = "mean_variance"
    SEMI_VARIANCE = "semi_variance"
    CVAR = "cvar"
    BEHAVIORAL_UTILITY = "behavioral_utility"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_concentration: float = 0.4  # Maximum weight in single asset
    target_return: Optional[float] = None
    max_risk: Optional[float] = None
    turnover_limit: Optional[float] = None


class AsymmetricPortfolioOptimizer:
    """
    Enhanced portfolio optimizer with asymmetric risk preferences
    
    Implements multiple optimization methods with behavioral overlays
    """
    
    def __init__(
        self,
        method: OptimizationMethod = OptimizationMethod.BEHAVIORAL_UTILITY,
        loss_aversion: float = 2.25,
        confidence_level: float = 0.95
    ):
        """
        Initialize asymmetric portfolio optimizer
        
        Args:
            method: Optimization method to use
            loss_aversion: Loss aversion parameter for behavioral utility
            confidence_level: Confidence level for CVaR calculation
        """
        self.method = method
        self.loss_aversion = loss_aversion
        self.confidence_level = confidence_level
        
        logger.info(f"Initialized optimizer with {method.value} method")
    
    def optimize_portfolio(
        self,
        returns: np.ndarray,
        risk_premia: Optional[Dict[str, np.ndarray]] = None,
        constraints: OptimizationConstraints = OptimizationConstraints(),
        current_weights: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Optimize portfolio with asymmetric risk preferences
        
        Args:
            returns: Historical returns matrix (n_periods x n_assets)
            risk_premia: Dictionary of alternative risk premia
            constraints: Portfolio constraints
            current_weights: Current portfolio weights for turnover constraint
            
        Returns:
            Dictionary with optimal weights and optimization results
        """
        n_assets = returns.shape[1]
        
        # Initial guess
        if current_weights is not None:
            x0 = current_weights
        else:
            x0 = np.ones(n_assets) / n_assets
        
        # Set up bounds and constraints
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Weight sum constraint
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        
        # Maximum concentration constraint
        if constraints.max_concentration < 1.0:
            for i in range(n_assets):
                constraints_list.append({
                    "type": "ineq", 
                    "fun": lambda w, i=i: constraints.max_concentration - w[i]
                })
        
        # Target return constraint
        if constraints.target_return is not None:
            mean_returns = np.mean(returns, axis=0)
            constraints_list.append({
                "type": "eq",
                "fun": lambda w: np.dot(w, mean_returns) - constraints.target_return
            })
        
        # Turnover constraint
        if constraints.turnover_limit is not None and current_weights is not None:
            constraints_list.append({
                "type": "ineq",
                "fun": lambda w: constraints.turnover_limit - np.sum(np.abs(w - current_weights))
            })
        
        # Choose objective function based on method
        if self.method == OptimizationMethod.BEHAVIORAL_UTILITY:
            objective_func = lambda w: -self._behavioral_utility_objective(w, returns, risk_premia)
        elif self.method == OptimizationMethod.SEMI_VARIANCE:
            objective_func = lambda w: self._semi_variance_objective(w, returns)
        elif self.method == OptimizationMethod.CVAR:
            objective_func = lambda w: self._cvar_objective(w, returns)
        elif self.method == OptimizationMethod.MULTI_OBJECTIVE:
            objective_func = lambda w: self._multi_objective(w, returns, risk_premia)
        else:  # Mean-variance
            objective_func = lambda w: self._mean_variance_objective(w, returns)
        
        # Optimize
        result = minimize(
            objective_func,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            # Try alternative method
            result = differential_evolution(
                objective_func,
                bounds,
                constraints=constraints_list,
                seed=42
            )
        
        # Calculate portfolio metrics
        optimal_weights = result.x
        portfolio_metrics = self._calculate_portfolio_metrics(optimal_weights, returns, risk_premia)
        
        return {
            "weights": optimal_weights,
            "success": result.success,
            "objective_value": result.fun,
            "metrics": portfolio_metrics,
            "message": result.message if hasattr(result, 'message') else ""
        }
    
    def _behavioral_utility_objective(
        self, 
        weights: np.ndarray, 
        returns: np.ndarray,
        risk_premia: Optional[Dict[str, np.ndarray]] = None
    ) -> float:
        """Behavioral utility objective with loss aversion"""
        portfolio_returns = returns @ weights
        
        # Add risk premia if provided
        if risk_premia is not None:
            premia_contrib = self._calculate_risk_premia_contribution(weights, risk_premia)
            portfolio_returns += premia_contrib
        
        # Apply behavioral utility function
        gains = np.maximum(portfolio_returns, 0)
        losses = np.minimum(portfolio_returns, 0)
        
        # Prospect theory value function
        utility = np.sum(np.power(gains, 0.88)) - self.loss_aversion * np.sum(np.power(-losses, 0.88))
        
        return utility / len(portfolio_returns)
    
    def _semi_variance_objective(self, weights: np.ndarray, returns: np.ndarray) -> float:
        """Semi-variance objective focusing on downside risk"""
        portfolio_returns = returns @ weights
        downside_returns = portfolio_returns[portfolio_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        return np.var(downside_returns)
    
    def _cvar_objective(self, weights: np.ndarray, returns: np.ndarray) -> float:
        """Conditional Value at Risk (Expected Shortfall) objective"""
        portfolio_returns = returns @ weights
        var_threshold = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0.0
        
        return -np.mean(tail_losses)  # Minimize expected shortfall
    
    def _multi_objective(
        self, 
        weights: np.ndarray, 
        returns: np.ndarray,
        risk_premia: Optional[Dict[str, np.ndarray]] = None
    ) -> float:
        """Multi-objective combining return, risk, and behavioral preferences"""
        portfolio_returns = returns @ weights
        
        # Expected return component
        expected_return = np.mean(portfolio_returns)
        
        # Risk components
        volatility = np.std(portfolio_returns)
        semi_variance = self._semi_variance_objective(weights, returns)
        cvar = self._cvar_objective(weights, returns)
        
        # Behavioral utility component
        behavioral_utility = self._behavioral_utility_objective(weights, returns, risk_premia)
        
        # Weighted combination (maximize utility, minimize risk)
        objective = (
            -0.4 * behavioral_utility +  # Behavioral preference (maximize)
            0.3 * semi_variance +        # Downside risk (minimize)
            0.2 * cvar +                 # Tail risk (minimize)
            0.1 * volatility             # Total volatility (minimize)
        )
        
        return objective
    
    def _mean_variance_objective(self, weights: np.ndarray, returns: np.ndarray) -> float:
        """Traditional mean-variance objective"""
        portfolio_returns = returns @ weights
        return np.var(portfolio_returns)
    
    def _calculate_risk_premia_contribution(
        self, 
        weights: np.ndarray, 
        risk_premia: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Calculate contribution from alternative risk premia"""
        # Simplified mapping of portfolio weights to risk premia exposure
        # In practice, this would be more sophisticated
        total_premia = np.zeros(len(next(iter(risk_premia.values()))))
        
        for factor, factor_returns in risk_premia.items():
            # Weight premia by portfolio allocation
            factor_weight = np.mean(weights)  # Simplified
            total_premia += factor_weight * factor_returns
        
        return total_premia
    
    def _calculate_portfolio_metrics(
        self, 
        weights: np.ndarray, 
        returns: np.ndarray,
        risk_premia: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        portfolio_returns = returns @ weights
        
        # Add risk premia if provided
        if risk_premia is not None:
            premia_contrib = self._calculate_risk_premia_contribution(weights, risk_premia)
            portfolio_returns += premia_contrib
        
        metrics = {
            "expected_return": np.mean(portfolio_returns),
            "volatility": np.std(portfolio_returns),
            "sharpe_ratio": np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0,
            "semi_variance": self._semi_variance_objective(weights, returns),
            "var_95": np.percentile(portfolio_returns, 5),
            "cvar_95": self._cvar_objective(weights, returns),
            "max_weight": np.max(weights),
            "min_weight": np.min(weights),
            "concentration": np.sum(weights ** 2),  # Herfindahl index
            "skewness": self._calculate_skewness(portfolio_returns),
            "excess_kurtosis": self._calculate_excess_kurtosis(portfolio_returns)
        }
        
        # Behavioral metrics
        gains = portfolio_returns[portfolio_returns > 0]
        losses = portfolio_returns[portfolio_returns < 0]
        
        if len(gains) > 0 and len(losses) > 0:
            metrics["gain_loss_ratio"] = np.mean(gains) / abs(np.mean(losses))
            metrics["win_rate"] = len(gains) / len(portfolio_returns)
        
        return metrics
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate sample skewness"""
        if len(returns) < 3:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        n = len(returns)
        skew = (n / ((n-1) * (n-2))) * np.sum(((returns - mean_return) / std_return) ** 3)
        return skew
    
    def _calculate_excess_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate sample excess kurtosis"""
        if len(returns) < 4:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        n = len(returns)
        kurt = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((returns - mean_return) / std_return) ** 4)
        kurt -= 3 * (n-1)**2 / ((n-2) * (n-3))
        
        return kurt
    
    def generate_efficient_frontier(
        self,
        returns: np.ndarray,
        n_portfolios: int = 100,
        risk_premia: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate efficient frontier with asymmetric risk preferences
        
        Args:
            returns: Historical returns matrix
            n_portfolios: Number of portfolios on frontier
            risk_premia: Alternative risk premia
            
        Returns:
            Tuple of (expected_returns, risks, weights_matrix)
        """
        mean_returns = np.mean(returns, axis=0)
        min_return = np.min(mean_returns)
        max_return = np.max(mean_returns)
        
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        frontier_returns = []
        frontier_risks = []
        frontier_weights = []
        
        for target_return in target_returns:
            constraints = OptimizationConstraints(target_return=target_return)
            
            result = self.optimize_portfolio(
                returns=returns,
                risk_premia=risk_premia,
                constraints=constraints
            )
            
            if result["success"]:
                weights = result["weights"]
                metrics = result["metrics"]
                
                frontier_returns.append(metrics["expected_return"])
                
                # Use asymmetric risk measure
                if self.method == OptimizationMethod.SEMI_VARIANCE:
                    frontier_risks.append(np.sqrt(metrics["semi_variance"]))
                elif self.method == OptimizationMethod.CVAR:
                    frontier_risks.append(abs(metrics["cvar_95"]))
                else:
                    frontier_risks.append(metrics["volatility"])
                
                frontier_weights.append(weights)
        
        return (
            np.array(frontier_returns),
            np.array(frontier_risks),
            np.array(frontier_weights)
        )


def compare_optimization_methods(
    returns: np.ndarray,
    risk_premia: Optional[Dict[str, np.ndarray]] = None
) -> pd.DataFrame:
    """
    Compare different optimization methods
    
    Args:
        returns: Historical returns matrix
        risk_premia: Alternative risk premia
        
    Returns:
        DataFrame comparing optimization results
    """
    methods = [
        OptimizationMethod.MEAN_VARIANCE,
        OptimizationMethod.SEMI_VARIANCE,
        OptimizationMethod.CVAR,
        OptimizationMethod.BEHAVIORAL_UTILITY
    ]
    
    results = []
    
    for method in methods:
        optimizer = AsymmetricPortfolioOptimizer(method=method)
        result = optimizer.optimize_portfolio(returns, risk_premia)
        
        if result["success"]:
            metrics = result["metrics"]
            metrics["method"] = method.value
            metrics["weights"] = result["weights"]
            results.append(metrics)
    
    return pd.DataFrame(results)


# ...existing code for backward compatibility...

def semi_variance(returns):
    downside = returns[returns < 0]
    return np.var(downside) if len(downside) > 0 else 0

def portfolio_objective(weights, returns, method="semi"):
    portfolio_returns = returns @ weights
    if method == "semi":
        return semi_variance(portfolio_returns)
    elif method == "cvar":
        alpha = 0.05
        threshold = np.percentile(portfolio_returns, alpha * 100)
        tail_losses = portfolio_returns[portfolio_returns <= threshold]
        return -np.mean(tail_losses)  # minimize expected shortfall
    else:
        raise ValueError("Unknown method")

def optimize_portfolio(returns, method="semi"):
    n_assets = returns.shape[1]
    init_weights = np.ones(n_assets) / n_assets
    bounds = [(0, 1)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(portfolio_objective, init_weights,
                      args=(returns, method),
                      bounds=bounds,
                      constraints=constraints)

    return result.x
