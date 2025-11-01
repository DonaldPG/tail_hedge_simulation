#!/usr/bin/env python3
"""
Complete Asymmetric Portfolio Strategy Example

This example demonstrates the full end-to-end workflow using all enhanced modules:
1. Asymmetric simulation with regime switching
2. Alternative risk premia modeling
3. Behavioral investor profiling
4. Portfolio optimization with multiple objectives
5. Systematic hedging overlay
6. Performance attribution and analysis

Based on Deutsche Bank's asymmetric strategies framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import all enhanced modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from asymmetric_sim import (
    AsymmetricSimulator, AssetParameters, SimulationParameters, 
    create_default_assets, MarketRegime
)
from behavioral_utility import (
    InvestorBehavior, InvestorType, create_default_risk_tolerance,
    create_default_behavioral_parameters
)
from optimizer import (
    AsymmetricPortfolioOptimizer, OptimizationMethod, 
    OptimizationConstraints, compare_optimization_methods
)
from risk_premia import AlternativeRiskPremia, RiskPremiaFactor
from systematic_hedging import (
    SystematicHedging, HedgeType, MarketModel,
    create_default_option_parameters
)


class CompleteAsymmetricStrategy:
    """
    Complete implementation of asymmetric portfolio strategy
    
    Integrates all components for end-to-end portfolio construction
    with asymmetric risk preferences and tail hedging
    """
    
    def __init__(
        self,
        investor_type: InvestorType = InvestorType.STRATEGIC_ALLOCATOR,
        hedge_type: HedgeType = HedgeType.PUT_PROTECTION,
        portfolio_value: float = 1000000
    ):
        """
        Initialize complete strategy system
        
        Args:
            investor_type: Type of investor behavior to model
            hedge_type: Type of systematic hedging to apply
            portfolio_value: Initial portfolio value
        """
        self.investor_type = investor_type
        self.hedge_type = hedge_type
        self.portfolio_value = portfolio_value
        
        # Initialize all components
        self._setup_components()
        
        logger.info(f"Initialized complete strategy for {investor_type.value} with {hedge_type.value} hedging")
    
    def _setup_components(self):
        """Initialize all system components"""
        
        # 1. Asset simulation setup
        self.assets = create_default_assets()
        self.sim_params = SimulationParameters(
            num_simulations=5000,
            num_days=252,
            random_seed=42
        )
        self.simulator = AsymmetricSimulator(self.assets, self.sim_params)
        
        # 2. Behavioral modeling setup
        self.risk_tolerance = create_default_risk_tolerance(self.investor_type)
        self.behavioral_params = create_default_behavioral_parameters()
        self.investor = InvestorBehavior(
            self.investor_type,
            self.risk_tolerance,
            self.behavioral_params
        )
        
        # 3. Risk premia modeling
        self.risk_premia_model = AlternativeRiskPremia()
        
        # 4. Portfolio optimizer
        self.optimizer = AsymmetricPortfolioOptimizer(
            method=OptimizationMethod.BEHAVIORAL_UTILITY,
            loss_aversion=self.behavioral_params.loss_aversion
        )
        
        # 5. Systematic hedging
        self.market_model = MarketModel()
        self.hedge_strategy = SystematicHedging(
            hedge_type=self.hedge_type,
            option_params=create_default_option_parameters(self.hedge_type),
            market_model=self.market_model,
            portfolio_value=self.portfolio_value
        )
        
        # Results storage
        self.results = {}
    
    def run_complete_analysis(self, save_results: bool = True) -> Dict:
        """
        Run complete asymmetric portfolio analysis
        
        Args:
            save_results: Whether to save results to files
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting complete asymmetric portfolio analysis...")
        
        # Step 1: Generate alternative risk premia
        logger.info("Step 1: Generating alternative risk premia...")
        factor_returns = self.risk_premia_model.simulate_risk_premia(
            n_periods=self.sim_params.num_days,
            seed=42
        )
        self.results['factor_returns'] = factor_returns
        
        # Step 2: Simulate asset paths with asymmetric features
        logger.info("Step 2: Simulating asset paths...")
        asset_paths = self.simulator.simulate_asset_paths()
        self.results['asset_paths'] = asset_paths
        
        # Calculate asset returns for optimization
        asset_returns = self._calculate_returns_from_paths(asset_paths)
        self.results['asset_returns'] = asset_returns
        
        # Step 3: Compare optimization methods
        logger.info("Step 3: Comparing optimization methods...")
        optimization_comparison = compare_optimization_methods(
            asset_returns, 
            factor_returns.to_dict('series')
        )
        self.results['optimization_comparison'] = optimization_comparison
        
        # Step 4: Optimize portfolio with behavioral preferences
        logger.info("Step 4: Optimizing portfolio with behavioral preferences...")
        optimal_result = self.optimizer.optimize_portfolio(
            returns=asset_returns,
            risk_premia=factor_returns.to_dict('series'),
            constraints=OptimizationConstraints(max_concentration=0.4)
        )
        self.results['optimal_portfolio'] = optimal_result
        
        # Step 5: Simulate portfolio paths
        logger.info("Step 5: Simulating portfolio paths...")
        if optimal_result['success']:
            # Map asset weights to asset names
            asset_weights = {
                asset.name: weight 
                for asset, weight in zip(self.assets, optimal_result['weights'])
            }
            
            portfolio_paths, _ = self.simulator.simulate_portfolio_paths(asset_weights)
            self.results['portfolio_paths'] = portfolio_paths
            
            # Calculate unhedged portfolio metrics
            unhedged_metrics = self.simulator.calculate_risk_metrics(portfolio_paths)
            self.results['unhedged_metrics'] = unhedged_metrics
        
        # Step 6: Apply systematic hedging
        logger.info("Step 6: Applying systematic hedging...")
        hedged_analysis = self._analyze_hedging_strategy(asset_weights, portfolio_paths)
        self.results['hedging_analysis'] = hedged_analysis
        
        # Step 7: Behavioral analysis
        logger.info("Step 7: Analyzing behavioral responses...")
        behavioral_analysis = self._analyze_behavioral_responses(portfolio_paths)
        self.results['behavioral_analysis'] = behavioral_analysis
        
        # Step 8: Factor attribution
        logger.info("Step 8: Performing factor attribution...")
        attribution_analysis = self._perform_factor_attribution(asset_returns, factor_returns)
        self.results['attribution_analysis'] = attribution_analysis
        
        # Step 9: Generate comprehensive report
        logger.info("Step 9: Generating comprehensive report...")
        self._generate_comprehensive_report()
        
        if save_results:
            self._save_results()
        
        logger.info("Complete analysis finished!")
        return self.results
    
    def _calculate_returns_from_paths(self, asset_paths: Dict) -> np.ndarray:
        """Calculate returns matrix from simulated asset paths"""
        returns_list = []
        
        for asset_name in asset_paths:
            paths = asset_paths[asset_name]
            # Calculate returns from first simulation path
            prices = paths[0, :]
            returns = np.diff(prices) / prices[:-1]
            returns_list.append(returns)
        
        return np.column_stack(returns_list)
    
    def _analyze_hedging_strategy(self, asset_weights: Dict, portfolio_paths: np.ndarray) -> Dict:
        """Analyze systematic hedging strategy effectiveness"""
        
        # Calculate hedging costs and payoffs
        hedging_results = {
            'costs': [],
            'payoffs': [],
            'net_portfolio_values': []
        }
        
        num_sims, num_periods = portfolio_paths.shape
        
        # Sample analysis on subset of simulations
        sample_size = min(100, num_sims)
        
        for sim in range(sample_size):
            path = portfolio_paths[sim, :]
            
            # Calculate hedge cost at beginning
            hedge_cost = self.hedge_strategy.calculate_hedge_cost(
                spot_price=100,  # Normalized price
                portfolio_allocation={"equity": 0.6, "bonds": 0.4}  # Simplified
            )
            
            # Calculate hedge payoff at end
            portfolio_return = (path[-1] / path[0]) - 1
            hedge_payoff = self.hedge_strategy.calculate_hedge_payoff(
                initial_price=100,
                final_price=100 * (1 + portfolio_return),
                portfolio_return=portfolio_return,
                portfolio_allocation={"equity": 0.6, "bonds": 0.4}
            )
            
            # Net portfolio value with hedging
            net_value = path[-1] * (1 - hedge_cost + hedge_payoff)
            
            hedging_results['costs'].append(hedge_cost)
            hedging_results['payoffs'].append(hedge_payoff)
            hedging_results['net_portfolio_values'].append(net_value)
        
        # Calculate hedge effectiveness metrics
        unhedged_final_values = portfolio_paths[:sample_size, -1]
        hedged_final_values = np.array(hedging_results['net_portfolio_values'])
        
        effectiveness = {
            'average_cost': np.mean(hedging_results['costs']),
            'average_payoff': np.mean(hedging_results['payoffs']),
            'downside_protection': self._calculate_downside_protection(
                unhedged_final_values, hedged_final_values
            ),
            'hit_ratio': np.sum(hedged_final_values > unhedged_final_values) / len(hedged_final_values)
        }
        
        hedging_results['effectiveness'] = effectiveness
        return hedging_results
    
    def _calculate_downside_protection(self, unhedged: np.ndarray, hedged: np.ndarray) -> float:
        """Calculate downside protection provided by hedging"""
        unhedged_losses = unhedged[unhedged < 100]  # Losses below initial value
        hedged_losses = hedged[hedged < 100]
        
        if len(unhedged_losses) == 0:
            return 0.0
        
        avg_unhedged_loss = np.mean(100 - unhedged_losses)
        avg_hedged_loss = np.mean(100 - hedged_losses) if len(hedged_losses) > 0 else 0
        
        return 1 - (avg_hedged_loss / avg_unhedged_loss) if avg_unhedged_loss > 0 else 0
    
    def _analyze_behavioral_responses(self, portfolio_paths: np.ndarray) -> Dict:
        """Analyze behavioral responses to portfolio performance"""
        
        behavioral_states = []
        allocation_changes = []
        hedging_preferences = []
        
        # Analyze sample of paths
        sample_size = min(50, portfolio_paths.shape[0])
        
        for sim in range(sample_size):
            path = portfolio_paths[sim, :]
            
            # Simulate performance updates
            for i in range(1, len(path)):
                period_return = (path[i] / path[i-1]) - 1
                self.investor.update_performance(period_return)
                
                behavioral_states.append(self.investor.behavioral_state)
                hedging_preferences.append(self.investor.hedging_preference())
        
        return {
            'state_distribution': pd.Series(behavioral_states).value_counts(),
            'average_hedging_preference': np.mean(hedging_preferences),
            'stress_periods': sum(1 for state in behavioral_states if state in ['stressed', 'panic']),
            'total_periods': len(behavioral_states)
        }
    
    def _perform_factor_attribution(self, asset_returns: np.ndarray, factor_returns: pd.DataFrame) -> Dict:
        """Perform factor attribution analysis"""
        
        # Create a portfolio return series for attribution
        if 'optimal_portfolio' in self.results and self.results['optimal_portfolio']['success']:
            weights = self.results['optimal_portfolio']['weights']
            portfolio_returns = asset_returns @ weights
            
            # Perform attribution
            attribution = self.risk_premia_model.factor_attribution(
                portfolio_returns=portfolio_returns,
                factor_returns=factor_returns,
                window=60
            )
            
            # Calculate factor statistics
            factor_stats = self.risk_premia_model.calculate_factor_statistics(factor_returns)
            
            return {
                'attribution': attribution,
                'factor_statistics': factor_stats,
                'portfolio_returns': portfolio_returns
            }
        
        return {}
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        
        print("\n" + "="*80)
        print("🎯 ASYMMETRIC PORTFOLIO STRATEGY - COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        # Portfolio Optimization Results
        if 'optimal_portfolio' in self.results:
            result = self.results['optimal_portfolio']
            if result['success']:
                print(f"\n📊 OPTIMAL PORTFOLIO ALLOCATION ({self.optimizer.method.value}):")
                print("-" * 50)
                for i, (asset, weight) in enumerate(zip(self.assets, result['weights'])):
                    print(f"{asset.name:20s}: {weight:7.1%}")
                
                metrics = result['metrics']
                print(f"\n📈 PORTFOLIO METRICS:")
                print("-" * 30)
                print(f"Expected Return:     {metrics['expected_return']:7.2%}")
                print(f"Volatility:          {metrics['volatility']:7.2%}")
                print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:7.2f}")
                print(f"Semi-Variance:       {metrics['semi_variance']:7.4f}")
                print(f"VaR (95%):           {metrics['var_95']:7.2%}")
                print(f"CVaR (95%):          {metrics['cvar_95']:7.2%}")
                print(f"Skewness:            {metrics['skewness']:7.2f}")
                print(f"Excess Kurtosis:     {metrics['excess_kurtosis']:7.2f}")
        
        # Risk Metrics Comparison
        if 'unhedged_metrics' in self.results:
            print(f"\n🎲 UNHEDGED PORTFOLIO RISK METRICS:")
            print("-" * 40)
            metrics = self.results['unhedged_metrics']
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 'return' in metric or 'var' in metric or 'shortfall' in metric:
                        print(f"{metric:25s}: {value:7.2%}")
                    else:
                        print(f"{metric:25s}: {value:7.4f}")
        
        # Hedging Analysis
        if 'hedging_analysis' in self.results:
            hedge_results = self.results['hedging_analysis']
            if 'effectiveness' in hedge_results:
                eff = hedge_results['effectiveness']
                print(f"\n🛡️ SYSTEMATIC HEDGING ANALYSIS ({self.hedge_type.value}):")
                print("-" * 55)
                print(f"Average Cost:        {eff['average_cost']:7.2%}")
                print(f"Average Payoff:      {eff['average_payoff']:7.2%}")
                print(f"Downside Protection: {eff['downside_protection']:7.1%}")
                print(f"Hit Ratio:           {eff['hit_ratio']:7.1%}")
        
        # Behavioral Analysis
        if 'behavioral_analysis' in self.results:
            behavioral = self.results['behavioral_analysis']
            print(f"\n🧠 BEHAVIORAL ANALYSIS ({self.investor_type.value}):")
            print("-" * 50)
            print(f"Avg Hedging Preference: {behavioral['average_hedging_preference']:7.1%}")
            print(f"Stress Periods:         {behavioral['stress_periods']:7d}")
            print(f"Total Periods:          {behavioral['total_periods']:7d}")
            print(f"Stress Ratio:           {behavioral['stress_periods']/behavioral['total_periods']:7.1%}")
        
        # Factor Analysis
        if 'attribution_analysis' in self.results and self.results['attribution_analysis']:
            factor_stats = self.results['attribution_analysis']['factor_statistics']
            print(f"\n📊 ALTERNATIVE RISK PREMIA ANALYSIS:")
            print("-" * 45)
            for _, factor in factor_stats.iterrows():
                print(f"{factor['Factor']:12s}: {factor['Annual_Return']:6.1%} return, "
                      f"{factor['Annual_Volatility']:6.1%} vol, "
                      f"{factor['Sharpe_Ratio']:5.2f} Sharpe")
        
        # Optimization Method Comparison
        if 'optimization_comparison' in self.results:
            comp = self.results['optimization_comparison']
            print(f"\n⚖️ OPTIMIZATION METHOD COMPARISON:")
            print("-" * 40)
            for _, method in comp.iterrows():
                print(f"{method['method']:18s}: {method['expected_return']:6.2%} return, "
                      f"{method['volatility']:6.2%} vol")
        
        print("\n" + "="*80)
        print("✅ Analysis Complete")
        print("="*80)
    
    def _save_results(self):
        """Save results to files"""
        # Create results directory
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save key results
        if 'factor_returns' in self.results:
            self.results['factor_returns'].to_csv(
                os.path.join(results_dir, 'factor_returns.csv')
            )
        
        if 'optimization_comparison' in self.results:
            self.results['optimization_comparison'].to_csv(
                os.path.join(results_dir, 'optimization_comparison.csv')
            )
        
        logger.info(f"Results saved to {results_dir}")
    
    def create_visualization_dashboard(self):
        """Create comprehensive visualization dashboard"""
        
        if not self.results:
            logger.warning("No results to visualize. Run analysis first.")
            return
        
        # Set up the plotting style
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Portfolio Paths Simulation
        if 'portfolio_paths' in self.results:
            ax1 = plt.subplot(3, 3, 1)
            paths = self.results['portfolio_paths']
            
            # Plot sample paths
            for i in range(min(100, paths.shape[0])):
                ax1.plot(paths[i, :], alpha=0.1, color='blue', linewidth=0.5)
            
            # Plot mean and percentiles
            mean_path = np.mean(paths, axis=0)
            p5 = np.percentile(paths, 5, axis=0)
            p95 = np.percentile(paths, 95, axis=0)
            
            ax1.plot(mean_path, color='red', linewidth=2, label='Mean Path')
            ax1.fill_between(range(len(p5)), p5, p95, alpha=0.3, color='gray', label='5th-95th Percentile')
            ax1.set_title('Portfolio Value Simulation')
            ax1.set_xlabel('Trading Days')
            ax1.set_ylabel('Portfolio Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Factor Returns Correlation
        if 'factor_returns' in self.results:
            ax2 = plt.subplot(3, 3, 2)
            corr_matrix = self.results['factor_returns'].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax2)
            ax2.set_title('Factor Correlation Matrix')
        
        # 3. Optimization Method Comparison
        if 'optimization_comparison' in self.results:
            ax3 = plt.subplot(3, 3, 3)
            comp = self.results['optimization_comparison']
            
            x = np.arange(len(comp))
            width = 0.35
            
            ax3.bar(x - width/2, comp['expected_return'], width, label='Expected Return', alpha=0.8)
            ax3.bar(x + width/2, comp['volatility'], width, label='Volatility', alpha=0.8)
            
            ax3.set_xlabel('Optimization Method')
            ax3.set_ylabel('Value')
            ax3.set_title('Optimization Method Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(comp['method'], rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Portfolio Allocation Pie Chart
        if 'optimal_portfolio' in self.results and self.results['optimal_portfolio']['success']:
            ax4 = plt.subplot(3, 3, 4)
            weights = self.results['optimal_portfolio']['weights']
            asset_names = [asset.name for asset in self.assets]
            
            ax4.pie(weights, labels=asset_names, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Optimal Portfolio Allocation')
        
        # 5. Risk Metrics Bar Chart
        if 'unhedged_metrics' in self.results:
            ax5 = plt.subplot(3, 3, 5)
            metrics = self.results['unhedged_metrics']
            
            risk_metrics = {k: v for k, v in metrics.items() 
                           if k in ['volatility', 'max_drawdown', 'var_95', 'expected_shortfall_95']}
            
            bars = ax5.bar(range(len(risk_metrics)), [abs(v) for v in risk_metrics.values()])
            ax5.set_xlabel('Risk Metrics')
            ax5.set_ylabel('Value')
            ax5.set_title('Portfolio Risk Metrics')
            ax5.set_xticks(range(len(risk_metrics)))
            ax5.set_xticklabels(risk_metrics.keys(), rotation=45)
            ax5.grid(True, alpha=0.3)
            
            # Color code negative metrics
            for i, (k, v) in enumerate(risk_metrics.items()):
                if v < 0:
                    bars[i].set_color('red')
        
        # 6. Factor Performance
        if 'attribution_analysis' in self.results and self.results['attribution_analysis']:
            ax6 = plt.subplot(3, 3, 6)
            factor_stats = self.results['attribution_analysis']['factor_statistics']
            
            x = np.arange(len(factor_stats))
            ax6.scatter(factor_stats['Annual_Volatility'], factor_stats['Annual_Return'], 
                       s=100, alpha=0.7, c=factor_stats['Sharpe_Ratio'], cmap='viridis')
            
            for i, factor in factor_stats.iterrows():
                ax6.annotate(factor['Factor'], 
                           (factor['Annual_Volatility'], factor['Annual_Return']))
            
            ax6.set_xlabel('Annual Volatility')
            ax6.set_ylabel('Annual Return')
            ax6.set_title('Risk-Return Profile of Factors')
            ax6.grid(True, alpha=0.3)
        
        # 7. Hedging Effectiveness
        if 'hedging_analysis' in self.results and 'effectiveness' in self.results['hedging_analysis']:
            ax7 = plt.subplot(3, 3, 7)
            eff = self.results['hedging_analysis']['effectiveness']
            
            metrics = ['Cost', 'Payoff', 'Protection', 'Hit Ratio']
            values = [eff['average_cost'], eff['average_payoff'], 
                     eff['downside_protection'], eff['hit_ratio']]
            
            bars = ax7.bar(metrics, values)
            ax7.set_title(f'Hedging Effectiveness ({self.hedge_type.value})')
            ax7.set_ylabel('Value')
            ax7.grid(True, alpha=0.3)
            
            # Color code
            for i, bar in enumerate(bars):
                if i == 0:  # Cost
                    bar.set_color('red')
                elif i == 1:  # Payoff
                    bar.set_color('green')
                else:
                    bar.set_color('blue')
        
        # 8. Behavioral State Distribution
        if 'behavioral_analysis' in self.results:
            ax8 = plt.subplot(3, 3, 8)
            behavioral = self.results['behavioral_analysis']
            
            if 'state_distribution' in behavioral:
                state_dist = behavioral['state_distribution']
                ax8.pie(state_dist.values, labels=state_dist.index, autopct='%1.1f%%')
                ax8.set_title(f'Behavioral State Distribution\n({self.investor_type.value})')
        
        # 9. Return Distribution
        if 'portfolio_paths' in self.results:
            ax9 = plt.subplot(3, 3, 9)
            paths = self.results['portfolio_paths']
            final_returns = (paths[:, -1] / paths[:, 0]) - 1
            
            ax9.hist(final_returns, bins=50, alpha=0.7, density=True)
            ax9.axvline(np.mean(final_returns), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(final_returns):.2%}')
            ax9.axvline(np.percentile(final_returns, 5), color='orange', linestyle='--', 
                       label=f'5th Percentile: {np.percentile(final_returns, 5)::.2%}')
            
            ax9.set_xlabel('Total Return')
            ax9.set_ylabel('Density')
            ax9.set_title('Portfolio Return Distribution')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save the plot
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'comprehensive_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        
        logger.info("Visualization dashboard created and saved")

def plot_payoff_distribution(payoffs, title):
    """Plot the distribution of payoffs."""
    plt.style.use("dark_background")
    plt.figure(figsize=(12, 7))
    
    # Use matplotlib histplot
    plt.hist(payoffs, bins=50, density=True, alpha=0.7, label="Payoff Distribution")
    
    # Fit a normal distribution to the data
    mu, std = norm.fit(payoffs)
    
    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2, label="Fit results")
    
    plt.title(title)
    plt.xlabel("Payoff")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def main():
    """
    Main function to run complete asymmetric portfolio strategy example
    """
    print("🚀 Starting Complete Asymmetric Portfolio Strategy Analysis")
    print("=" * 70)
    
    # Example 1: Strategic Allocator with Put Protection
    print("\n📊 EXAMPLE 1: Strategic Allocator with Put Protection")
    print("-" * 55)
    
    strategy1 = CompleteAsymmetricStrategy(
        investor_type=InvestorType.STRATEGIC_ALLOCATOR,
        hedge_type=HedgeType.PUT_PROTECTION,
        portfolio_value=1000000
    )
    
    results1 = strategy1.run_complete_analysis(save_results=True)
    
    # Example 2: Market Timer with VIX Calls
    print("\n📊 EXAMPLE 2: Market Timer with VIX Calls")
    print("-" * 45)
    
    strategy2 = CompleteAsymmetricStrategy(
        investor_type=InvestorType.MARKET_TIMER,
        hedge_type=HedgeType.VIX_CALLS,
        portfolio_value=1000000
    )
    
    results2 = strategy2.run_complete_analysis(save_results=False)
    
    # Create comprehensive visualizations
    print("\n📈 Creating Visualization Dashboard...")
    strategy1.create_visualization_dashboard()
    
    print("\n✅ Complete analysis finished!")
    print("📁 Results saved to: results/")
    print("📊 Visualization saved to: results/comprehensive_analysis.png")
    
    return results1, results2


if __name__ == "__main__":
    # Set PYTHONPATH to include src directory
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    # Run main analysis
    results1, results2 = main()