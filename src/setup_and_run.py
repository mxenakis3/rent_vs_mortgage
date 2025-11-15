from collections import defaultdict
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
from calculations import *
from scenario_builder import build_scenario_from_layers

# Layer-based configs - define scenarios using layers
configs = {
    'global_configs': {
        'time_horizon_years': 20,
        'compound_annual_growth_rate': 0.07,  # expected return on investment
        'export_to_excel': True  # Toggle Excel/CSV export
    },
    'scenario_configs': [
        # SCENARIO 1: Buy and live in it for years 1-3, then rent it out while renting elsewhere
        {
            'name': 'Buy, occupy 1-3, then rent out',
            'layers': [
                {
                    'type': 'buy',
                    'occupied': [1, 2, 3],  # Live in the property years 1-3
                    'params': {
                        'purchase_year': 1,
                        'principal': 275000,
                        'down_payment_pct': 0.20,
                        'loan_years': 5,
                        'annual_mortgage_rate': 0.0625,
                        'annual_appreciation_rate': 0.0125,
                        'base_monthly_HOA_bill': 500,
                        'base_quarterly_tax_bill': 900,
                        'base_annual_insurance_bill': 1750,
                        'base_monthly_utility_bill': 200,
                        'rent_monthly_income': 2300,
                        'annual_rent_income_growth_rate': 0.03,
                        'rent_income_vacancy_rate': 0.05,
                        'annual_inflation_rate': 0.03,
                    }
                },
                {
                    'type': 'rental',
                    'occupied': list(range(4, 21)),  # Rent elsewhere years 4-20
                    'params': {
                        'rent_monthly_payment': 2200,
                        'annual_rent_growth_rate': 0.03,
                        'base_monthly_utility_bill': 150,
                        'base_annual_insurance_bill': 300,
                        'annual_inflation_rate': 0.03,
                    }
                },
            ]
        },

        # SCENARIO 2: Just rent the whole time
        {
            'name': 'Rent only',
            'layers': [
                {
                    'type': 'rental',
                    'occupied': list(range(1, 21)),  # Rent for all 20 years
                    'params': {
                        'rent_monthly_payment': 2200,
                        'annual_rent_growth_rate': 0.03,
                        'base_monthly_utility_bill': 150,
                        'base_annual_insurance_bill': 300,
                        'annual_inflation_rate': 0.03,
                    }
                },
            ]
        },

        # SCENARIO 3: Live at home for 2 years, then rent
        {
            'name': 'Live at home years 1-2, rent years 3-20',
            'layers': [
                {
                    'type': 'rental',
                    'occupied': list(range(3, 21)),  # Rent years 3-20
                    'params': {
                        'rent_monthly_payment': 2200,
                        'annual_rent_growth_rate': 0.03,
                        'base_monthly_utility_bill': 150,
                        'base_annual_insurance_bill': 300,
                        'annual_inflation_rate': 0.03,
                    }
                },
                # Years 1-2 implicitly "at home" (no layers = living at home, no costs)
            ]
        },

        # SCENARIO 4: Buy property, live at home while renting it out, then move in
        {
            'name': 'Buy year 1, live at home years 1-2, occupy years 3-5, rent out 6-20',
            'layers': [
                {
                    'type': 'buy',
                    'occupied': [3, 4, 5],  # Move into property years 3-5
                    'params': {
                        'purchase_year': 1,  # Buy in year 1
                        'principal': 275000,
                        'down_payment_pct': 0.20,
                        'loan_years': 5,
                        'annual_mortgage_rate': 0.0625,
                        'annual_appreciation_rate': 0.0125,
                        'base_monthly_HOA_bill': 500,
                        'base_quarterly_tax_bill': 900,
                        'base_annual_insurance_bill': 1750,
                        'base_monthly_utility_bill': 200,
                        'rent_monthly_income': 2300,
                        'annual_rent_income_growth_rate': 0.03,
                        'rent_income_vacancy_rate': 0.05,
                        'annual_inflation_rate': 0.03,
                    }
                },
                {
                    'type': 'rental',
                    'occupied': list(range(6, 21)),  # Rent elsewhere years 6-20
                    'params': {
                        'rent_monthly_payment': 2200,
                        'annual_rent_growth_rate': 0.03,
                        'base_monthly_utility_bill': 150,
                        'base_annual_insurance_bill': 300,
                        'annual_inflation_rate': 0.03,
                    }
                },
                # Years 1-2: Living at home (no layer), property renting out
            ]
        },

        # SCENARIO 5: Alternate between living at home and living in owned property
        {
            'name': 'Buy year 1, alternate home/property',
            'layers': [
                {
                    'type': 'buy',
                    'occupied': [1, 3, 5, 7, 9],  # Live in property odd years
                    'params': {
                        'purchase_year': 1,
                        'principal': 275000,
                        'down_payment_pct': 0.20,
                        'loan_years': 5,
                        'annual_mortgage_rate': 0.0625,
                        'annual_appreciation_rate': 0.0125,
                        'base_monthly_HOA_bill': 500,
                        'base_quarterly_tax_bill': 900,
                        'base_annual_insurance_bill': 1750,
                        'base_monthly_utility_bill': 200,
                        'rent_monthly_income': 2300,
                        'annual_rent_income_growth_rate': 0.03,
                        'rent_income_vacancy_rate': 0.05,
                        'annual_inflation_rate': 0.03,
                    }
                },
                # Years 2, 4, 6, 8, 10+: Living at home, renting out property
            ]
        },

        # SCENARIO 6: Buy property A, then buy property B while keeping A
        {
            'name': 'Multi-property: Buy A year 1, buy B year 6',
            'layers': [
                {
                    'type': 'buy',
                    'occupied': [1, 2, 3, 4, 5],  # Live in property A years 1-5
                    'params': {
                        'purchase_year': 1,
                        'principal': 275000,
                        'down_payment_pct': 0.20,
                        'loan_years': 5,
                        'annual_mortgage_rate': 0.0625,
                        'annual_appreciation_rate': 0.0125,
                        'base_monthly_HOA_bill': 500,
                        'base_quarterly_tax_bill': 900,
                        'base_annual_insurance_bill': 1750,
                        'base_monthly_utility_bill': 200,
                        'rent_monthly_income': 2300,
                        'annual_rent_income_growth_rate': 0.03,
                        'rent_income_vacancy_rate': 0.05,
                        'annual_inflation_rate': 0.03,
                    }
                },
                {
                    'type': 'buy',
                    'occupied': list(range(6, 21)),  # Live in property B years 6-20
                    'params': {
                        'purchase_year': 6,
                        'principal': 400000,
                        'down_payment_pct': 0.20,
                        'loan_years': 10,
                        'annual_mortgage_rate': 0.07,
                        'annual_appreciation_rate': 0.0125,
                        'base_monthly_HOA_bill': 600,
                        'base_quarterly_tax_bill': 1200,
                        'base_annual_insurance_bill': 2400,
                        'base_monthly_utility_bill': 250,
                        'rent_monthly_income': 3000,
                        'annual_rent_income_growth_rate': 0.03,
                        'rent_income_vacancy_rate': 0.05,
                        'annual_inflation_rate': 0.03,
                    }
                },
                # Property A rents out years 6-20 while living in B
            ]
        },
    ]
}


def export_results_to_excel(results_dict, config, scenario_names):
    """
    Export all results to Excel in a timestamped directory.

    Creates one Excel file with multiple sheets (one per metric).
    Each sheet has scenarios as rows and months as columns.
    """
    # Create timestamped directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('results', f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Create Excel file
    excel_path = os.path.join(output_dir, 'simulation_results.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Export each metric to a separate sheet
        for metric_name, data in results_dict.items():
            # Skip if not a numpy array
            if not isinstance(data, np.ndarray):
                continue

            # Create DataFrame with scenario names as index
            # Columns: month_1, month_2, ..., month_n
            n_months = data.shape[1]
            month_cols = [f'month_{i+1}' for i in range(n_months)]

            df = pd.DataFrame(
                data,
                index=scenario_names,
                columns=month_cols
            )

            # Add a total column
            df['TOTAL'] = df.sum(axis=1)

            # Write to sheet (truncate sheet name to 31 chars for Excel)
            sheet_name = metric_name[:31]
            df.to_excel(writer, sheet_name=sheet_name)

    print(f"\n{'='*60}")
    print(f"Results exported to: {output_dir}")
    print(f"Excel file: {excel_path}")
    print(f"{'='*60}\n")

    return output_dir


def plot_opportunity_adjusted_balance(results_dict, scenario_names, output_dir=None):
    """
    Plot cumulative opportunity-adjusted net balance for all scenarios.

    Args:
        results_dict: Dictionary of results from runner()
        scenario_names: List of scenario names
        output_dir: Optional directory to save plot (if None, just display)
    """
    # Get data
    net_balance = results_dict['net_balance']
    opportunity_adjusted = results_dict['opportunity_cost']
    n_scenarios, n_months = net_balance.shape

    # Calculate cumulative net balance and add the already-cumulative opportunity cost
    cumulative_net_balance = np.cumsum(net_balance, axis=1)
    cumulative_data = cumulative_net_balance + opportunity_adjusted

    # Convert months to years for x-axis
    months = np.arange(1, n_months + 1)
    years = months / 12

    # Create plot
    plt.figure(figsize=(12, 7))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    for i in range(n_scenarios):
        plt.plot(years, cumulative_data[i, :],
                label=scenario_names[i],
                linewidth=2.5,
                color=colors[i % len(colors)])

    # Formatting
    plt.xlabel('Years', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Net Worth ($)', fontsize=12, fontweight='bold')
    plt.title('Opportunity-Adjusted Net Balance Over Time', fontsize=14, fontweight='bold', pad=20)

    # Format y-axis with currency
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.legend(loc='best', fontsize=11, framealpha=0.9)

    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    # Save if output directory provided
    if output_dir:
        plot_path = os.path.join(output_dir, 'opportunity_adjusted_balance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

    plt.close()

    return None


def runner(config):
    """
    Vectorized runner - executes calculations from layer-based config.

    Builds timelines from layers, then runs all calculation functions.
    """
    time_horizon_years = config['global_configs']['time_horizon_years']
    scenario_configs = config['scenario_configs']
    n_scenarios = len(scenario_configs)
    n_months = time_horizon_years * 12

    # Build timelines from layers
    timelines = []
    scenario_names = []

    for scenario_config in scenario_configs:
        layers = scenario_config.get('layers', [])
        name = scenario_config.get('name', 'Unnamed scenario')

        # Build timeline from layers
        timeline = build_scenario_from_layers(layers, time_horizon_years)
        timelines.append(timeline)
        scenario_names.append(name)

    # Create M (month) and Y (year) arrays
    m = np.arange(1, n_months + 1)
    y = (m - 1) // 12 + 1
    M = np.broadcast_to(m, (n_scenarios, n_months))
    Y = np.broadcast_to(y, (n_scenarios, n_months))

    # Define calculations
    calculations = [
        ('rent', calc_rent),
        ('utilities', calc_utilities),
        ('insurance', calc_insurance),
        ('HOA', calc_HOA),
        ('tax', calc_tax),
        ('appreciation', calc_appreciation),
        ('rent_income', calc_rent_income),
        ('alternative_rent', calc_alternative_rent),
        ('mortgage', calc_mortgage),
        ('equity_buildup', calc_equity_buildup),
    ]

    # Execute each calculation
    results_dict = {}
    for name, calc_func in calculations:
        results_dict[name] = calc_func(timelines, M, Y)

    # Add summary calculations
    results_dict['net_gains'] = results_dict['appreciation'] + results_dict['rent_income'] + results_dict['equity_buildup']
    results_dict['net_losses'] = sum(results_dict[k] for k in ['rent', 'utilities', 'insurance', 'HOA', 'tax', 'mortgage', 'alternative_rent'])
    results_dict['net_balance'] = results_dict['net_gains'] + results_dict['net_losses']

    # Calculate cash flow only (excludes equity buildup and appreciation - non-cash gains)
    results_dict['cash_flow_gains'] = results_dict['rent_income']
    results_dict['cash_flow_balance'] = results_dict['cash_flow_gains'] + results_dict['net_losses']

    # Calculate opportunity cost using CASH FLOW only (not equity buildup or appreciation)
    compound_rate = config['global_configs']['compound_annual_growth_rate']
    results_dict['opportunity_cost'] = calculate_opportunity_cost(
        results_dict['cash_flow_balance'],
        compound_rate,
        n_months
    )

    # Add opportunity-adjusted net balance
    results_dict['net_balance_opportunity_adjusted'] = results_dict['net_balance'] + results_dict['opportunity_cost']

    return results_dict, scenario_names


if __name__ == '__main__':
    res_dict, scenario_names = runner(configs)

    # Export to Excel if enabled
    if configs['global_configs'].get('export_to_excel', False):
        output_dir = export_results_to_excel(res_dict, configs, scenario_names)
        plot_opportunity_adjusted_balance(res_dict, scenario_names, output_dir)
