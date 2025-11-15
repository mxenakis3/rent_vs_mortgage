"""
Layer-based scenario builder for rent vs mortgage simulations.

Converts high-level layer definitions into month-by-month timelines
that can be fed directly to calculation functions.
"""
import numpy as np
from typing import List, Dict, Any


def build_scenario_from_layers(layers: List[Dict[str, Any]], time_horizon_years: int) -> Dict[str, np.ndarray]:
    """
    Build a scenario timeline from layers.

    Each layer represents a period of time in a specific living situation:
    - 'buy': Own a property (may or may not live in it)
    - 'rental': Rent a place
    - 'home': Living at parents (implicit for unspecified months)

    Args:
        layers: List of layer configs
        time_horizon_years: Total simulation length in years

    Returns:
        Timeline dict with month-by-month arrays for all financial components
    """
    n_months = time_horizon_years * 12

    # Initialize timeline with defaults
    timeline = {
        'n_months': n_months,
        'living_location': np.full(n_months, 'home', dtype=object),
    }

    # Track which properties and rentals we have
    buy_layers = []
    rental_layers = []

    # Separate layers by type
    for i, layer in enumerate(layers):
        layer_type = layer.get('type')
        if layer_type == 'buy':
            buy_layers.append((i, layer))
        elif layer_type == 'rental':
            rental_layers.append((i, layer))

    # Process buy layers (properties)
    for prop_idx, (layer_id, layer) in enumerate(buy_layers):
        prop_name = f'property_{prop_idx}'
        params = layer.get('params', {})
        occupied_years = layer.get('occupied', [])

        # Get purchase timing
        purchase_year = params.get('purchase_year', 1)  # Default: buy in year 1
        purchase_month_idx = (purchase_year - 1) * 12

        # Get sale timing (optional)
        sale_year = params.get('sale_year', None)  # Default: never sell
        sale_month_idx = ((sale_year - 1) * 12) if sale_year else n_months

        # Ensure indices are within bounds
        purchase_month_idx = max(0, min(purchase_month_idx, n_months - 1))
        sale_month_idx = max(purchase_month_idx + 1, min(sale_month_idx, n_months))

        # Convert occupied years to month indices
        occupied_months = _years_to_month_indices(occupied_years, n_months)

        # Determine ownership period
        owns_months = np.zeros(n_months, dtype=bool)
        owns_months[purchase_month_idx:sale_month_idx] = True

        # Create arrays for this property
        timeline[f'{prop_name}_owns'] = owns_months
        timeline[f'{prop_name}_occupied'] = np.isin(np.arange(n_months), occupied_months) & owns_months
        timeline[f'{prop_name}_principal'] = np.full(n_months, params.get('principal', 0), dtype=float)
        timeline[f'{prop_name}_down_payment_pct'] = params.get('down_payment_pct', 0)
        timeline[f'{prop_name}_loan_years'] = params.get('loan_years', 0)
        timeline[f'{prop_name}_annual_mortgage_rate'] = params.get('annual_mortgage_rate', 0)
        timeline[f'{prop_name}_annual_appreciation_rate'] = params.get('annual_appreciation_rate', 0)
        timeline[f'{prop_name}_base_monthly_HOA_bill'] = params.get('base_monthly_HOA_bill', 0)
        timeline[f'{prop_name}_base_quarterly_tax_bill'] = params.get('base_quarterly_tax_bill', 0)
        timeline[f'{prop_name}_base_annual_insurance_bill'] = params.get('base_annual_insurance_bill', 0)
        timeline[f'{prop_name}_base_monthly_utility_bill'] = params.get('base_monthly_utility_bill', 0)
        timeline[f'{prop_name}_rent_monthly_income'] = params.get('rent_monthly_income', 0)
        timeline[f'{prop_name}_annual_rent_income_growth_rate'] = params.get('annual_rent_income_growth_rate', 0)
        timeline[f'{prop_name}_rent_income_vacancy_rate'] = params.get('rent_income_vacancy_rate', 0)
        timeline[f'{prop_name}_annual_inflation_rate'] = params.get('annual_inflation_rate', 0.03)

        # Update living location for occupied months (only if we own during those months)
        valid_occupied_months = occupied_months[occupied_months >= purchase_month_idx]
        valid_occupied_months = valid_occupied_months[valid_occupied_months < sale_month_idx]
        timeline['living_location'][valid_occupied_months] = f'{prop_name}_occupied'

    # Store count
    timeline['n_properties'] = len(buy_layers)

    # Process rental layers
    for rental_idx, (layer_id, layer) in enumerate(rental_layers):
        rental_name = f'rental_{rental_idx}'
        params = layer.get('params', {})
        occupied_years = layer.get('occupied', [])

        # Convert years to month indices
        occupied_months = _years_to_month_indices(occupied_years, n_months)

        # Create arrays for this rental
        timeline[f'{rental_name}_active'] = np.isin(np.arange(n_months), occupied_months)
        timeline[f'{rental_name}_monthly_payment'] = params.get('rent_monthly_payment', 0)
        timeline[f'{rental_name}_annual_growth_rate'] = params.get('annual_rent_growth_rate', 0)
        timeline[f'{rental_name}_base_monthly_utility_bill'] = params.get('base_monthly_utility_bill', 0)
        timeline[f'{rental_name}_base_annual_insurance_bill'] = params.get('base_annual_insurance_bill', 0)
        timeline[f'{rental_name}_annual_inflation_rate'] = params.get('annual_inflation_rate', 0.03)

        # Update living location for occupied months
        timeline['living_location'][occupied_months] = f'{rental_name}_active'

    # Store count
    timeline['n_rentals'] = len(rental_layers)

    return timeline


def _years_to_month_indices(years, n_months):
    """Convert year numbers/ranges to month indices."""
    months = []
    for year in years:
        if isinstance(year, range):
            for y in year:
                months.extend(range((y - 1) * 12, min(y * 12, n_months)))
        else:
            # Single year
            months.extend(range((year - 1) * 12, min(year * 12, n_months)))

    return np.array(months, dtype=int)
