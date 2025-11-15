"""
Vectorized calculation functions for rent vs mortgage simulations.

DATA STRUCTURE:
---------------
timeline: List of dictionaries, one per scenario
    - Length of list = number of scenarios being compared
    - Each dict contains one scenario's complete timeline over all months
    - Keys in each dict: 'property_0_owns', 'rental_0_active', etc.
    - Values: Arrays of length n_months with month-by-month data

Example:
    timeline = [
        # Scenario 0: Buy a condo
        {
            'n_months': 240,
            'n_properties': 1,
            'property_0_owns': array([False, True, True, ...]),      # (240,)
            'property_0_occupied': array([False, True, False, ...]), # (240,)
            'property_0_principal': 275000,
            ...
        },
        # Scenario 1: Just rent
        {
            'n_months': 240,
            'n_properties': 0,
            'n_rentals': 1,
            'rental_0_active': array([False, False, True, ...]),     # (240,)
            ...
        },
    ]

M, Y: Arrays of shape (n_scenarios, n_months)
    - M[i, j] = month number (1 to n_months) for scenario i, month j
    - Y[i, j] = year number (1 to n_years) for scenario i, month j

Return: Array of shape (n_scenarios, n_months) with cash flows for each scenario/month
"""
import numpy as np


def calc_rent(timeline, M, Y):
    """
    Calculate rent payments from rental layers with growth.

    Args:
        timeline: List of dicts, one per scenario (see module docstring)
        M: Month indices, shape (n_scenarios, n_months)
        Y: Year indices, shape (n_scenarios, n_months)

    Returns:
        Array of shape (n_scenarios, n_months) with rental payments (negative values)
    """
    n_scenarios = len(timeline) if isinstance(timeline, list) else 1
    timeline_list = timeline if isinstance(timeline, list) else [timeline]

    n_months = timeline_list[0]['n_months']
    result = np.zeros((n_scenarios, n_months))

    # Loop through each scenario
    for scen_idx, tl in enumerate(timeline_list):
        n_rentals = tl.get('n_rentals', 0)

        # Sum rent from all rental layers in this scenario
        for rental_idx in range(n_rentals):
            rental_name = f'rental_{rental_idx}'
            active = tl[f'{rental_name}_active']  # Boolean array: (n_months,)
            monthly_payment = tl[f'{rental_name}_monthly_payment']
            growth_rate = tl[f'{rental_name}_annual_growth_rate']

            # Calculate rent with growth over time
            rent_payment = -monthly_payment * (1 + growth_rate)**((M[scen_idx] - 1) / 12)

            # Only pay when rental is active
            result[scen_idx] += rent_payment * active

    return result


def calc_utilities(timeline, M, Y):
    """
    Calculate utilities with inflation - paid when occupying property or rental.

    Args:
        timeline: List of dicts, one per scenario
        M: Month indices, shape (n_scenarios, n_months)
        Y: Year indices, shape (n_scenarios, n_months)

    Returns:
        Array of shape (n_scenarios, n_months) with utility payments (negative values)
    """
    n_scenarios = len(timeline) if isinstance(timeline, list) else 1
    timeline_list = timeline if isinstance(timeline, list) else [timeline]

    n_months = timeline_list[0]['n_months']
    result = np.zeros((n_scenarios, n_months))

    for scen_idx, tl in enumerate(timeline_list):
        # Utilities from owned properties (only when occupied)
        n_properties = tl.get('n_properties', 0)
        for prop_idx in range(n_properties):
            prop_name = f'property_{prop_idx}'
            occupied = tl[f'{prop_name}_occupied']  # Boolean array: (n_months,)
            utility_bill = tl[f'{prop_name}_base_monthly_utility_bill']
            inflation_rate = tl[f'{prop_name}_annual_inflation_rate']

            utilities = -utility_bill * (1 + inflation_rate)**Y[scen_idx]
            result[scen_idx] += utilities * occupied

        # Utilities from rentals (when active)
        n_rentals = tl.get('n_rentals', 0)
        for rental_idx in range(n_rentals):
            rental_name = f'rental_{rental_idx}'
            active = tl[f'{rental_name}_active']  # Boolean array: (n_months,)
            utility_bill = tl[f'{rental_name}_base_monthly_utility_bill']
            inflation_rate = tl[f'{rental_name}_annual_inflation_rate']

            utilities = -utility_bill * (1 + inflation_rate)**Y[scen_idx]
            result[scen_idx] += utilities * active

    return result


def calc_insurance(timeline, M, Y):
    """
    Calculate insurance (paid annually).
    - Property insurance: paid when owned (regardless of occupation)
    - Renter's insurance: paid when renting

    Args:
        timeline: List of dicts, one per scenario
        M: Month indices, shape (n_scenarios, n_months)
        Y: Year indices, shape (n_scenarios, n_months)

    Returns:
        Array of shape (n_scenarios, n_months) with insurance payments (negative values)
    """
    n_scenarios = len(timeline) if isinstance(timeline, list) else 1
    timeline_list = timeline if isinstance(timeline, list) else [timeline]

    n_months = timeline_list[0]['n_months']
    result = np.zeros((n_scenarios, n_months))
    annual_mask = ((M[0] - 1) % 12 == 0).astype(float)

    for scen_idx, tl in enumerate(timeline_list):
        # Property insurance (paid when owned, regardless of occupation)
        n_properties = tl.get('n_properties', 0)
        for prop_idx in range(n_properties):
            prop_name = f'property_{prop_idx}'
            owns = tl[f'{prop_name}_owns']  # Boolean array: (n_months,)
            insurance_bill = tl[f'{prop_name}_base_annual_insurance_bill']
            inflation_rate = tl[f'{prop_name}_annual_inflation_rate']

            insurance = -annual_mask * insurance_bill * (1 + inflation_rate)**Y[scen_idx]
            result[scen_idx] += insurance * owns

        # Renter's insurance (paid when renting)
        n_rentals = tl.get('n_rentals', 0)
        for rental_idx in range(n_rentals):
            rental_name = f'rental_{rental_idx}'
            active = tl[f'{rental_name}_active']  # Boolean array: (n_months,)
            insurance_bill = tl[f'{rental_name}_base_annual_insurance_bill']
            inflation_rate = tl[f'{rental_name}_annual_inflation_rate']

            insurance = -annual_mask * insurance_bill * (1 + inflation_rate)**Y[scen_idx]
            result[scen_idx] += insurance * active

    return result


def calc_HOA(timeline, M, Y):
    """
    Calculate HOA fees with inflation - paid when owned.

    Args:
        timeline: List of dicts, one per scenario
        M: Month indices, shape (n_scenarios, n_months)
        Y: Year indices, shape (n_scenarios, n_months)

    Returns:
        Array of shape (n_scenarios, n_months) with HOA payments (negative values)
    """
    n_scenarios = len(timeline) if isinstance(timeline, list) else 1
    timeline_list = timeline if isinstance(timeline, list) else [timeline]

    n_months = timeline_list[0]['n_months']
    result = np.zeros((n_scenarios, n_months))

    for scen_idx, tl in enumerate(timeline_list):
        n_properties = tl.get('n_properties', 0)

        for prop_idx in range(n_properties):
            prop_name = f'property_{prop_idx}'
            owns = tl[f'{prop_name}_owns']  # Boolean array: (n_months,)
            hoa_bill = tl[f'{prop_name}_base_monthly_HOA_bill']
            inflation_rate = tl[f'{prop_name}_annual_inflation_rate']

            hoa = -hoa_bill * (1 + inflation_rate)**Y[scen_idx]
            result[scen_idx] += hoa * owns

    return result


def calc_tax(timeline, M, Y):
    """
    Calculate property tax (paid quarterly) - paid when owned.

    Args:
        timeline: List of dicts, one per scenario
        M: Month indices, shape (n_scenarios, n_months)
        Y: Year indices, shape (n_scenarios, n_months)

    Returns:
        Array of shape (n_scenarios, n_months) with tax payments (negative values)
    """
    n_scenarios = len(timeline) if isinstance(timeline, list) else 1
    timeline_list = timeline if isinstance(timeline, list) else [timeline]

    n_months = timeline_list[0]['n_months']
    result = np.zeros((n_scenarios, n_months))
    quarterly_mask = ((M[0] - 1) % 4 == 0).astype(float)

    for scen_idx, tl in enumerate(timeline_list):
        n_properties = tl.get('n_properties', 0)

        for prop_idx in range(n_properties):
            prop_name = f'property_{prop_idx}'
            owns = tl[f'{prop_name}_owns']  # Boolean array: (n_months,)
            tax_bill = tl[f'{prop_name}_base_quarterly_tax_bill']
            inflation_rate = tl[f'{prop_name}_annual_inflation_rate']

            tax = -quarterly_mask * tax_bill * (1 + inflation_rate)**Y[scen_idx]
            result[scen_idx] += tax * owns

    return result


def calc_appreciation(timeline, M, Y):
    """
    Calculate monthly appreciation (incremental gain) for all owned properties.

    Args:
        timeline: List of dicts, one per scenario
        M: Month indices, shape (n_scenarios, n_months)
        Y: Year indices, shape (n_scenarios, n_months)

    Returns:
        Array of shape (n_scenarios, n_months) with appreciation gains (positive values)
    """
    n_scenarios = len(timeline) if isinstance(timeline, list) else 1
    timeline_list = timeline if isinstance(timeline, list) else [timeline]

    n_months = timeline_list[0]['n_months']
    result = np.zeros((n_scenarios, n_months))

    for scen_idx, tl in enumerate(timeline_list):
        n_properties = tl.get('n_properties', 0)

        for prop_idx in range(n_properties):
            prop_name = f'property_{prop_idx}'
            owns = tl[f'{prop_name}_owns']  # Boolean array: (n_months,)
            principal = tl[f'{prop_name}_principal'][0]  # Get scalar value from array
            appreciation_rate = tl[f'{prop_name}_annual_appreciation_rate']

            # Home value at each month
            home_value = principal * (1 + appreciation_rate)**(M[scen_idx] / 12)

            # Previous month's value
            prev_value = np.concatenate([np.array([principal]), home_value[:-1]])

            # Monthly appreciation gain (only when owned)
            monthly_gain = (home_value - prev_value) * owns
            result[scen_idx] += monthly_gain

    return result


def calc_rent_income(timeline, M, Y):
    """
    Calculate rental income - only when property is owned but NOT occupied.

    Args:
        timeline: List of dicts, one per scenario
        M: Month indices, shape (n_scenarios, n_months)
        Y: Year indices, shape (n_scenarios, n_months)

    Returns:
        Array of shape (n_scenarios, n_months) with rental income (positive values)
    """
    n_scenarios = len(timeline) if isinstance(timeline, list) else 1
    timeline_list = timeline if isinstance(timeline, list) else [timeline]

    n_months = timeline_list[0]['n_months']
    result = np.zeros((n_scenarios, n_months))

    for scen_idx, tl in enumerate(timeline_list):
        n_properties = tl.get('n_properties', 0)

        for prop_idx in range(n_properties):
            prop_name = f'property_{prop_idx}'
            owns = tl[f'{prop_name}_owns']  # Boolean array: (n_months,)
            occupied = tl[f'{prop_name}_occupied']  # Boolean array: (n_months,)
            rent_income = tl[f'{prop_name}_rent_monthly_income']
            growth_rate = tl[f'{prop_name}_annual_rent_income_growth_rate']
            vacancy_rate = tl[f'{prop_name}_rent_income_vacancy_rate']

            # Calculate rental income with growth
            income = rent_income * (1 + growth_rate)**((M[scen_idx] - 1) / 12) * (1 - vacancy_rate)

            # Only collect when owned AND not occupied
            can_rent_out = owns & ~occupied
            result[scen_idx] += income * can_rent_out

    return result


def calc_alternative_rent(timeline, M, Y):
    """
    DEPRECATED: This calculation is no longer used in the layer-based system.
    Living arrangements are now handled explicitly through rental layers.
    Returns zeros for compatibility.

    Args:
        timeline: List of dicts, one per scenario
        M: Month indices, shape (n_scenarios, n_months)
        Y: Year indices, shape (n_scenarios, n_months)

    Returns:
        Array of shape (n_scenarios, n_months) filled with zeros
    """
    n_scenarios = len(timeline) if isinstance(timeline, list) else 1
    n_months = timeline[0]['n_months'] if isinstance(timeline, list) else timeline['n_months']
    return np.zeros((n_scenarios, n_months))


def calc_mortgage(timeline, M, Y):
    """
    Calculate mortgage payments using amortization formula - paid when owned.

    Args:
        timeline: List of dicts, one per scenario
        M: Month indices, shape (n_scenarios, n_months)
        Y: Year indices, shape (n_scenarios, n_months)

    Returns:
        Array of shape (n_scenarios, n_months) with mortgage payments (negative values)
    """
    n_scenarios = len(timeline) if isinstance(timeline, list) else 1
    timeline_list = timeline if isinstance(timeline, list) else [timeline]

    n_months = timeline_list[0]['n_months']
    result = np.zeros((n_scenarios, n_months))

    for scen_idx, tl in enumerate(timeline_list):
        n_properties = tl.get('n_properties', 0)

        for prop_idx in range(n_properties):
            prop_name = f'property_{prop_idx}'
            owns = tl[f'{prop_name}_owns']  # Boolean array: (n_months,)
            principal = tl[f'{prop_name}_principal'][0]  # Get scalar value
            down_payment_pct = tl[f'{prop_name}_down_payment_pct']
            loan_years = tl[f'{prop_name}_loan_years']
            mortgage_rate = tl[f'{prop_name}_annual_mortgage_rate']

            loan_principal = principal * (1 - down_payment_pct)
            loan_months = int(loan_years * 12)
            monthly_rate = mortgage_rate / 12

            # Calculate monthly payment
            if loan_months > 0 and loan_principal > 0 and monthly_rate > 0:
                numerator = loan_principal * monthly_rate * (1 + monthly_rate)**loan_months
                denominator = (1 + monthly_rate)**loan_months - 1
                monthly_payment = numerator / denominator
            else:
                monthly_payment = 0

            # Create mask for months within loan period and when owned
            m_array = M[scen_idx]
            # Find first month of ownership
            first_own_month = np.where(owns)[0][0] + 1 if owns.any() else n_months + 1
            # Mortgage payments occur for loan_months starting from first ownership month
            mortgage_mask = (m_array >= first_own_month) & (m_array < first_own_month + loan_months)
            mortgage_mask = mortgage_mask.astype(float)

            result[scen_idx] += -monthly_payment * mortgage_mask

    return result


def calc_equity_buildup(timeline, M, Y):
    """
    Calculate equity buildup (principal portion of mortgage payment each month).

    Args:
        timeline: List of dicts, one per scenario
        M: Month indices, shape (n_scenarios, n_months)
        Y: Year indices, shape (n_scenarios, n_months)

    Returns:
        Array of shape (n_scenarios, n_months) with equity buildup (positive values)
    """
    n_scenarios = len(timeline) if isinstance(timeline, list) else 1
    timeline_list = timeline if isinstance(timeline, list) else [timeline]

    n_months = timeline_list[0]['n_months']
    result = np.zeros((n_scenarios, n_months))

    for scen_idx, tl in enumerate(timeline_list):
        n_properties = tl.get('n_properties', 0)

        for prop_idx in range(n_properties):
            prop_name = f'property_{prop_idx}'
            owns = tl[f'{prop_name}_owns']  # Boolean array: (n_months,)
            principal = tl[f'{prop_name}_principal'][0]  # Get scalar value
            down_payment_pct = tl[f'{prop_name}_down_payment_pct']
            loan_years = tl[f'{prop_name}_loan_years']
            mortgage_rate = tl[f'{prop_name}_annual_mortgage_rate']

            loan_principal = principal * (1 - down_payment_pct)
            loan_months = int(loan_years * 12)
            monthly_rate = mortgage_rate / 12

            # Calculate monthly payment
            if loan_months > 0 and loan_principal > 0 and monthly_rate > 0:
                numerator = loan_principal * monthly_rate * (1 + monthly_rate)**loan_months
                denominator = (1 + monthly_rate)**loan_months - 1
                monthly_payment = numerator / denominator
            else:
                monthly_payment = 0
                result[scen_idx] += 0
                continue

            # Find first month of ownership
            first_own_month = np.where(owns)[0][0] + 1 if owns.any() else n_months + 1

            # Calculate remaining balance and equity buildup for each month
            m_array = M[scen_idx]

            for month_idx in range(n_months):
                month_num = m_array[month_idx]

                # Check if within loan period
                if month_num >= first_own_month and month_num < first_own_month + loan_months:
                    # Months since loan start
                    months_into_loan = month_num - first_own_month + 1

                    # Remaining balance at this month
                    numerator_balance = (1 + monthly_rate)**loan_months - (1 + monthly_rate)**months_into_loan
                    denominator_balance = (1 + monthly_rate)**loan_months - 1
                    remaining_balance = loan_principal * numerator_balance / denominator_balance

                    # Interest payment this month
                    interest_payment = remaining_balance * monthly_rate

                    # Principal payment (equity buildup)
                    principal_payment = monthly_payment - interest_payment
                    result[scen_idx, month_idx] += principal_payment

    return result


def calculate_opportunity_cost(net_balance_all_scenarios, compound_annual_growth_rate, n_months):
    """
    Calculate opportunity cost (investment gains from savings) for each scenario.

    At each month, we compare each scenario to the most expensive option. The cheaper options
    save money vs. the expensive option, and we assume they invest those savings. The most
    expensive option has 0 opportunity cost (no savings to invest).

    Args:
        net_balance_all_scenarios: Array of shape (n_scenarios, n_months)
        compound_annual_growth_rate: Annual growth rate (e.g., 0.085 for 8.5%)
        n_months: Total number of months in simulation

    Returns:
        Array of shape (n_scenarios, n_months) with cumulative investment gains.
        The most expensive option gets 0.
        Cheaper options get positive values (cumulative invested savings).
    """
    n_scenarios = net_balance_all_scenarios.shape[0]
    monthly_rate = (1 + compound_annual_growth_rate) ** (1/12) - 1

    # Initialize opportunity cost array
    opportunity_cost = np.zeros_like(net_balance_all_scenarios)

    # For each scenario, track the cumulative invested savings
    for scenario_idx in range(n_scenarios):
        cumulative_invested = 0.0

        for month_idx in range(n_months):
            # Get net balance for all scenarios at this month
            balances = net_balance_all_scenarios[:, month_idx]

            # Find the worst balance (most expensive/most negative option)
            worst_balance = np.min(balances)

            # Calculate savings compared to the worst option
            # If this scenario is cheaper, difference is positive (savings to invest)
            # If this scenario is the worst, difference is 0 (no savings)
            savings = balances[scenario_idx] - worst_balance

            # Add this month's savings to the invested amount
            # Then compound the entire invested amount by one month's growth
            cumulative_invested = (cumulative_invested + savings) * (1 + monthly_rate)

            # Store the cumulative invested savings at this point in time
            opportunity_cost[scenario_idx, month_idx] = cumulative_invested

    return opportunity_cost
