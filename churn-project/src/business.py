"""Cost-benefit logic: turn churn probabilities into a money number."""
import numpy as np

# Default assumptions (editable in the app). All amounts are in the same currency unit.
OFFER_COST = 20.0        # cost of one retention offer (discount, call, gift)
CUSTOMER_VALUE = 500.0   # revenue kept when a would-be churner is saved
SUCCESS_RATE = 0.30      # share of contacted churners the offer actually saves


def net_benefit(y_true, proba, threshold,
                offer_cost=OFFER_COST, value=CUSTOMER_VALUE, success=SUCCESS_RATE) -> float:
    """Profit from contacting everyone whose churn probability >= threshold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    contacted = proba >= threshold
    true_pos = int((contacted & (y_true == 1)).sum())
    return float(true_pos * success * value - int(contacted.sum()) * offer_cost)


def benefit_curve(y_true, proba, thresholds, **kwargs):
    return np.array([net_benefit(y_true, proba, t, **kwargs) for t in thresholds])


def contact_everyone(y_true, offer_cost=OFFER_COST, value=CUSTOMER_VALUE, success=SUCCESS_RATE) -> float:
    """Baseline: send the offer to every customer, no model."""
    y_true = np.asarray(y_true)
    return float(int((y_true == 1).sum()) * success * value - len(y_true) * offer_cost)
