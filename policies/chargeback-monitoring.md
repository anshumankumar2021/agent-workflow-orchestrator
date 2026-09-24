# Chargeback monitoring thresholds

Northwind Payments measures each merchant's **chargeback ratio** for a period:

chargeback ratio = (number of chargebacks raised against the merchant's transactions from the period) ÷ (number of the merchant's transactions from the period with status 'approved') × 100

- Below 0.9%: normal. No action.
- 0.9% to 1.8%: **monitoring**. The merchant is placed on the watch list and reviewed monthly.
- Above 1.8%: **escalation**. Open a P1 risk ticket for the merchant within 2 business days.

Refunded and declined transactions are excluded from the denominator. The ratio is reported as a percentage with two decimals.
