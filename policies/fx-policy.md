# Currency conversion policy

Transaction amounts are stored in the merchant's settlement currency.
For cross-merchant reporting, convert amounts to **USD** using the daily reference table (fx_rates, units of currency per 1 USD):
amount_usd = amount / units_per_usd.
Round reported USD totals to 2 decimal places. Do not convert declined transactions into volume figures.
