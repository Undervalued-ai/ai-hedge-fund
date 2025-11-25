# AI Hedge Fund · Undervalued.ai

Undervalued.ai runs an AI hedge fund.

The fund runs several strategies autonomously.  
The same system produces stock analysis on the website so investors can find undervalued stocks.

## Purpose

The main priority is building and improving the fund.

If there is clear public demand and, if time permits, I intend to make some components open source and publish them here.  
Stars are one of the signals I will look at when deciding what is worth the effort.

## Current

For now, there is a single module that shows how fund metrics are calculated in a simple way:

- `examples/fund_metrics.py`  
  Portfolio metrics such as annualised return, max drawdown, Sharpe ratio, etc. on synthetic data.

More modules may be added later if they are safe and useful on their own.

## Performance

To see what the AI hedge fund is doing in practice, use the site:

- Fund overview: https://undervalued.ai/fund  

For a detailed view of each strategy:

- Original fund: https://undervalued.ai/fund/original  
- S&P 500 AI Select: https://undervalued.ai/fund/sp500  

To access stock analysis, search for a ticker from the home page:
https://undervalued.ai/
