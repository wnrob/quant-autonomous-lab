"""Fixed ETF universe (liquid, multi-asset). Survivorship caveat: these are funds that exist TODAY; delisted/merged ETFs are absent."""
ETF24 = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "DBC", "USO",
         "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLU", "XLI", "XLB", "AGG"]
UNIVERSES = {"etf24": ETF24}
