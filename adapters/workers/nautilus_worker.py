"""nautilus_trader worker (runs ONLY inside envs/nautilus, launched by adapters/nautilus_adapter.py via run_in_env).

Scope: offline BacktestEngine with ONE simulated venue ("SIM"). No live node, no execution or data clients, no credentials,
no network. nautilus_trader (LGPL-3.0) is used unmodified as an installed PyPI wheel through its public API only.

Input (argv[1] JSON): {"data": <parquet path with columns [close] on a daily DatetimeIndex>, "out": <json path>,
  "lookback_months": int, "cost_bps": float, "starting_cash": float, "mode": "same_bar_close"|"next_bar_close",
  "month_ends": [ISO dates], "price_precision": int}
Output: writes a JSON result file and prints {"ok": true, "out": path} as the last stdout line.

Execution semantics (verified from fill prices in the output, not assumed):
  same_bar_close  - signal computed in on_bar(t) from closes <= t; a MARKET order is submitted inside on_bar(t). With
                    bar_execution the simulated L1 book is at bar t's close when on_bar runs, so the order fills at close t
                    (a market-on-close assumption, identical to the vector engine's "weight decided at close t earns t+1").
  next_bar_close  - same decision at close t, but the order is only submitted in on_bar(t+1) and fills at close t+1
                    (one extra bar of delay; a pre-declared sensitivity, not a tuning knob).
Sizing: long = floor(equity / (close * (1 + fee))) whole shares (lot size 1), flat = sell the whole position.
Fees: MakerTakerFeeModel, taker_fee = cost_bps / 1e4 of fill notional. Fill model: no slippage (prob_slippage=0).
"""
import json, sys
from decimal import Decimal
DEV_END = "2019-12-31"


def main(a: dict) -> dict:
    import pandas as pd
    from nautilus_trader import __version__ as NV
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
    from nautilus_trader.config import LoggingConfig, StrategyConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
    from nautilus_trader.model.instruments import Equity
    from nautilus_trader.model.objects import Money, Price, Quantity
    from nautilus_trader.trading.strategy import Strategy

    mode = a["mode"]
    if mode not in ("same_bar_close", "next_bar_close"): raise ValueError(f"unknown mode {mode}")
    px = pd.read_parquet(a["data"])["close"].astype(float)
    if len(px) and pd.DatetimeIndex(px.index).max() > pd.Timestamp(DEV_END):      # second, independent guard (no override)
        raise PermissionError(f"bars after {DEV_END} (protected test window) refused by worker")
    fee = Decimal(str(a["cost_bps"])) / Decimal(10_000)
    prec = int(a.get("price_precision", 4)); L = int(a["lookback_months"])
    month_ends = set(a["month_ends"])

    venue = Venue("SIM"); iid = InstrumentId.from_str("SPY.SIM")
    inst = Equity(instrument_id=iid, raw_symbol=Symbol("SPY"), currency=USD, price_precision=prec,
                  price_increment=Price(10.0 ** -prec, prec), lot_size=Quantity.from_int(1), ts_event=0, ts_init=0,
                  maker_fee=fee, taker_fee=fee)
    bar_type = BarType.from_str("SPY.SIM-1-DAY-LAST-EXTERNAL")
    bars = []
    for d, c in px.items():
        ts = int((pd.Timestamp(d).normalize() + pd.Timedelta(hours=21)).value)     # session close ~16:00 ET as 21:00 UTC
        p = Price(round(float(c), prec), prec)
        bars.append(Bar(bar_type, p, p, p, p, Quantity.from_int(1_000_000_000), ts, ts))   # close-only data: O=H=L=C

    class TrendCfg(StrategyConfig, frozen=True):
        lookback_months: int = 12

    class TrendLongFlat(Strategy):
        def __init__(self, config):
            super().__init__(config); self.me_closes = []; self.pending = None; self.equity = []; self.decisions = []

        def on_start(self): self.subscribe_bars(bar_type)

        def _qty(self):
            pos = self.cache.positions_open(instrument_id=iid)
            return sum(float(p.signed_qty) for p in pos)

        def on_bar(self, bar):
            d = pd.Timestamp(bar.ts_event, unit="ns").strftime("%Y-%m-%d"); close = float(bar.close)
            acct = self.portfolio.account(venue); cash = float(acct.balance_total(USD).as_double()); q = self._qty()
            self.equity.append({"date": d, "cash": cash, "qty": q, "close": close, "equity": cash + q * close})
            if self.pending is not None:                      # next_bar_close: act now on yesterday's decision
                self._target(self.pending, close, cash, q); self.pending = None
            if d in month_ends:
                self.me_closes.append(close)
                if len(self.me_closes) > self.config.lookback_months:
                    mom = self.me_closes[-1] / self.me_closes[-1 - self.config.lookback_months] - 1.0
                    sig = 1 if mom > 0 else 0
                    self.decisions.append({"date": d, "mom": mom, "signal": sig})
                    if mode == "same_bar_close": self._target(sig, close, cash, q)
                    else: self.pending = sig

        def _target(self, sig, close, cash, q):
            if sig == 1 and q == 0:
                n = int((cash + q * close) / (close * (1.0 + float(fee))))
                if n > 0:
                    self.submit_order(self.order_factory.market(iid, OrderSide.BUY, Quantity.from_int(n), time_in_force=TimeInForce.DAY))
            elif sig == 0 and q > 0:
                self.submit_order(self.order_factory.market(iid, OrderSide.SELL, Quantity.from_int(int(q)), time_in_force=TimeInForce.DAY))

    eng = BacktestEngine(BacktestEngineConfig(trader_id=TraderId("QAL-BACKTEST-001"), logging=LoggingConfig(log_level="ERROR")))
    eng.add_venue(venue=venue, oms_type=OmsType.NETTING, account_type=AccountType.CASH, base_currency=USD,
                  starting_balances=[Money(float(a["starting_cash"]), USD)],
                  fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
                  fee_model=MakerTakerFeeModel(), bar_execution=True)
    eng.add_instrument(inst); eng.add_data(bars)
    strat = TrendLongFlat(TrendCfg(lookback_months=L)); eng.add_strategy(strat)
    eng.run()

    fills = []
    fr = eng.trader.generate_order_fills_report()
    for _, r in fr.iterrows():
        fills.append({"ts": str(r.get("ts_last")), "side": str(r["side"]), "qty": float(r["filled_qty"]),
                      "avg_px": float(r["avg_px"]), "commission": str(r.get("commissions"))})
    # authoritative per-fill events from the cache (price, qty, commission)
    events = []
    for o in eng.cache.orders():
        for e in o.events:
            if type(e).__name__ == "OrderFilled":
                events.append({"date": pd.Timestamp(e.ts_event, unit="ns").strftime("%Y-%m-%d"), "side": "BUY" if e.order_side == OrderSide.BUY else "SELL",
                               "qty": float(e.last_qty), "price": float(e.last_px), "commission": float(e.commission.as_double())})
    acct = eng.portfolio.account(venue)
    final = {"cash": float(acct.balance_total(USD).as_double()), "qty": strat._qty(), "last_close": float(px.iloc[-1])}
    final["equity"] = final["cash"] + final["qty"] * final["last_close"]
    res = {"nautilus_version": NV, "mode": mode, "equity": strat.equity, "decisions": strat.decisions, "fills": events,
           "fills_report_rows": len(fills), "final": final,
           "config": {"venue": "SIM (simulated)", "oms": "NETTING", "account": "CASH", "fee_model": "MakerTakerFeeModel",
                      "taker_fee": str(fee), "fill_model": "FillModel(prob_slippage=0)", "bar_execution": True,
                      "price_precision": prec, "bars": "close-only O=H=L=C, ts 21:00 UTC"}}
    eng.dispose()
    with open(a["out"], "w") as f: json.dump(res, f)
    return {"ok": True, "out": a["out"]}


if __name__ == "__main__":
    print(json.dumps(main(json.loads(sys.argv[1]))))
