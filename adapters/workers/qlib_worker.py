import json, sys
import pandas as pd, qlib
from qlib.config import REG_US
from qlib.data import D
a = json.loads(sys.argv[1]); qlib.init(provider_uri=a["provider_uri"], region=REG_US, kernels=1, expression_cache=None, dataset_cache=None)
names = list(a["exprs"]); df = D.features(D.instruments("all"), [a["exprs"][n] for n in names], start_time=a["start"], end_time=a["end"], freq="day")
df.columns = names; df = df.reset_index(); df.to_parquet(a["out"]); print(json.dumps({"rows": len(df), "cols": names}))
