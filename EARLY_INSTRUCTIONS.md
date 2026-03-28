python triple_reservoir_coreml.py --export triple_reservoir_ane.mlpackage --n-nodes 192

---

import coremltools as ct
import numpy as np

m = ct.models.MLModel(
    "triple_reservoir_ane.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
state = m.make_state()
x = np.asarray([[0.02, -0.10, 0.30]], dtype=np.float16)
y = m.predict({"x": x}, state=state)["y"]
print(y)

