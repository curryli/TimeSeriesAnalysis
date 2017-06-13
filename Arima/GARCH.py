# -*- coding: utf-8 -*-
#https://pypi.python.org/pypi/arch
#conda install -c https://conda.binstar.org/bashtage arch


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from arch import arch_model

r = np.array([0.945532630498276,
              0.614772790142383,
              0.834417758890680,
              0.862344782601800,
              0.555858715401929,
              0.641058419842652,
              0.720118656981704,
              0.643948007732270,
              0.138790608092353,
              0.279264178231250,
              0.993836948076485,
              0.531967023876420,
              0.964455754192395,
              0.873171802181126,
              0.937828816793698])
 

garch11 = arch_model(r, p=1, q=1)
res = garch11.fit(update_freq=10)
print(res.summary())