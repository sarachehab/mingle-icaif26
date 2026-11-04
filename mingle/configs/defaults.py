from pathlib import Path
import pandas as pd

from mingle.utils.enums import Index, BloombergField, CutType
from mingle.utils.rebalancing.frequency import RebalancingFrequency

# ======== 

INDICES = [Index.SPX, Index.STOXX, Index.NIKKEI]
DATA_DIR = Path("data")
PRICE_FIELD = BloombergField.CLOSE

START_DATE = pd.Timestamp("2017-01-04")
END_DATE = pd.Timestamp("2026-03-30")


# ========= REPRESENTATION LEARNING CONFIGURATION =========

SEED = 42

TOP_N_BY_VOLUME = 100

REBALANCING_FREQUENCY = RebalancingFrequency.MONTHLY
LOOKBACK_HORIZON = 24

NB_FACTORS = 6

ALPHA = 2.2
BETA = 3.0
LAMBDA_L = 0.1
DELTA = 1.0e3
EPS = 1.0e-4

DECAY = 0.997

RHO_B = 5.0
RHO_W = 1.0
RHO_D = 1.0
ADAPT_EVERY = 40
MU = 15.0
TAU = 2.0
RHO_MIN = 1.0e-3
RHO_MAX = 1.0e2

MAX_EPOCHS = 400

ABS_EPS = 1.0e-3
REL_EPS = 1e-4
ABS_EPS_B = 0.05
REL_EPS_B = 0.01

CUT_TYPE = CutType.CutV
CUT_V_COUNT = 24
