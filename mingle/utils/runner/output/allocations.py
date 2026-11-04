import numpy as np
import pandas as pd
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict


class AllocationView(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: Union[np.ndarray, pd.DataFrame]
    returns: Union[np.ndarray, pd.Series]
    in_returns: Union[np.ndarray, pd.Series]
    history: Optional[Any] = None
