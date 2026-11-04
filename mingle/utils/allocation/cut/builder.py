from mingle.utils.enums import CutType
from mingle.utils.allocation.cut.cutn import CutNAlgorithm
from mingle.utils.allocation.cut.cutv import CutVAlgorithm

def get_cut_alg(cut_type: CutType, splits_count: int):

    match cut_type:
        case CutType.CutN: return CutNAlgorithm(splits_count=splits_count)
        case CutType.CutV: return CutVAlgorithm(splits_count=splits_count)

    raise NotImplementedError(f"Cut type not recogined in builder: {cut_type}")
