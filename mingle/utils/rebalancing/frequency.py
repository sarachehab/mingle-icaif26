from enum import Enum

class RebalancingFrequency(Enum):
    MONTHLY = "1BMS"
    QUARTERLY = "3BMS"
    TRIMESTER = "4BMS"
    SEMI_ANNUAL = "6BMS"
    ANNUAL = "12BMS"
