# ID TO LABEL
IDTOLABEL = {
     'fewrel': {
         71: 'P931',
         8: 'P4552',
         20: 'P140',
         12: 'P1923',
         19: 'P150',
         54: 'P6',
         14: 'P27',
         4: 'P449',
         59: 'P1435',
         61: 'P175',
         73: 'P1344',
         17: 'P39',
         75: 'P527',
         50: 'P740',
         67: 'P706',
         57: 'P84',
         25: 'P495',
         72: 'P123',
         79: 'P57',
         18: 'P22',
         70: 'P178',
         66: 'P241',
         3: 'P403',
         60: 'P1411',
         39: 'P135',
         35: 'P991',
         43: 'P156',
         46: 'P176',
         64: 'P31',
         52: 'P1877',
         55: 'P102',
         6: 'P1408',
         69: 'P159',
         36: 'P3373',
         74: 'P1303',
         51: 'P17',
         11: 'P106',
         34: 'P551',
         1: 'P937',
         26: 'P355',
         21: 'P710',
         56: 'P137',
         33: 'P674',
         68: 'P466',
         27: 'P136',
         10: 'P306',
         76: 'P127',
         62: 'P400',
         38: 'P974',
         40: 'P1346',
         0: 'P460',
         41: 'P86',
         58: 'P118',
         16: 'P264',
         24: 'P750',
         42: 'P58',
         63: 'P3450',
         44: 'P105',
         28: 'P276',
         13: 'P101',
         5: 'P407',
         22: 'P1001',
         65: 'P800',
         53: 'P131',
         32: 'P177',
         49: 'P364',
         77: 'P2094',
         7: 'P361',
         30: 'P641',
         37: 'P59',
         9: 'P413',
         45: 'P206',
         47: 'P412',
         2: 'P155',
         78: 'P26',
         15: 'P410',
         48: 'P25',
         29: 'P463',
         31: 'P40',
         23: 'P921'
    },
}


ENT_START_ = ' [ENTITYSTART] '
ENT_END_ = ' [ENTITYEND] '
ROOT = " [ROOT] "
SUBJ_START = ' [SUBJSTART] '
SUBJ_END = ' [SUBJEND] '
OBJ_START = ' [OBJSTART] '
OBJ_END = ' [OBJEND] '
ENT = ' [ENTITY] '
ENT_START = ENT
ENT_END = ENT
AUG = " [AUG] "
ENT_MASK = " [ENTITYMASK] "
CLS = '[CLS]'
SEP = '[SEP]'
ENTITY = ' [ENTITY] '

MARK_TOKS = [
     SUBJ_START,
     SUBJ_END,
     OBJ_START,
     OBJ_END,
     ENTITY,
     AUG,
     ENT_START,
     ENT_END,
     ROOT,
     ENT_MASK,
     ENT_START_,
     ENT_END_,
     "[mask]"
]