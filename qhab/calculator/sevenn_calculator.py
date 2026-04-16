"""
Modified based on Jinmu Yu's code
"""

from types import NotImplementedType
import warnings
from sevenn.calculator import SevenNetCalculator

CALC_DCT = {
    'ompa': '/data2/shared_data/cps/o50mp7a_ft/checkpoint_2.pth',
    'omni': '/data2/shared_data/cps/omni/oob_v5/checkpoint_1.pth',
    'nano': '/data2_1/jinvk/checkpoints/nano',
    }

NANO_DCT = {
        4.5: 'nano_cut4.5',
        5.0: 'nano_cut5.0',
        5.5: 'nano_cut5.5',
        6.0: 'nano_cut6.0',
        }

FUNC_DCT = {
    'mpa': 'PBE',
    'omat24': 'PBE',
    'matpes_pbe': 'PBE',
    'spice': 'wB97M',
    'qcml': 'PBE0',
    'oc20': 'RPBE',
    'oc22': 'PBE',
    'mp_r2scan': 'r2SCAN',
    'matpes_r2scan': 'r2SCAN',
}

def return_calc(config):
    conf = config['calculator']
    model, modal = conf.get('model', 'zero'), conf.get('modal', None)

    is_zero = (str(model).lower() in ['zero', '0'])
    is_nano = (str.(model).lower() == 'nano')
    model_path = CALC_DCT.get(model, None) 
    if is_nano:
        model_path = os.path.join(model_path, NANO_DCT[modal], 'checkpoint_2.pth')

    if is_nano:
        calc_kwargs = {
                'model': model_path,
                'enable_flash': True
                } 

    elif model_path:
        calc_kwargs = {
            'model': model_path,
            'modal': modal,
            'enable_flash': True, #TODO;
        }

    else:
        if is_zero:
            calc_kwargs = {
                    'enable_flash': True # default is 7net-0
                    }
        else:
            calc_kwargs = {
                'model': f'7net-{model}', #TODO, 7net-omni, 7net-mf-ompa, 7net-l3i5, 7net-omat
                'modal': modal,
                'enable_flash': True, #TODO;
            }

    # functional = FUNC_DCT.get(modal, None)
    print(f"[SevenNet] model={model}, modal={modal}")
    print(f"[SevenNet] potential path: {model_path}")

    calc = SevenNetCalculator(**calc_kwargs)
    if conf.get('d3'):
        from ase.calculators.mixing import SumCalculator
        from sevenn.calculator import D3Calculator
        d3 = D3Calculator()
        calc_d3 = D3Calculator(functional_name = 'pbe')
        return SumCalculator([calc, calc_d3])
    return calc
