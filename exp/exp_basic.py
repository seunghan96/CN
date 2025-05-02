import os
import torch
from models import iTransformer, iTransformer_CN, iTransformer_ACN
from models import RMLP, RMLP_CN, RMLP_ACN
from models import TSMixer, TSMixer_CN, TSMixer_ACN
from models import S_Mamba, S_Mamba_CN, S_Mamba_ACN


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'iTransformer': iTransformer,
            'iTransformer_CN': iTransformer_CN,
            'iTransformer_ACN': iTransformer_ACN,
            'RMLP': RMLP,
            'RMLP_CN': RMLP_CN,
            'RMLP_ACN': RMLP_ACN,
            'TSMixer': TSMixer,
            'TSMixer_CN': TSMixer_CN,
            'TSMixer_ACN': TSMixer_ACN,
            'S_Mamba': S_Mamba,
            'S_Mamba_CN': S_Mamba_CN,
            'S_Mamba_ACN': S_Mamba_ACN,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
