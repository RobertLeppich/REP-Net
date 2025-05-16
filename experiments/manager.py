from setups import *
import os
import torch
import random

GPU = "MIG-11c29e81-e611-50b5-b5ef-609c0a0fe58b"  # GPU0-1 40gb


os.environ["CUDA_VISIBLE_DEVICES"] = GPU

if __name__ == '__main__':
    dry_run = False

    # if dry_run:
    #     torch.autograd.detect_anomaly(True)

    traffic.run_downstream(dry_run=dry_run)
    #exp2.run_pretrain(dry_run=dry_run)
    #exp2.run_finetune(dry_run=dry_run)

    pass