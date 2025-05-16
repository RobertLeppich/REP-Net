from experiments.setups import *
import os

GPU = "MIG-11c29e81-e611-50b5-b5ef-609c0a0fe58b"  # GPU0-1 40gb
#GPU = "MIG-a1208c4e-caad-5519-9d69-6b0998c74b9f" # GPU1-1 40gb
#GPU = "MIG-e5cc7cc7-0a84-5237-9062-30b65511cb40"  # GPU0-2 20gb
#GPU = "MIG-2e5917ed-61c1-56e9-b63e-77a713e2379e"  # GPU0-3 10gb
#GPU = "MIG-f29aed64-88d8-567c-9102-1b0a7b4e0b3a"  # 20gb GPU1-2
#GPU = "MIG-5ffb975c-ed57-5d17-88ea-eee6c8c0325e"  # 10gb GPU1-3
#GPU  # os.getenv("CUDA_VISIBLE_DEVICES", GPU)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU

if __name__ == '__main__':
    dry_run = True

    ecl_96.run(dry_run=dry_run)
    ettm1_96.run(dry_run=dry_run)
    etth1_96.run(dry_run=dry_run)
