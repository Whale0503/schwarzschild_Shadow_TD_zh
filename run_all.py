import os
import subprocess
import sys
import time
def get_abs_path(rel_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 当前 run_all.py 所在目录
    return os.path.join(base_dir, rel_path)
start_time = time.time() 
try:
    print("[1/3] Running step1_all_geodesic.py ...")
    subprocess.run([sys.executable, get_abs_path("scripts/step1_all_geodesic.py")], check=True)

    print("[2/3] Running step2_theta0_psi0.py ...")
    subprocess.run([sys.executable, get_abs_path("scripts/step2_theta0_psi0_v2.py")], check=True)

    print("[3/3] Running step3_shadowplot.py ...")
    subprocess.run([sys.executable, get_abs_path("scripts/step3_shadowplot.py")], check=True)

    end_time = time.time()  # 记录总结束时间
    total_seconds = end_time - start_time

    print(f" All steps completed successfully in {total_seconds:.2f} seconds.")

except subprocess.CalledProcessError as e:
    print(f" A script failed with exit code {e.returncode}")
