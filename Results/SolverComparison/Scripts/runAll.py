import subprocess

subprocess.run("python basic_MPC.py", shell=True)
subprocess.run("python basic_SLS.py", shell=True)
subprocess.run("python transformed_SLS.py", shell=True)
subprocess.run("python sparse_SLS.py", shell=True)
subprocess.run("python robust_SLS.py", shell=True)
