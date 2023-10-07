import subprocess

subprocess.run("python SinglePlaneKepler.py", shell=True)
subprocess.run("python SinglePlaneJ2.py", shell=True)
# subprocess.run("python MultiPlane2Kepler.py", shell=True)
subprocess.run("python MultiPlane2J2.py", shell=True)
# subprocess.run("python MultiPlane6Kepler.py", shell=True)
subprocess.run("python MultiPlane6J2.py", shell=True)
