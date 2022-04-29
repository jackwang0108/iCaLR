import os
from pathlib import Path

from helper import ProjectPath

def runs(num_class: int, message):
    global ProjectPath
    mian = ProjectPath.base.joinpath("mian.py")
    main = r"C:\Users\Alienware\.conda\envs\torch\python.exe c:\Users\Alienware\projects\iCaRL\main.py "
    main += "-l "
    main += "-s "
    main += f"-n {num_class} "
    main += f"-m \"{message}\" "
    os.system(main)


if __name__ == "__main__":
    # runs(num_class=10, message=f"Task Length 10, Debug RUN")
    for num_class in [2, 5, 10, 20, 50]:
        for run_idx in range(1, 11, 1):
            runs(num_class=num_class, message=f"Task Length {num_class} Run {run_idx}")
