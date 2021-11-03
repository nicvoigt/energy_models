import typing
import os


class Paths(typing.NamedTuple):
    main_project_dir: str = os.path.join(r"C:\Users\nicoj\python_projects\medium_articles", "Energy_Models")
    data_dir: str = os.path.join(main_project_dir, "data")
    output_dir: str = os.path.join(main_project_dir, "output")
    output_mpc: str = os.path.join(output_dir, "mpc")
    output_rl: str = os.path.join(output_dir, "rl")


paths = Paths()

