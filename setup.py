from pathlib import Path
from shutil import copy2

import setuptools
from setuptools.command.build_py import build_py


RESOURCE_FILES = {
    "params.yaml": Path("dynaphos/data/defaults/params.yaml"),
    "safety.yaml": Path("dynaphos/data/defaults/safety.yaml"),
    "coords_400um.yaml": Path("dynaphos/data/electrodes/coords_400um.yaml"),
    "coords_800um.yaml": Path("dynaphos/data/electrodes/coords_800um.yaml"),
    "coords_1200um.yaml": Path("dynaphos/data/electrodes/coords_1200um.yaml"),
}


class BuildPyWithResources(build_py):
    """Copy canonical source configs into the installed package."""

    def run(self):
        super().run()
        project_root = Path(__file__).resolve().parent
        for source_name, package_path in RESOURCE_FILES.items():
            source = project_root / "config" / source_name
            target = Path(self.build_lib) / package_path
            target.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, target)
        fixture_target = Path(self.build_lib) / "dynaphos/data/fixtures/cat.jpg"
        fixture_target.parent.mkdir(parents=True, exist_ok=True)
        copy2(project_root / "examples" / "cat.jpg", fixture_target)


if __name__ == "__main__":
    setuptools.setup(cmdclass={"build_py": BuildPyWithResources})
