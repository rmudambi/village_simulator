import sys

min_version, max_version = ((3, 8), "3.8"), ((3, 11), "3.11")

if not (min_version[0] <= sys.version_info[:2] <= max_version[0]):
    # Python 3.5 does not support f-strings
    py_version = ".".join([str(v) for v in sys.version_info[:3]])
    error = (
        "\n----------------------------------------\n"
        "Error: Village Simulator runs under python {min_version}-{max_version}.\n"
        "You are running python {py_version}".format(
            min_version=min_version[1], max_version=max_version[1], py_version=py_version
        )
    )
    print(error, file=sys.stderr)
    sys.exit(1)

from pathlib import Path

from setuptools import find_packages, setup

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src"

    about = {}
    with (src_dir / "village_simulator" / "__about__.py").open() as f:
        exec(f.read(), about)

    with (base_dir / "README.rst").open() as f:
        long_description = f.read()

    install_requirements = [
        "pandas",
        "scipy",
        "vivarium",
    ]

    setup_requires = ["setuptools_scm"]

    interactive_requirements = [
        "IPython",
        "ipywidgets",
        "jupyter",
    ]

    test_requirements = [
        "pytest",
        "pytest-mock",
    ]

    setup(
        name=about["__title__"],
        description=about["__summary__"],
        long_description=long_description,
        # license=about["__license__"],
        url=about["__uri__"],
        author=about["__author__"],
        # author_email=about["__email__"],
        classifiers=[
            "Natural Language :: English",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX",
            "Operating System :: POSIX :: BSD",
            "Operating System :: POSIX :: Linux",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: Python",
            "Programming Language :: Python :: Implementation :: CPython",
        ],
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,
        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            "test": test_requirements,
            "interactive": interactive_requirements,
            "dev": test_requirements + interactive_requirements,
        },
        # entry_points="""
        #         [console_scripts]
        #         simulate=village_simulator.interface.cli:simulate
        #     """,
        zip_safe=False,
        use_scm_version={
            "write_to": "src/village_simulator/_version.py",
            "write_to_template": '__version__ = "{version}"\n',
            "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
        },
        setup_requires=setup_requires,
    )
