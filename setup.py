from setuptools import setup, find_packages

setup(
    name         = "daga",
    version      = "0.1.0",
    description  = "Dynamic Agentic Architecture Generation — energy-efficient multi-agent task solving",
    packages     = find_packages(),
    python_requires = ">=3.10",
    install_requires = [
        "httpx>=0.27.0",
    ],
    extras_require = {
        "eval": [
            "datasets>=2.20.0",
            "swebench>=2.1.0",
        ],
        "local": [
            "vllm>=0.4.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio",
        ],
    },
    entry_points = {
        "console_scripts": [
            "daga-eval=daga.evaluation.swebench_harness:main",
        ],
    },
)
