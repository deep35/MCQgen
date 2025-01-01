from setuptools import find_packages,setup

setup(
    name='mcqGenerator',
    version='0.0.1',
    author='Deep Patel',
    author_email='deep.dp2782002@gmail.com',
    install_requires=['groq','langchain','streamlit','python-dotenv','PyPDF2','langchain_groq'],
    packages=find_packages()
)