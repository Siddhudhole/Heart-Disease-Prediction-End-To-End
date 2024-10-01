from setuptools import find_packages,setup 
from typing import List 

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    ''' This function returns a list of requirements'''
    with open(file_path, 'r') as f: 
        lines = f.readlines() 
        requirements = [req.replace('\n',"") for req in lines ] 
    requirements.pop()  
    return requirements  



setup(
    name='my_package',
    version='1.0',
    description='My Python package',
    author='Siddharth Dhole ',
    author_email='shidhudhole358@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)



