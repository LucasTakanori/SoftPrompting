import subprocess
import sys

def create_requirements_txt():
    # Run pip list and capture the output
    result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], capture_output=True, text=True)
    
    # Check if the command was successful
    if result.returncode != 0:
        print("Error running pip list:")
        print(result.stderr)
        return
    
    # Get the list of installed packages
    packages = result.stdout.split('\n')
    
    # Write the packages to requirements.txt
    with open('requirements.txt', 'w') as f:
        for package in packages:
            if package and not package.startswith('pip==') and not package.startswith('setuptools=='):
                f.write(f"{package}\n")
    
    print("requirements.txt file has been created successfully.")

if __name__ == "__main__":
    create_requirements_txt()