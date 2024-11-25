# Setup Instructions

1. You will need to download and install the Anaconda python distribution. Anaconda is distribution of the Python programming language for scientific computing and simplifies package management and deployment. You can download it here... https://www.anaconda.com/download. The current setup has only been tested with Mac so far. 

2. Clone the piView repository to your local machine: 

    '''
    git clone https://github.com/pixxelhq/piView.git
    ''' 

3. Navigate to the piView folder on your local machine, for example:
    
    ```
    cd /Users/<Username>/Desktop/piView/
    ```

2. We will then create a separate computing environment using Anaconda to create a container specifically for running the pixxel viewer. By running the following command in your terminal, it will download and install all the packages and dependencies that are required to run all the functionality of the pixxel viewer. Type the following in your terminal to create a compatible Python environment:

    ```
    conda env create -f setup/mac_requirements.yml
    ```

3. This will have created an environment called "piview". Your default environment in Anaconda is called "Base", and you will see it in parenthesis before your username in the terminal. You can see a list of all your environments by typing the following command. Try it and make sure you see pixxel_viewer in the list:

    ```
    conda envs list 
    ```
    
4. You can now activate the pixxel viewer environment  by typing:

    ```
    conda activate piview 
    ```

5. Now you can launch Jupyter Lab to open the notebooks included.

    ```
    jupyter lab 
    ```

## Contact Info  

Email: <jeremy@pixxel.space>  
Organization: Pixxel Space Technologies
Website: pixxel.space  
Date last modified: 18-01-2024  
 
