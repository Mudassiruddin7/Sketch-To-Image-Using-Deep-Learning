![Sketch](https://miro.medium.com/v2/resize:fit:1400/1*-FnxXrH-ZiWlmIenM6-NvQ.gif)

# _**Sketch To Image Using Deep Learning**_
How many of us loved watching the show “Shaka Laka Bom Bom”. The magic pencil with him. Whatever thing the boy drew turned into a real thing. We will do something like that but in a virtual mode. We trained the machine to identify the sketches of airplane, ant, apple, axe, banana, bed, bench, bicycle, book and bus. You draw a sketch like one of these objects and our program will return the prediction along with the real world photo of the same object. 

# _**Base Paper**_
+ https://www.researchgate.net/publication/343275935_Diagram_Image_Retrieval_using_Sketch-Based_Deep_Learning_and_Transfer_Learning

# _**Algorithm Description**_

As we all are aware of the fact, how deep learning and transfer learning is revolutionizing the world with its immense capability of handling any kind of data and learning so efficiently. So, similarly we have applied the same concept by picking a deep learning model i.e., Convolutional neural network which basically work son the principle of having filters. Each convolutional layer has some specific filters to identify and extract the features from the input image and learn it and transfer it to other layers for further processing. We can have as many filters as possible in the convolutional layer depending on the data we are dealing on. Filter are nothing but feature detectors in the input data. Along with the convolutional layer we also have other layers which does further pre-processing such as Maxpooling, Activation function, Batch Normalization and dropout layer. These all contribute to the CNN model creation and along with the flatten and output layer. The reason we do flattening is to feed the output of the CNN model to the dense layer which gives us the probability of the predicted value.
We used transfer learning to build our model. We used VGG-16 to achieve this feat. This is a very useful model to use for classification and has performed exceedingly well in the Imagenet competition. It takes input with the dimensions of 224,224,3. 

![CNN](https://user-images.githubusercontent.com/88571564/180369609-a5272f38-7248-4939-911d-388b26c24fa1.png)

**Reference**

+ https://www.geeksforgeeks.org/vgg-16-cnn-model/

# _**How to Execute?**_
So, before execution we have some pre-requisites that we need to download or install i.e., anaconda environment, python and a code editor.
**Anaconda**: Anaconda is like a package of libraries and offers a great deal of information which allows a data engineer to create multiple environments and install required libraries easy and neat.

**Download link:**

![Anaconda](https://1.bp.blogspot.com/-UJ1Ws2zZ9V4/TtMbG2ynJiI/AAAAAAAABbM/m6t2kuEhKdY/s1600/The-biggest-anaconda-snake-3.jpg)

https://www.anaconda.com/

**Python**: Python is a most popular interpreter programming language, which is used in almost every field. Its syntax is very similar to English language and even children and learning it nowadays, due to its readability and easy syntax and large community of users to help you whenever you face any issues.

**Download link:**

![Python](https://i0.wp.com/reptileworldfacts.com/wp-content/uploads/2019/05/male-blonde-super-tiger-reticulated-python.jpg?resize=351%2C351&ssl=1)

https://www.python.org/downloads/

**Code editor**: Code editor is like a notepad for a programming language which allows user to write, run and execute program which we have written. Along with these some code editors also allows us to debug, which usually allows users to execute the code line by line and allows them to see where and how to solve the errors. But I personally feel visual code is very good to work with any programming language and makes a great deal of attachment with user.

**Download links:**

![Vs code](https://schwabencode.com/contents/logos/VS2019-Badge.png) ![Pycharm](https://i0.wp.com/scracked.com/wp-content/uploads/2020/01/PyCharm-2019.3.4-Crack.png?fit=200%2C200&ssl=1)

+ https://code.visualstudio.com/Download, 
+ https://www.jetbrains.com/pycharm/download/#section=windows

# _**How to create a new environment and configure jupyter notebook with it.**_
Let us define an environment and why we need different environments. An environment is a collection of libraries that are required to run our project. When we already have an environment with the necessary libraries, why do we need a new environment?
To avoid version mismatches, we create a new environment for each project. For example, in your previous project, you used "tf env" with tensorflow 2.4 and keras 2.4, but in your current project, you must use tensorflow 2.6 and keras 2.6. If you continue your project in the "tf env" environment, there will be a version mismatch and you will need to update tensorflow and keras, but this will cause problems with the previous project's execution. To avoid this, we create a new environment with tensorflow 2.6 and keras 2.6 and resume our project.

Let us now see how to create an environment in anaconda.
+ Type “conda create –n <<name_of_your_env>>”
example: conda create -n env
+ It will ask to proceed with the environment location, type ‘y’ and press enter.
+ When you press ‘y’, the environment will be created. To activate your environment type conda activate <<your_env_name>> . E.g., conda activate myenv.
+ You can see that the environment got changed after conda activate myenv line. It changed from “base” to “myenv” which means you are now working in “myenv” environment.
+ To install a library in your virtual environment type pip install <library_name>.
e.g., pip install pandas
+ Instead of installing libraries one by one you can even install by bunch, i.e., we have a txt file called requirements.tx which consists of all the libraries required to proceed with the project, so we can use it.
+ so, before installing requirements.txt, make sure you are in the specific path where your requirements.txt is located, basically this file is located in the folder where our executable files are located, so we need to move to that directory by following command.
**cd C:\folder_name**
+ Here A -> drive, folder name -> path where your executable file is saved
+ I go to that file path in anaconda using cd command 
1.	Go to drive where your project file is.
2.	Go to the path of your project using cd <path>
3.	Type pip install –r requirements.txt 
+ And all your required libraries will be downloaded and you can start your project.
+ But if you want to use jupyter notebook on the new environment you have to set it up for the new environment.
+ After you have installed all the libraries and created an environment, you need an editor to run the code, that is starting jupyter notebook, as soon as you enter jupyter notebook in the terminal you will definitely get this error. “Jupiter” is not recognized as an internal or external command.
So, to solve it it we have 2 commands.
1.	conda install –c conda-forge jupyterlab
2.	conda install –c anaconda python
Now you are ready to use jupyter on this environment and start with your project!
  
# _**Steps to execute**_
**Note:** Make sure you have added path while installing the software’s.

1.	Install the prerequisites/software’s required to execute the code.
2.	Press windows key and type in anaconda prompt a terminal opens up.
3.	Before executing the code, we need to create a specific environment which allows us to install the required libraries necessary for our project.
•	Type conda create -name “env_name”, e.g.: conda create -name project_1
•	Type conda activate “env_name, e.g.: conda activate project_1
4.	Make sure you are in the correct path in your terminal, where you have saved your executable file/folder. E.g.: cd A:\project\AI\Completed\project_name, then press enter.
5.	Install necessary libraries from requirements.txt file provided.
6.	Run pip install -r requirements.txt or conda install requirements.txt (Requirements.txt is a text file consisting of all the necessary libraries required for executing this python file. If it gives any error while installing libraries, you might need to install them individually.)
7.	Run sketch_to_image.ipynb and make sure to change the path where your executable files are located and also the model path in the code, please follow the link on how to install and set up anaconda environment to execute files.

# _**Data Description**_
We downloaded the dataset from kaggle. The dataset has been provided in the zip file along with the code. It contains sketch images of 80 objects. From this dataset we took images of the 10 objects mentioned above and used for training our model.

![image](https://user-images.githubusercontent.com/88571564/180369547-04ab80f0-d739-4380-9b44-6cfa8c39c3ae.png)

![image](https://user-images.githubusercontent.com/88571564/180369563-4db7a86d-e7c4-4070-9e4c-60961e2ab005.png)

![image](https://user-images.githubusercontent.com/88571564/180369577-135a94b8-23eb-45fe-967e-2a15731f3ff9.png)

# _**Issues Faced**_
1. We might face an issue while installing specific libraries.
2. Make sure you have the latest version of python or 3.8, since sometimes it might cause version mismatch.
3. Adding path to environment variables in order to run python files and anaconda environment in code editor, specifically in visual studio code.
4. Make sure you have the appropriate versions of tensorflow and keras.

#_**Results:**_
 
![image](https://user-images.githubusercontent.com/88571564/180369652-66dadc2f-5b2f-4bc9-9392-1b2be4dae953.png)

![image](https://user-images.githubusercontent.com/88571564/180369682-1f13e935-5494-45f3-bb8b-b188d24bcd07.png)

![image](https://user-images.githubusercontent.com/88571564/180369666-4423d56c-6642-4c20-a7a7-28ebbaf403db.png)

### _**Let’s Connect**_
<a href="https://linkedin.com/in/mudassiruddin21" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/linkedin.svg" alt="mudassiruddin21" height="30" width="40" /></a>

![Connect](https://media2.giphy.com/media/l1O6zvqu7O317887HF/source.gif)

# _**Yes, you now have more knowledge than yesterday, Keep Going.**_
![Happy](https://media.giphy.com/media/GK7grZYLG7cs0/giphy.gif)
  
