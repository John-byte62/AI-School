import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib
from PIL import Image

sys.path.append('tl_gan')
sys.path.append('pg_gan')
import feature_axis
import tfutil
import tfutil_cpu

# This should not be hashed by Streamlit when using st.cache.
TL_GAN_HASH_FUNCS = {
    tf.Session : id
}

def main():
    st.title("Welcome to the world of Programming & AI")

    """
    We are living in an era where we definitely must have heard about the term AI . And many a time we might have wondered 
    what exactly is this AI . And why it is getting discussed now a days. Lets find out

    As per the definition from Wikipedia , `Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals,
    which involves consciousness and emotionality.` In the recent years capability of AI has progressed enormously and its effect can be seen in every field . Chatbots , Amazon Alexa,
    Siri , Google Map , Robotics , Social Media , Airline industry are a small example where we can see the usage of AI.

    But but but... What's the starting point of AI ? 

    You might have guessed it right . It's programming . So what is programming ? Let us start. """

    st.header('What is programming ?')
    st.write('Programming is the process of creating a set of instructions that tell a computer how to perform a task. Programming can be done using a variety of computer programming languages, such as JavaScript, Python, and C++. And what is a programming language, it is just like a normal language , which we used to chat with computers. ')

    st.header('Which programming language is the best ?')
    st.write('Just as we in day to day life cannot choose , which spoken language is the best. Similarly in the world of computers, all the programming languages are beautiful.') 
    
    st.write('But as we are focused on understanding the working of AI , ``Python`` is the most preferred language for it. Why python ? , because it is relatively easy to understand and it has a huge array of applications.Python is more intuitive than other programming dialects. And with a great community support. (I promise ,you can find solution for any of your python related issues :blush: ) ')

    st.subheader('Let deep dive into the world of Python. But before that lets play with an AI program , which can generate an imaginary image based on random values ')

    st.write('On the LHS, you can play with the sliders . But beware, selecting some of the features may produce some **biases**. This AI program used , a neural network which is known as ``` Transparent Latent-space GAN method ``` for tuning the output face characteristics ') 
    st.write('  ')


    # Download all data files if they aren't already in the working directory.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Read in models from the data files.
    tl_gan_model, feature_names = load_tl_gan_model()
    session, pg_gan_model = load_pg_gan_model()

    st.sidebar.title('Features')
    seed = 27834096
    # If the user doesn't want to select which features to control, these will be used.
    default_control_features = ['Young','Smiling','Male']

    if st.sidebar.checkbox('Show advanced options'):
        # Randomly initialize feature values. 
        features = get_random_features(feature_names, seed)

        # Some features are badly calibrated and biased. Removing them
        block_list = ['Attractive', 'Big_Lips', 'Big_Nose', 'Pale_Skin']
        sanitized_features = [feature for feature in features if feature not in block_list]
        
        # Let the user pick which features to control with sliders.
        control_features = st.sidebar.multiselect( 'Control which features?',
            sorted(sanitized_features), default_control_features)
    else:
        features = get_random_features(feature_names, seed)
        # Don't let the user pick feature values to control.
        control_features = default_control_features
    
    # Insert user-controlled values from sliders into the feature vector.
    for feature in control_features:
        features[feature] = st.sidebar.slider(feature, 0, 100, 50, 5)


    st.sidebar.title('Note')
    st.sidebar.write(
        """Playing with the sliders, you _will_ find **biases** that exist in this model.
        """
    )
    st.sidebar.write(
        """For example, moving the `Smiling` slider can turn a face from masculine to feminine or from lighter skin to darker. 
        """
    )
    st.sidebar.write(
        """Apps like these that allow you to visually inspect model inputs help you find these biases so you can address them in your model _before_ it's put into production.
        """
    )


    # Generate a new image from this feature vector (or retrieve it from the cache).
    with session.as_default():
        image_out = generate_image(session, pg_gan_model, tl_gan_model,
                features, feature_names)

    st.image(image_out, width=500, use_column_width=False)

    st.header('Let us start with Python !! ')

    path = os.path.dirname(__file__)
    st.write(path)
    with st.beta_expander(label=' What is Python ?'):
        st.write('Python is a cross-platform programming language, which means that it can run on multiple platforms like Windows, macOS, Linux, and has even been ported to the Java and .NET virtual machines. It is free and open-source.')
        st.write('Even though most of today\'s Linux and Mac have Python pre-installed in it, the version might be out-of-date. So, it is always a good idea to install the most current version.')
    
    with st.beta_expander(label ='Python installation instructions. '):
        st.subheader('A. The Easiest Way to Run Python')
        st.write('The easiest way to run Python is by using Thonny IDE.')
        st.write('The **Thonny IDE** comes with the latest version of Python bundled in it. So you don\'t have to install Python separately.Follow the following steps to run Python on your computer.')
        st.write('  1.Download Thonny IDE.')
        st.write('  2.Run the installer to install Thonny on your computer.')
        st.write('  3.Go to: File > New. Then save the file with **.py** extension. For example, **hello.py**, **example.py** , etc.You can give any name to the file. However, the file name should end with .py')
        st.write('  4.Write Python code in the file and save it.')
        st.subheader('B. Run Python in the Integrated Development Environment (IDE)')
        st.write('We can use any text editing software to write a Python script file.')
        st.write('We just need to save it with the .py extension. But using an IDE can make our life a lot easier. IDE is a piece of software that provides useful features like code hinting, syntax highlighting and checking, file explorers, etc. to the programmer for application development.')
        st.write('By the way, when you install Python, an IDE named **IDLE** is also installed. You can use it to run Python on your computer. It\'s a decent IDE for beginners.Apart from that you can install other popular IDE like **Visual Studio , Pycharm , Spyder etc**')
        st.subheader('Your first Python Program')
        st.write('Now that we have Python up and running, we can write our first Python program.')
        st.write('Let\'s create a very simple program called Hello World. A "Hello, World!" is a simple program that outputs Hello, World! on the screen. Since it\'s a very simple program, it\'s often used to introduce a new programming language to beginners.')
        st.write('Type the following code') 
        st.write('**print(','\'Hello World\')**')
        st.write('in any text editor or an IDE and save it as **hello_world.py**','.Then, run the file. You will get the following output. **Hello World** ')
        st.write('*Congratulations!! You ran your first python program :thumbsup:*')

    with st.beta_expander(label ='Python Keywords and Identifier.'):
        st.subheader('Python Keywords')
        st.write('Keywords are the reserved words in Python.')
        st.write('We cannot use a keyword as a variable name, function name or any other identifier. They are used to define the syntax and structure of the Python language.')
        st.write('In Python, keywords are case sensitive.')
        st.write('There are 33 keywords in Python 3.7. This number can vary slightly over the course of time.')
        st.write('All the keywords except **True, False and None** are in lowercase and they must be written as they are. The list of all the keywords is given below.')
        st.image(Image.open('keywords.JPG'),width=650,use_column_width=False)
        st.write('Looking at all the keywords at once and trying to figure out what they mean might be overwhelming. So let us ignore it for some moment and let us focus on that we cannot use these names as any other identifier')
        st.subheader('Python Identifiers')
        st.write('An identifier is a name given to entities like class, functions, variables, etc. It helps to differentiate one entity from another.')
        st.write('Rules for writing identifiers')
        st.write('1.Identifiers can be a combination of letters in lowercase (a to z) or uppercase (A to Z) or digits (0 to 9) or an underscore _ . Names like **myClass, var_1 and print_this_to_screen**, all are valid example.')
        st.write('2.An identifier cannot start with a digit. **1variable** is invalid, but **variable1** is a valid name.')
        st.write('3.Keywords cannot be used as identifiers. ')
        st.write('4.We cannot use special symbols like !, @, #, $, etc. in our identifier. ')
        st.write('5.An identifier can be of any length.')
        st.subheader('Things to Remember')
        st.write('A. Python is a case-sensitive language. This means, **Variable** and **variable** are not the same.Multiple words can be separated using an underscore, like **this_is_a_long_variable.**')
        st.write('B. Most of the programming languages like C, C++, and Java use braces { } to define a block of code. Python, however, uses indentation.')
        st.write('C. Code block like a body of a function, loop, etc.**(All this will be discussed later)** starts with indentation and ends with the first unindented line. The amount of indentation is up to you, but it must be consistent throughout that block.')
        st.write('D. Comments are very important while writing a program. They describe what is going on inside a program, so that a person looking at the source code does not have a hard time figuring it out.Comments are for programmers to better understand a program. Python Interpreter ignores comments.In Python, we use the **hash (#)** symbol to start writing a comment.')
        st.write('E. For multi line comment , we can use Another way of doing this is to use triple quotes, either  \'\'\' or """. ')


    with st.beta_expander(label ='Python Variables and Datatypes.'):
        st.subheader('Python Variables')
        st.write('A variable is a named location used to store data in the memory. It is helpful to think of variables as a container that holds data that can be changed later in the program')
        st.write('For example , number = 10')
        st.write('Here, we have created a variable named **number**. We have assigned the value 10 to the variable. You can use the **assignment operator =** to assign a value to a variable.')
        st.write('You can think of variables as a bag to store books in it and that book can be replaced at any time. :smile:')
        st.write('number = 10')
        st.write('number = 1.1')
        st.write('Initially, the value of **number** was **10**. Later, it was changed to **1.1**. This reassignment can be done several times')
        st.write('Use **print(number)** command to print the value stored in the variable. Here **number** is the variable name. In this case **1.1** will be printed')
        st.write('***Note: In Python, we don\'t actually assign values to the variables. Instead, Python gives the reference of the object(value) to the variable.***')
        st.write('** You can assign multiple values to multiple variables** like ***a, b, c = 5, 3.2, "Hello"*** OR **x = y = z = "same"**')
        st.write("In first case , a will be 5 , b will be 3.2 and c='Hello' ")
        st.subheader('Literal')
        st.write('Literal is a raw data given in a variable or constant. In Python, there are various types of literals they are as follows:')
        st.write('**Numeric Literals** - Numeric Literals are immutable (unchangeable). Numeric literals can belong to 3 different numerical types: *Integer, Float, and Complex.*')
        st.write('**String literals** - A string literal is a sequence of characters surrounded by quotes. We can use both single, double, or triple quotes for a string. And, a character literal is a single character surrounded by single or double quotes.Example , *strings = "This is Python"*')
        st.write('**Boolean literals** - A Boolean literal can have any of the two values: True or False.In Python, True represents the value as 1 and False as 0')        
        st.write('**Special literals** - Python contains one special literal i.e. *None*. We use it to specify that the field has not been created.')
        st.write("**Literal Collections**-There are four different literal collections *List literals, Tuple literals, Dict literals, and Set literals.* We will deep dive into it , don't worry")
        st.subheader(' Decoding Literal Collections ')
        st.write("")
        st.write('**Python List**')
        st.write("List is an ordered sequence of items. It is one of the most used datatype in Python and is very flexible. All the items in a list do not need to be of the same type. Declaring a list is pretty straight forward. Items separated by commas are enclosed within brackets [ ]. Example **a = [1, 2.2, 'python']**.We can use the slicing operator [ ] to extract an item or a range of items from a list. The index starts from 0 in Python.")
        st.image(Image.open('list.JPG'),width=500,use_column_width=False)
        st.write('Lists are mutable, meaning, the value of elements of a list can be altered.')
        st.write("")
        st.write('**Python Tuple**')
        st.write("Tuple is an ordered sequence of items same as a list. The only difference is that tuples are immutable. Tuples once created cannot be modified.Tuples are used to write-protect data and are usually faster than lists as they cannot change dynamically.It is defined within parentheses () where items are separated by commas.We can use the slicing operator [] to extract items but we cannot change its value." )
        st.image(path+'/tuples.JPG',width=500,use_column_width=False)
        st.write("")
        st.write("**Python Set**")
        st.write("Set is an unordered collection of unique items. Set is defined by values separated by comma inside braces { }. Items in a set are not ordered.We can perform set operations like union, intersection on two sets. Sets have unique values. They eliminate duplicates.Since, set are unordered collection, indexing has no meaning. Hence, the slicing operator [] does not work.")
        st.image(path+'/set.JPG',width=500,use_column_width=False)
        st.write("")
        st.write("**Python Dictionary**")
        st.write("Dictionary is an unordered collection of key-value pairs.It is generally used when we have a huge amount of data. Dictionaries are optimized for retrieving data. We must know the key to retrieve the value.In Python, dictionaries are defined within braces \{\} with each item being a pair in the form key:value. Key and value can be of any type.")        
        st.image(path+'/dict.JPG',width=500,use_column_width=False)

    with st.beta_expander(label ='Python Operators '):
        st.header("What are operators in python?")
        st.write("Operators are special symbols in Python that carry out arithmetic or logical computation. The value that the operator operates on is called the operand.")
        st.write("For example : 2+3 will result in 5 .Here, **+** is the operator that performs addition. 2 and 3 are the operands and 5 is the output of the operation.")
        st.write('**Arithmetic operators**')
        st.write("Arithmetic operators are used to perform mathematical operations like addition, subtraction, multiplication, etc.")
        st.image(path+'/arith.JPG',width=600,use_column_width=False)
        st.write("")
        st.write("**Comparison operators**")
        st.write('Comparison operators are used to compare values. It returns either True or False according to the condition')
        st.image(path+'/comp.JPG',width=600,use_column_width=False)
        st.write("")
        st.write("**Logical operators**")
        st.write('Logical operators are the and, or, not operators.')
        st.image(path+'/log.JPG',width=600,use_column_width=False)
        st.write("")
        st.write("**Bitwise  operators**")
        st.write('Bitwise operators act on operands as if they were strings of binary digits. They operate bit by bit, hence the name.For example, 2 is 10 in binary and 7 is 111.In the table below: Let x = 10 (0000 1010 in binary) and y = 4 (0000 0100 in binary)')
        st.image(path+'/3bit.JPG',width=600,use_column_width=False)
        st.write("")
        st.write("**Assignment operators**")
        st.write("Assignment operators are used in Python to assign values to variables.a = 5 is a simple assignment operator that assigns the value 5 on the right to the variable a on the left.There are various compound operators in Python like a += 5 that adds to the variable and later assigns the same. It is equivalent to a = a + 5.")
        st.image('assign.JPG',width=600,use_column_width=False)
        st.write("")
        st.write("Python language offers some special types of operators like the identity operator or the membership operator. They are described below with examples.")
        st.write("**Identity operatorss :**  *is* and *is not* are the identity operators in Python. They are used to check if two values (or variables) are located on the same part of the memory. Two variables that are equal does not imply that they are identical. Kindly look at the below example ")
        st.image('iden.JPG',width=600,use_column_width=False)
        st.write('Here, we see that x1 and y1 are integers of the same values, so they are equal as well as identical. Same is the case with x2 and y2 (strings).But x3 and y3 are lists. They are equal but not identical. It is because the interpreter locates them separately in memory although they are equal.')
        st.write("")
        st.write("**Membership operators : ** *in and not in* are the membership operators in Python. They are used to test whether a value or variable is found in a sequence (string, list, tuple, set and dictionary).In a dictionary we can only test for presence of key, not the value.")
        st.image('mem.JPG',width=600,use_column_width=False)


    with st.beta_expander(label ='Python Flow Controls'):
        st.header("Python if...else Statement")
        st.write("Decision making is required when we want to execute a code only if a certain condition is satisfied.The **if…elif…else** statement is used in Python for decision making.")
        st.image('flowchartif.JPG',width=600,use_column_width=False)
        st.write("**Example of if...elif...else**")
        st.image('ifelse.JPG',width=600,use_column_width=False)
        st.write("When variable num is positive, **Positive number** is printed.")
        st.write("If num is equal to 0, **Zero** is printed.")
        st.write("If num is negative, **Negative number** is printed.")
        st.write("")
        st.write("**Python Nested if statements**")
        st.write("We can have a if...elif...else statement inside another if...elif...else statement. This is called nesting in computer programming.Any number of these statements can be nested inside one another. Indentation is the only way to figure out the level of nesting. They can get confusing, so they must be avoided unless necessary.")
        st.write("**Example of nested if...elif...else**")
        st.image('nestedif.JPG',width=600,use_column_width=False)
        st.header("Python for Loop")
        st.write("In this article, you'll learn to iterate over a sequence of elements using the different variations of for loop.")
        st.write("**What is for loop in Python?**")
        st.write("The *for* loop in Python is used to iterate over a sequence (list, tuple, string) or other iterable objects. Iterating over a sequence is called traversal.Loop continues until we reach the last item in the sequence. The body of for loop is separated from the rest of the code using indentation.")
        st.write("There is something known as **range() function** ,through which we can generate a sequence of numbers.range(10) will generate numbers from 0 to 9 (10 numbers).We can also define the start, stop and step size as range(start, stop,step_size). step_size defaults to 1 if not provided. This range() is extensively used with for loop ")
        st.write("**Example of for loop**")
        st.image('forloop.JPG',width=600,use_column_width=False)
        st.write("**for loop with else**")
        st.write("A for *loop* can have an optional *else* block as well. The else part is executed if the items in the sequence used in for loop exhausts.The *break* keyword can be used to stop a for loop. In such cases, the else part is ignored.Hence, *a for loop's else part runs if no break occurs.*")
        st.image('forelse.JPG',width=300,use_column_width=False)
        st.write("Here, the *for* loop prints items of the list until the loop exhausts. When the *for* loop exhausts, it executes the block of code in the *else* and prints *No items left.* Moreover a proper *if .. elif...else* block can also be nested inside for loop")
        st.header("Python while Loop")
        st.write("Loops are used in programming to repeat a specific block of code. In this article, you will learn to create a while loop in Python.")
        st.write("The while loop in Python is used to iterate over a block of code as long as the test expression (condition) is true.We generally use this loop when we don't know the number of times to iterate beforehand.")
        st.write("In the while loop, test expression is checked first. The body of the loop is entered only if the *test_expression* evaluates to *True*. After one iteration, the *test expression* is checked again. This process continues until the *test_expression* evaluates to *False*.In Python, the body of the while loop is determined through indentation.The body starts with indentation and the first unindented line marks the end.Python interprets any non-zero value as *True. None and 0* are interpreted as *False*.")
        st.write("Example: Python while Loop")
        st.image('while.JPG',width=600,use_column_width=False)
        st.write("In the above program, the test expression will be True as long as our counter variable i is less than or equal to n (10 in our program).We need to increase the value of the counter variable in the body of the loop. This is very important (and mostly forgotten). Failing to do so will result in an infinite loop (never-ending loop).Finally, the result is displayed.")
        st.write("**While loop with else**")
        st.write("Same as with for loops, while loops can also have an optional else block.The else part is executed if the condition in the while loop evaluates to False.The while loop can be terminated with a break statement. In such cases, the else part is ignored. Hence, a while loop's else part runs if no break occurs and the condition is false.")
        st.write("Example to illustrate this")
        st.image('whileelse.JPG',width=600,use_column_width=False)
        st.header("Python break , continue and pass")
        st.write("**What is the use of break and continue in Python?**")
        st.write("In Python, *break* and *continue* statements can alter the flow of a normal loop.Loops iterate over a block of code until the test expression is false, but sometimes we wish to terminate the current iteration or even the whole loop without checking test expression.The *break* and *continue* statements are used in these cases.")
        st.write("**break : **The break statement terminates the loop containing it. Control of the program flows to the statement immediately after the body of the loop.If the break statement is inside a nested loop (loop inside another loop), the break statement will terminate the innermost loop.")
        st.write("**continue :** The continue statement is used to skip the rest of the code inside a loop for the current iteration only. Loop does not terminate but continues on with the next iteration. ")
        st.write("**pass :** In Python programming, the pass statement is a null statement. The difference between a comment and a *pass* statement in Python is that while the interpreter ignores a *comment* entirely, *pass* is not ignored.Suppose we have a loop or a function that is not implemented yet, but we want to implement it in the future. They cannot have an empty body. The interpreter would give an error. So, we use the pass statement to construct a body that does nothing. ")

    with st.beta_expander(label ='Python Functions'):
        st.write("In Python, a function is a group of related statements that performs a specific task.Functions help break our program into smaller and modular chunks. As our program grows larger and larger, functions make it more organized and manageable.Furthermore, it avoids repetition and makes the code reusable.")
        st.write("**Syntax of Function**")
        st.image('func.JPG',width=300,use_column_width=False)
        st.write('Above shown is a function definition that consists of the following components.')
        st.write("1.Keyword *def* that marks the start of the function header.")
        st.write("2.A function name to uniquely identify the function. Function naming follows the same rules of writing identifiers in Python.")
        st.write("3.Parameters (arguments) through which we pass values to a function. They are optional.")
        st.write("4.A colon (:) to mark the end of the function header.")
        st.write("5.Optional documentation string (docstring) to describe what the function does.")
        st.write("6.One or more valid python statements that make up the function body. Statements must have the same indentation level (usually 4 spaces).")
        st.write("7.An optional *return* statement to return a value from the function.")
        st.write("**How to call a function in python?**")
        st.write('Once we have defined a function, we can call it from another function, program or even the Python prompt. To call a function we simply type the function name with appropriate parameters.')
        st.write("Below example illustrate how to create a function , define its functionality , returning of value and calling of the function")
        st.image('func_return.JPG',width=500,use_column_width=False)
        st.write("**Lambda Function **")
        st.write("Lambda functions can have any number of arguments but only one expression. The expression is evaluated and returned. Lambda functions can be used wherever function objects are required.")
        st.write("Example of lambda function")
        st.image('lamba.JPG',width=500,use_column_width=False)

    with st.beta_expander(label ='Food for thought. '):
        st.write("Apart from all the different topics that we have discussed above , there are many more areas in Python which we can explore. But as we are focusing on how AI program works, using the above knowledge we can now explore the basic nature of neural networks and AI. ")


    st.header('Let us explore the basic working of Neural Networks and AI !! ')
    st.write("**Artificial neural networks are a fascinating area of study, although they can be intimidating when just getting started.But do not worry I will try to make everything as simple as possible. So let start our journey ! **")
    with st.beta_expander("Multi-Layer Perceptrons"):
        st.write("The field of artificial neural networks is often just called neural networks or multi-layer perceptrons after perhaps the most useful type of neural network. A perceptron is a single neuron model that was a precursor to larger neural networks.")
        st.write("It is a field that investigates how simple models of biological brains can be used to solve difficult computational tasks like the predictive modeling tasks we see in machine learning. The goal is not to create realistic models of the brain, but instead to develop robust algorithms and data structures that we can use to model difficult problems.")
        st.write("The power of neural networks comes from their ability to learn the representation in your training data and how to best relate it to the output variable that you want to predict. In this sense neural networks learn a mapping. Mathematically, they are capable of learning any mapping function and have been proven to be a universal approximation algorithm.")
        st.write("The predictive capability of neural networks comes from the hierarchical or multi-layered structure of the networks. The data structure can pick out (learn to represent) features at different scales or resolutions and combine them into higher-order features. For example from lines, to collections of lines to shapes.")

    with st.beta_expander("Neurons"):
        st.write("The building block for neural networks are artificial neurons.")
        st.write("These are simple computational units that have weighted input signals and produce an output signal using an activation function.")
        st.image('neuron.JPG',width=300,use_column_width=False)
        st.write("**Neuron Weights**")
        st.write("Neuron Weights are nothing but a numerical value which gets attached to an input. These weights have the ability of the getting readjusted . And this behaviour of the weights help us to create awesome neural networks")
        st.write("For example ,suppose we are passing an input of 10 , then weight can be 0.1 or anything random. That means value of 10*0.1 i.e 1 is passed to next neurons")

        st.write("**Activation Functions**")
        st.write("The weighted inputs are summed and passed through an activation function, sometimes called a transfer function.")
        st.write("An activation function is a simple mapping of summed weighted input to the output of the neuron. It is called an activation function because it governs the threshold at which the neuron is activated and strength of the output signal.")
        st.write("Historically simple step activation functions were used where if the summed input was above a threshold, for example 0.5, then the neuron would output a value of 1.0, otherwise it would output a 0.0.")
        st.write("Traditionally non-linear activation functions are used. This allows the network to combine the inputs in more complex ways and in turn provide a richer capability in the functions they can model. Non-linear functions like the logistic also called the sigmoid function were used that output a value between 0 and 1 with an s-shaped distribution, and the hyperbolic tangent function also called tanh that outputs the same distribution over the range -1 to +1.More recently the rectifier activation function has been shown to provide better results.")

    with st.beta_expander("Networks of Neurons"):
        st.write("Neurons are arranged into networks of neurons.A row of neurons is called a layer and one network can have multiple layers. The architecture of the neurons in the network is often called the network topology.")
        st.image('nn.JPG',width=300,use_column_width=False)
        st.write("**Input or Visible Layers**")
        st.write("The bottom layer that takes input from your dataset is called the visible layer, because it is the exposed part of the network. Often a neural network is drawn with a visible layer with one neuron per input value or column in your dataset. These are not neurons as described above, but simply pass the input value though to the next layer.")
        st.write("**Hidden Layers**")
        st.write("Layers after the input layer are called hidden layers because that are not directly exposed to the input. The simplest network structure is to have a single neuron in the hidden layer that directly outputs the value.")
        st.write("Given increases in computing power and efficient libraries, very deep neural networks can be constructed. Deep learning can refer to having many hidden layers in your neural network. They are deep because they would have been unimaginably slow to train historically, but may take seconds or minutes to train using modern techniques and hardware.")
        st.write("**Output Layer**")
        st.write("The final hidden layer is called the output layer and it is responsible for outputting a value or vector of values that correspond to the format required for the problem.The choice of activation function in he output layer is strongly constrained by the type of problem that you are modeling")

    with st.beta_expander("Training of Networks"):    
        st.write("Once configured, the neural network needs to be trained on your dataset")
        st.write("**Data Preparation**")
        st.write("You must first prepare your data for training on a neural network. Data must be numerical, for example real values. If you have categorical data, such as a sex attribute with the values “male” and “female”, you can convert it to a real-valued representation called a *one hot encoding*. This is where one new column is added for each class value (two columns in the case of sex of male and female) and a 0 or 1 is added for each row depending on the class value for that row.")
        st.write("This same one hot encoding can be used on the output variable in classification problems with more than one class. This would create a binary vector from a single column that would be easy to directly compare to the output of the neuron in the network’s output layer, that as described above, would output one value for each class.")
        st.write("Neural networks require the input to be scaled in a consistent way. You can rescale it to the range between 0 and 1 called normalization. Another popular technique is to standardize it so that the distribution of each column has the mean of zero and the standard deviation of 1.")
        st.write("Scaling also applies to image pixel data. Data such as words can be converted to integers, such as the popularity rank of the word in the dataset and other encoding techniques.")
        
        st.write("** Stochastic Gradient Descent **")
        st.write("The classical and still preferred training algorithm for neural networks is called stochastic gradient descent.")
        st.write("This is where one row of data is exposed to the network at a time as input. The network processes the input upward activating neurons as it goes to finally produce an output value. This is called a forward pass on the network. It is the type of pass that is also used after the network is trained in order to make predictions on new data.")
        st.write("The output of the network is compared to the expected output and an error is calculated. This error is then propagated back through the network, one layer at a time, and the weights are updated according to the amount that they contributed to the error. This clever bit of math is called the backpropagation algorithm.")
        st.write("The process is repeated for all of the examples in your training data. One round of updating the network for the entire training dataset is called an epoch. A network may be trained for tens, hundreds or many thousands of epochs.")
        
        st.write("** Weight Updates **")
        st.write("The weights in the network can be updated from the errors calculated for each training example and this is called online learning. It can result in fast but also chaotic changes to the network.")
        st.write("Alternatively, the errors can be saved up across all of the training examples and the network can be updated at the end. This is called batch learning and is often more stable.")
        st.write("Typically, because datasets are so large and because of computational efficiencies, the size of the batch, the number of examples the network is shown before an update is often reduced to a small number, such as tens or hundreds of examples.")
        st.write("The amount that weights are updated is controlled by a configuration parameters called the learning rate. It is also called the step size and controls the step or change made to network weight for a given error. Often small weight sizes are used such as 0.1 or 0.01 or smaller.")
        st.write("The update equation can be complemented with additional configuration terms that you can set.  ")
        st.write("1.Momentum is a term that incorporates the properties from the previous weight update to allow the weights to continue to change in the same direction even when there is less error being calculated.")
        st.write("2.Learning Rate Decay is used to decrease the learning rate over epochs to allow the network to make large changes to the weights at the beginning and smaller fine tuning changes later in the training schedule.")

        st.write("** Prediction **")
        st.write("Once a neural network has been trained it can be used to make predictions.")
        st.write("You can make predictions on test or validation data in order to estimate the skill of the model on unseen data. You can also deploy it operationally and use it to make predictions continuously.")
        st.write("The network topology and the final set of weights is all that you need to save from the model. Predictions are made by providing the input to the network and performing a forward-pass allowing it to generate an output that you can use as a prediction.")

    st.write('## Summary')
    st.write(" So we just completed the basics of Python & Neural Networks.Below are the points that we covered in the blog")
    st.write("1.Basics of Python language & different concepts related to coding like loops,if..else statement,functions etc.")
    st.write("2.How neural networks are not models of the brain but are instead computational models for solving complex machine learning problems.")
    st.write("3.The neural networks are comprised of neurons that have weights and activation functions.")
    st.write("4.The networks are organized into layers of neurons and are trained using stochastic gradient descent.Moreover it is a good idea to prepare your data before training a neural network model.")
    st.write("***The field of AI , Machine Learning , Data Science is very vast, but they are one of the most interesting field in today's era. I hope you all will try to deep dive into this field and will make the best out of it ***")
    st.write("Do you have any questions about neural networks or about this post? You can connect with me using below mentioned links and I will do my best to answer it. **Stay Safe , Stay Healthy , Stay Strong !!** ")
    
    st.write("##### Fiver profile: https://www.fiverr.com/riteshsingh545")
    st.write("##### Upwork profile: https://www.upwork.com/freelancers/~01825a78e7bea22084")
    st.write("##### LinkedIn profile: https://www.linkedin.com/in/ritesh-singh-6619ab190/")
    st.write("")
    st.write("")
    st.write("")
    st.image('end.JPG',width=800,use_column_width=False)
    st.write("**----------------------------------------------------------END----------------------------------------------------------**")

    
    

        
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

# Ensure that load_pg_gan_model is called only once, when the app first loads.
@st.cache(allow_output_mutation=True, hash_funcs=TL_GAN_HASH_FUNCS)
def load_pg_gan_model():
    """
    Create the tensorflow session.
    """
    # Open a new TensorFlow session.
    config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=config)

    # Must have a default TensorFlow session established in order to initialize the GAN.
    with session.as_default():
        # Read in either the GPU or the CPU version of the GAN
        with open(MODEL_FILE_GPU if USE_GPU else MODEL_FILE_CPU, 'rb') as f:
            G = pickle.load(f)
    return session, G

# Ensure that load_tl_gan_model is called only once, when the app first loads.
@st.cache(hash_funcs=TL_GAN_HASH_FUNCS)
def load_tl_gan_model():
    """
    Load the linear model (matrix) which maps the feature space
    to the GAN's latent space.
    """
    with open(FEATURE_DIRECTION_FILE, 'rb') as f:
        feature_direction_name = pickle.load(f)

    # Pick apart the feature_direction_name data structure.
    feature_direction = feature_direction_name['direction']
    feature_names = feature_direction_name['name']
    num_feature = feature_direction.shape[1]
    feature_lock_status = np.zeros(num_feature).astype('bool')

    # Rearrange feature directions using Shaobo's library function.
    feature_direction_disentangled = \
        feature_axis.disentangle_feature_axis_by_idx(
            feature_direction,
            idx_base=np.flatnonzero(feature_lock_status))
    return feature_direction_disentangled, feature_names

def get_random_features(feature_names, seed):
    """
    Return a random dictionary from feature names to feature
    values within the range [40,60] (out of [0,100]).
    """
    np.random.seed(seed)
    features = dict((name, 40+np.random.randint(0,21)) for name in feature_names)
    return features

# Hash the TensorFlow session, the pg-GAN model, and the TL-GAN model by id
# to avoid expensive or illegal computations.
@st.cache(show_spinner=False, hash_funcs=TL_GAN_HASH_FUNCS)
def generate_image(session, pg_gan_model, tl_gan_model, features, feature_names):
    """
    Converts a feature vector into an image.
    """
    # Create rescaled feature vector.
    feature_values = np.array([features[name] for name in feature_names])
    feature_values = (feature_values - 50) / 250
    # Multiply by Shaobo's matrix to get the latent variables.
    latents = np.dot(tl_gan_model, feature_values)
    latents = latents.reshape(1, -1)
    dummies = np.zeros([1] + pg_gan_model.input_shapes[1][1:])
    # Feed the latent vector to the GAN in TensorFlow.
    with session.as_default():
        images = pg_gan_model.run(latents, dummies)
    # Rescale and reorient the GAN's output to make an image.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0),
                              0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
    if USE_GPU:
        images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    return images[0]

USE_GPU = False
FEATURE_DIRECTION_FILE = "feature_direction_2018102_044444.pkl"
MODEL_FILE_GPU = "karras2018iclr-celebahq-1024x1024-condensed.pkl"
MODEL_FILE_CPU = "karras2018iclr-celebahq-1024x1024-condensed-cpu.pkl"
EXTERNAL_DEPENDENCIES = {
    "feature_direction_2018102_044444.pkl" : {
        "url": "https://streamlit-demo-data.s3-us-west-2.amazonaws.com/facegan/feature_direction_20181002_044444.pkl",
        "size": 164742
    },
    "karras2018iclr-celebahq-1024x1024-condensed.pkl": {
        "url": "https://streamlit-demo-data.s3-us-west-2.amazonaws.com/facegan/karras2018iclr-celebahq-1024x1024-condensed.pkl",
        "size": 92338293
    },
    "karras2018iclr-celebahq-1024x1024-condensed-cpu.pkl": {
        "url": "https://streamlit-demo-data.s3-us-west-2.amazonaws.com/facegan/karras2018iclr-celebahq-1024x1024-condensed-cpu.pkl",
        "size": 92340233
    }
}

if __name__ == "__main__":
    main()
