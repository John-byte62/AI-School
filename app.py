import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib

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
    We are living in an era where we definitly must have heard about the term AI . And many a time we might have wondered 
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
    
    st.write('But as we are focused on understanding the working of AI , ``Python`` is the most preferred language for it. Why python ? , because it is relatively easy to understand and it has a huge array of applications.Python is more intuitive than other programming dialects. And with a great community support. (I promise ,you can find solution for any of your python related issues :) ) ')

    st.subheader('Let deep dive into the world of Python. But before that lets play with an AI program , which can generate an imaginary image based on random values ')

    st.subheader('On the LHS, you can play with the sliders . But beware, selecting some of the features may produce some bias. This AI program used , a neural network which is known as ``` Transparent Latent-space GAN method ``` for tuning the output face characteristics ') 
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
