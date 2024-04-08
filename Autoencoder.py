
import streamlit as st

# Define the Streamlit app
def app():
    st.header("Convolutional Autoencoders for Image Colorization")

    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Department of Computer Science
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('autoencoder.png', caption="Convolutional Neural Network Autoencoder")

    text = """Convolutional Neural Networks (CNNs) can be leveraged to construct 
    powerful autoencoders specifically suited for image data. These architectures 
    consist of two primary components: an encoder and a decoder.
    \nEncoder: The encoder utilizes a series of convolutional layers interspersed with 
    downsampling operations (e.g., pooling). These layers progressively extract high-level 
    features from the input grayscale image, capturing essential information about 
    spatial relationships and patterns. Notably, the final convolutional layer often 
    employs a lower dimensionality compared to the input, forming a compressed latent 
    representation that embodies the core characteristics of the image.
    \nDecoder: The decoder takes the latent representation generated by the encoder and aims 
    to reconstruct the original image. It leverages a sequence of convolutional layers with 
    upsampling techniques (e.g., transposed convolutions). These layers progressively 
    increase the spatial resolution of the feature maps, ultimately generating a reconstructed 
    image in the original color space. The decoder network is trained to minimize the 
    reconstruction error between the output and the original input image.
    \nBy training the autoencoder on a large collection of grayscale landscape images 
    paired with their corresponding colored counterparts, the decoder network learns a mapping 
    function. This mapping allows it to transform a low-dimensional latent representation, 
    encoding grayscale information, into a high-dimensional output representing a colorized 
    version of the image."""
    st.write(text)

    text = """Traditional image colorization techniques often necessitate significant  human 
    intervention, requiring expertise, time, and effort. However, recent advancements in deep 
    learning architectures, particularly convolutional neural networks (CNNs) configured as 
    autoencoders, have offered a promising approach for automating this process. 
    Autoencoders possess the remarkable capability to learn the underlying, essential 
    characteristics of an image and subsequently reconstruct it based on these acquired 
    features. This enables them to effectively colorize grayscale images, streamlining the 
    process and reducing human involvement."""

    st.write(text)
    st.image('colorization.jpg', caption="Image Colorization Task")
#run the app
if __name__ == "__main__":
    app()
