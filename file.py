import streamlit as st
from skimage import io, color, morphology, measure, segmentation
from skimage.filters import sobel
from skimage.segmentation import active_contour
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("Aplikasi Manipulasi Gambar dengan Streamlit dan Scikit-Image")

    # Upload Gambar
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = io.imread(uploaded_file)
        gray_image = color.rgb2gray(image)
        st.image(image, caption="Gambar Asli", use_column_width=True)

        st.subheader("Operasi Gambar")

        operation = st.selectbox("Pilih operasi", ["Convex Hull", "Skeletonization", "Active Contour"])

        if operation == "Convex Hull":
            chull = morphology.convex_hull_image(gray_image)
            st.image(chull, caption="Convex Hull", use_column_width=True)

        elif operation == "Skeletonization":
            skeleton = morphology.skeletonize(gray_image)
            st.image(skeleton, caption="Skeletonization", use_column_width=True)

        elif operation == "Active Contour":
            init = np.array([[int(gray_image.shape[0]/5), int(gray_image.shape[1]/5)],
                             [int(gray_image.shape[0]/5), int(4*gray_image.shape[1]/5)],
                             [int(4*gray_image.shape[0]/5), int(4*gray_image.shape[1]/5)],
                             [int(4*gray_image.shape[0]/5), int(gray_image.shape[1]/5)]])
            snake = active_contour(sobel(gray_image), init, alpha=0.015, beta=10, gamma=0.001)
            fig, ax = plt.subplots()
            ax.imshow(gray_image, cmap=plt.cm.gray)
            ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
            ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig)

if __name__ == "__main__":
    main()
